from __future__ import annotations

import torch

from fairchem.core import FAIRChemCalculator
from fairchem.core.common.utils import conditional_grad, scatter_det
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.gemnet_oc.gemnet_oc import GemNetOC
from fairchem.core.models.gemnet_oc.utils import repeat_blocks


@conditional_grad(torch.enable_grad())
def newforward(self, data):
    pos = data.pos
    batch = data.batch
    atomic_numbers = data.atomic_numbers.long()
    num_atoms = atomic_numbers.shape[0]

    if self.regress_forces and not self.direct_forces:
        pos.requires_grad_(True)

    (
        main_graph,
        a2a_graph,
        a2ee2a_graph,
        qint_graph,
        id_swap,
        trip_idx_e2e,
        trip_idx_a2e,
        trip_idx_e2a,
        quad_idx,
    ) = self.get_graphs_and_indices(data)
    _, idx_t = main_graph["edge_index"]

    (
        basis_rad_raw,
        basis_atom_update,
        basis_output,
        bases_qint,
        bases_e2e,
        bases_a2e,
        bases_e2a,
        basis_a2a_rad,
    ) = self.get_bases(
        main_graph=main_graph,
        a2a_graph=a2a_graph,
        a2ee2a_graph=a2ee2a_graph,
        qint_graph=qint_graph,
        trip_idx_e2e=trip_idx_e2e,
        trip_idx_a2e=trip_idx_a2e,
        trip_idx_e2a=trip_idx_e2a,
        quad_idx=quad_idx,
        num_atoms=num_atoms,
    )

    # Embedding block
    h = self.atom_emb(atomic_numbers)
    # (nAtoms, emb_size_atom)
    m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])
    # (nEdges, emb_size_edge)

    x_E, x_F = self.out_blocks[0](h, m, basis_output, idx_t)
    # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
    xs_E, xs_F = [x_E], [x_F]

    for i in range(self.num_blocks):
        # Interaction block
        h, m = self.int_blocks[i](
            h=h,
            m=m,
            bases_qint=bases_qint,
            bases_e2e=bases_e2e,
            bases_a2e=bases_a2e,
            bases_e2a=bases_e2a,
            basis_a2a_rad=basis_a2a_rad,
            basis_atom_update=basis_atom_update,
            edge_index_main=main_graph["edge_index"],
            a2ee2a_graph=a2ee2a_graph,
            a2a_graph=a2a_graph,
            id_swap=id_swap,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
        )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

        x_E, x_F = self.out_blocks[i + 1](h, m, basis_output, idx_t)
        # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
        xs_E.append(x_E)
        xs_F.append(x_F)

    # Global output block for final predictions
    x_E = self.out_mlp_E(torch.cat(xs_E, dim=-1))
    if self.direct_forces:
        x_F = self.out_mlp_F(torch.cat(xs_F, dim=-1))
    with torch.cuda.amp.autocast(False):
        E_t = self.out_energy(x_E.float())
        if self.direct_forces:
            F_st = self.out_forces(x_F.float())

    nMolecules = torch.max(batch) + 1
    if self.extensive:
        E_t = scatter_det(
            E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
        )  # (nMolecules, num_targets)
    else:
        E_t = scatter_det(
            E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
        )  # (nMolecules, num_targets)

    E_t = E_t.squeeze(1)  # (num_molecules)
    outputs = {"energy": E_t}
    if self.regress_forces:
        if self.direct_forces:
            if self.forces_coupled:  # enforce F_st = F_ts
                nEdges = idx_t.shape[0]
                id_undir = repeat_blocks(
                    main_graph["num_neighbors"] // 2,
                    repeats=2,
                    continuous_indexing=True,
                )
                F_st = scatter_det(
                    F_st,
                    id_undir,
                    dim=0,
                    dim_size=int(nEdges / 2),
                    reduce="mean",
                )  # (nEdges/2, num_targets)
                F_st = F_st[id_undir]  # (nEdges, num_targets)

            # map forces in edge directions
            F_st_vec = F_st[:, :, None] * main_graph["vector"][:, None, :]
            # (nEdges, num_targets, 3)
            F_t = scatter_det(
                F_st_vec,
                idx_t,
                dim=0,
                dim_size=num_atoms,
                reduce="add",
            )  # (nAtoms, num_targets, 3)
        else:
            F_t = self.force_scaler.calc_forces_and_update(E_t, pos)

        F_t = F_t.squeeze(1)  # (num_atoms, 3)

        outputs["forces"] = F_t

    # This is the section I adapted from abishek's code
    if hasattr(self, "return_embedding") and self.return_embedding:
        nMolecules = (torch.max(batch) + 1).item()

        # This seems to be an earlier block
        outputs["h sum"] = scatter_det(
            h, batch, dim=0, dim_size=nMolecules, reduce="add"
        )

        # These embedding are closer to energy output
        outputs["x_E sum"] = scatter_det(
            x_E, batch, dim=0, dim_size=nMolecules, reduce="add"
        )

        # This is an embedding related to forces.
        # Something seems off on I couldn't do the same thing as above with scatter_det.
        outputs["x_F sum"] = torch.sum(x_F, axis=0)[None, :]

        # tuples with nMolecules tensors of size nAtoms x embedding_size.
        outputs["x_E"] = x_E.split(data.natoms.tolist(), dim=0)

        outputs["h"] = h.split(data.natoms.tolist(), dim=0)

    return outputs


GemNetOC.forward = newforward


def embed(self, atoms):
    self.trainer._unwrapped_model.return_embedding = True
    data_object = self.a2g.convert(atoms)
    batch_list = data_list_collater([data_object], otf_graph=True)

    self.trainer.model.eval()
    if self.trainer.ema:
        self.trainer.ema.store()
        self.trainer.ema.copy_to()

    with (
        torch.cuda.amp.autocast(enabled=self.trainer.scaler is not None),
        torch.no_grad(),
    ):
        out = self.trainer.model(batch_list)

    if self.trainer.normalizers is not None and "target" in self.trainer.normalizers:
        out["energy"] = self.trainer.normalizers["target"].denorm(out["energy"])
        out["forces"] = self.trainer.normalizers["grad_target"].denorm(out["forces"])

    if self.trainer.ema:
        self.trainer.ema.restore()

    return out


FAIRChemCalculator.embed = embed
