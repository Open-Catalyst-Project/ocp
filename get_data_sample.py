"""
Exploration file to get a `batch` in-memory and play around with it.
Use it in notebooks or ipython console

$ ipython
...
In [1]: run get_data_sample.py
Out[1]: ...

In [2]: print(batch)

"""
import sys
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401
from minydra import resolved_args
from torch import cat, isin, tensor, where
from torch_geometric.utils import remove_self_loops, sort_edge_index
from tqdm import tqdm

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports, setup_logging
from ocpmodels.preprocessing import (
    one_supernode_per_atom_type,
    one_supernode_per_atom_type_dist,
    one_supernode_per_graph,
    remove_tag0_nodes,
)

if __name__ == "__main__":

    opts = resolved_args()

    sys.argv[1:] = ["--mode=train", "--config=configs/is2re/10k/schnet/schnet.yml"]
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    config["optim"]["num_workers"] = 4
    config["optim"]["batch_size"] = 3
    config["logger"] = "dummy"

    if opts.victor_local:
        config["dataset"][0]["src"] = "data/is2re/All/train/data.lmdb"
        config["dataset"] = config["dataset"][:1]
        config["optim"]["num_workers"] = 0
        config["optim"]["batch_size"] = opts.bs or config["optim"]["batch_size"]

    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model_attributes=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 100),
        seed=config.get("seed", 0),
        logger=config.get("logger", "wandb"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        new_gnn=config.get("new_gnn", True),
        data_split=config.get("data_split", None),
        note=config.get("note", ""),
    )

    task = registry.get_task_class(config["mode"])(config)
    task.setup(trainer)

    if opts.no_frame_averaging is None:
        for batch in trainer.train_loader:
            break
        # Set up 
        b = batch[0]
        device = b.pos.device 
        batch_size = b.sid.shape[0]
        for g in b: 
            num_atoms = g.pos.shape[0]
            # Compute centroid and covariance
            t_ones = torch.ones(num_atoms).unsqueeze(1)
            t = 1/num_atoms * g.pos.T @ t_ones  # , device=device
            C = (g.pos - t_ones @ t.T).T @ (g.pos - t_ones @ t.T)
            # Eigendecomposition
            eigenval, eigenvec = torch.linalg.eig(C)
            # Sort, if necessary 
            idx = eigenval.real.argsort(descending=True)
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:,idx]
            # Compute new positions
            g.updated_pos = (g.pos - t.squeeze()) @ eigenvec.real
        # Extensions to 2/3 possible U
        # Look at positive U only




    if opts.no_single_super_node is None:

        for batch in trainer.train_loader:
            break
        b = batch[0]
        b_bis = deepcopy(b)
        b_bisbis = deepcopy(b)
        data_bis_bis = one_supernode_per_graph(b_bisbis)
        data_bis = one_supernode_per_atom_type(b_bis)
        data = one_supernode_per_atom_type_dist(b)
        assert data == data_bis

        # final object that would be returned in a proper function
        data = deepcopy(b)
        # start time
        t0 = time()

        # Call function from graph_rewiring

        batch_size = max(b.batch).item() + 1
        device = b.edge_index.device

        # ids of sub-surface nodes, per batch
        sub_nodes = [
            where((b.tags == 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # single tensor of all the sub-surface nodes
        # all_sub_nodes = torch.cat(sub_nodes)

        # idem for non-sub-surface nodes
        non_sub_nodes = [
            where((b.tags != 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # all_non_sub_nodes = torch.cat(non_sub_nodes)

        # super node index per batch: they are last in their batch
        new_sn_ids = [
            sum([len(nsn) for nsn in non_sub_nodes[: i + 1]]) + i
            for i in range(batch_size)
        ]
        data.ptr = tensor(
            [0] + [nsi + 1 for nsi in new_sn_ids], dtype=b.ptr.dtype, device=device
        )
        data.natoms = data.ptr[1:] - data.ptr[:-1]

        # number of aggregated nodes into the super node, per batch
        data.sn_nodes_aggregates = tensor([len(s) for s in sub_nodes], device=device)
        # super node position for a batch is the mean of its aggregates
        sn_pos = [b.pos[sub_nodes[i]].mean(0) for i in range(batch_size)]
        # target relaxed position is the mean of the super node's aggregates
        # (per batch)
        sn_pos_relaxed = [
            b.pos_relaxed[sub_nodes[i]].mean(0) for i in range(batch_size)
        ]
        # the force applied on the super node is the mean of the force applied
        # to its aggregates (per batch)
        sn_force = [b.force[sub_nodes[i]].mean(0) for i in range(batch_size)]

        # per-atom tensors

        # SNs are last in their batch
        data.atomic_numbers = cat(
            [
                cat([b.atomic_numbers[non_sub_nodes[i]], tensor([84], device=device)])
                for i in range(batch_size)
            ]
        )

        # all super nodes have atomic number -1
        # assert all([data.atomic_numbers[s].cpu().item() == -1 for s in new_sn_ids])

        # position exclude the sub-surface atoms but include an extra super-node
        data.pos = cat(
            [
                cat([b.pos[non_sub_nodes[i]], sn_pos[i][None, :]])
                for i in range(batch_size)
            ]
        )
        data.pos_relaxed = cat(
            [
                cat([b.pos_relaxed[non_sub_nodes[i]], sn_pos_relaxed[i][None, :]])
                for i in range(batch_size)
            ]
        )
        # idem
        data.force = cat(
            [
                cat([b.force[non_sub_nodes[i]], sn_force[i][None, :]])
                for i in range(batch_size)
            ]
        )
        data.fixed = cat(
            [
                cat(
                    [
                        b.fixed[non_sub_nodes[i]],
                        tensor([1.0], dtype=b.fixed.dtype, device=device),
                    ]
                )
                for i in range(batch_size)
            ]
        )
        data.tags = cat(
            [
                cat(
                    [
                        b.tags[non_sub_nodes[i]],
                        tensor([0], dtype=b.tags.dtype, device=device),
                    ]
                )
                for i in range(batch_size)
            ]
        )

        expensive_ops_time = [time()]
        # edge indices per batch
        # 53ms (128)
        ei_batch_ids = [
            (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
            for i in range(batch_size)
        ]
        expensive_ops_time.append(time())
        print(f"ei_batch_ids: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")
        # edges per batch
        # 78ms (128)
        ei_batch = [b.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]
        expensive_ops_time.append(time())
        print(f"ei_batch: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")
        co_batch = [b.cell_offsets[ei_batch_ids[i], :] for i in range(batch_size)]
        expensive_ops_time.append(time())
        print(f"co_batch: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")

        # boolean src node is not sub per batch
        # 58ms (128)
        src_is_not_sub = [
            isin(b.edge_index[0][ei_batch_ids[i]], ns)
            for i, ns in enumerate(non_sub_nodes)
        ]
        expensive_ops_time.append(time())
        print(f"src_is_not_sub: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")
        # boolean target node is not sub per batch
        # 60ms (128)
        target_is_not_sub = [
            isin(b.edge_index[1][ei_batch_ids[i]], ns)
            for i, ns in enumerate(non_sub_nodes)
        ]
        expensive_ops_time.append(time())
        print(
            f"target_is_not_sub: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s"
        )
        # edges for which both nodes are below the surface
        both_are_sub = [~s & ~t for s, t in zip(src_is_not_sub, target_is_not_sub)]
        expensive_ops_time.append(time())
        print(f"both_are_sub: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")
        # edges for which NOT both nodes are below the surface
        not_both_are_sub = [~bas for bas in both_are_sub]
        expensive_ops_time.append(time())
        print(
            f"not_both_are_sub: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s"
        )
        # ^ turn into [(s|t) for s, t in zip(src_is_not_sub, target_is_not_sub)]
        # when both_are_sub is deleted

        # number of edges that end-up being removed
        data.sn_edges_aggregates = tensor(
            [len(n) - n.sum() for n in not_both_are_sub], device=device
        )
        expensive_ops_time.append(time())
        print(
            f"edges_aggregates: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s"
        )
        data.cell_offsets = cat(
            [
                cat([co_batch[i][not_both_are_sub[i]], tensor([[0, 0, 0]])])
                for i in range(batch_size)
            ]
        )
        expensive_ops_time.append(time())
        print(f"cell_offsets: {expensive_ops_time[-1] - expensive_ops_time[-2]:.3f}s")

        # -----------------------------
        # -----  Graph re-wiring  -----
        # -----------------------------
        vt = time()
        all_not_both_are_sub = cat(not_both_are_sub)
        all_non_sub_nodes = cat(non_sub_nodes)
        all_sub_nodes = cat(sub_nodes)
        # future, super-node-adjusted edge index
        ei_not_both = b.edge_index.clone()[:, all_not_both_are_sub]
        # number of nodes in this batch: all existing + 1 super node
        num_nodes = b.ptr[-1].item() + batch_size
        # mask to reindex the edges
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=b.edge_index.device)
        # mask is 1 for non-sub nodes
        mask[all_non_sub_nodes] = 1
        # lookup table
        assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
        assoc[mask] = cat(
            [
                torch.arange(data.ptr[e], data.ptr[e + 1] - 1, device=assoc.device)
                for e in range(batch_size)
            ]
        )

        # new values for non-sub-indices
        # assert (assoc[batch_sub_nodes] == -1).all()
        # assert (assoc[mask] != -1).all()

        # re-index edges ; select only the edges for which not
        # both nodes are sub-surface atoms
        ei_sn = assoc[ei_not_both]

        distance = torch.sqrt(
            ((data.pos[ei_sn[0, :]] - data.pos[ei_sn[1, :]]) ** 2).sum(-1)
        )

        vt = time() - vt

        data.edge_index = ei_sn.to(dtype=b.edge_index.dtype)
        data.distances = distance.to(dtype=b.distances.dtype)
        _, data.neighbors = torch.unique(
            data.batch[data.edge_index[0, :]], return_counts=True
        )

        data.batch = torch.zeros(data.ptr[-1], dtype=b.batch.dtype, device=device)
        for i, p in enumerate(data.ptr[:-1]):
            data.batch[torch.arange(p, data.ptr[i + 1], dtype=torch.long)] = tensor(
                i, dtype=b.batch.dtype
            )
        tf = time()

        n_e_total = b.neighbors.sum()
        n_e_kept = data.neighbors.sum()
        n_e_removed = n_e_total - n_e_kept
        n_both = [b.sum().item() for b in both_are_sub]
        print(f"Total edges {n_e_total} | kept {n_e_kept} | removed {n_e_removed}")
        print(f"Average ratio kept {n_e_kept/n_e_total}")
        print(
            "Total ei rewiring processing time (batch size",
            f"{batch_size}) {vt:.3f}",
        )
        print(f"Total processing time: {tf-t0:.5f}")
        print(f"Total processing time per batch: {(tf-t0) / batch_size:.5f}")

    # check conditionno_several_super_node
    if opts.no_several_super_node is None:

        for batch in trainer.train_loader:
            break
        b = batch[0]
        data = deepcopy(b)
        t0 = time()

        batch_size = max(b.batch).item() + 1
        device = b.edge_index.device

        # ids of sub-surface nodes, per batch
        sub_nodes = [
            torch.where((b.tags == 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # idem for non-sub-surface nodes
        non_sub_nodes = [
            torch.where((b.tags != 0) * (b.batch == i))[0] for i in range(batch_size)
        ]
        # atom types per supernode
        atom_types = [
            torch.unique(b.atomic_numbers[(b.tags == 0) * (b.batch == i)])
            for i in range(batch_size)
        ]
        # number of supernodes per batch
        num_supernodes = [atom_types[i].shape[0] for i in range(batch_size)]
        total_num_supernodes = sum(num_supernodes)
        # indexes of nodes belonging to each supernode
        supernodes_composition = [
            [
                torch.where((b.atomic_numbers == an) * (b.tags == 0) * (b.batch == i))[
                    0
                ]
                for an in atom_types[i]
            ]
            for i in range(batch_size)
        ]
        # supernode indexes
        sn_idxes = [
            [b.ptr[1:][i] + sn for sn in range(num_supernodes[i])]
            for i in range(len(num_supernodes))
        ]

        # supernode positions
        supernodes_pos = [
            b.pos[sn, :][0] for sublist in supernodes_composition for sn in sublist
        ]

        ### Compute supernode edge-index
        ei_batch_ids = [
            (b.ptr[i] <= b.edge_index[0]) * (b.edge_index[0] < b.ptr[i + 1])
            for i in range(batch_size)
        ]
        # list of graph level adj.
        ei_batch = [b.edge_index[:, ei_batch_ids[i]] for i in range(batch_size)]

        # Define new edge_index matrix per batch
        for i in range(batch_size):
            for j, sc in enumerate(supernodes_composition[i]):
                ei_batch[i] = torch.where(
                    torch.isin(ei_batch[i], sc), sn_idxes[i][j], ei_batch[i]
                )

        # Remove self loops and duplicates
        clean_new_edge_index = [
            torch.unique(remove_self_loops(adj)[0], dim=1) for adj in ei_batch
        ]

        # re-index batch adj matrix one by one
        max_num_nodes = 0
        reindexed_clean_edge_index = clean_new_edge_index.copy()
        for i in range(batch_size):
            num_nodes = data.ptr[i + 1] + num_supernodes[i]
            mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
            mask[sub_nodes[i]] = 0
            mask[: data.ptr[i]] = torch.zeros(
                data.ptr[i], dtype=torch.bool, device=device
            )
            assoc = torch.full(
                (mask.shape[0],), -1, dtype=torch.long, device=mask.device
            )
            assoc[mask] = torch.arange(
                start=max_num_nodes, end=max_num_nodes + mask.sum(), device=assoc.device
            )
            max_num_nodes = max(assoc) + 1
            reindexed_clean_edge_index[i] = assoc[clean_new_edge_index[i]]

        # Concat into one
        concat_reindexed_clean_edge_index = torch.cat(reindexed_clean_edge_index, dim=1)

        # Distances
        distance = torch.sqrt(
            (
                (
                    data.pos[concat_reindexed_clean_edge_index[0, :]]
                    - data.pos[concat_reindexed_clean_edge_index[1, :]]
                )
                ** 2
            ).sum(-1)
        )

    if opts.plot_tags is not None:
        tags = {
            0: [],
            1: [],
            2: [],
        }
        for batch in tqdm(trainer.train_loader):
            for b in batch:
                for t in tags:
                    tags[t].append((b.tags == t).sum().item())

        x = np.arange(len(tags[0]))
        ys = [np.array(tags[t]) for t in range(3)]
        z = np.zeros(len(x))
        fig = plt.figure(num=1)
        ax = fig.add_subplot(111)
        colors = {
            0: "b",
            1: "y",
            2: "g",
        }
        for t in tags:
            ax.plot(x, ys[t], color=colors[t], lw=1, label=f"tag {t}")
        for t in tags:
            ax.fill_between(
                x, ys[t], where=ys[t] > z, color=colors[t], interpolate=True
            )
        plt.legend()
        plt.savefig("tags_dist.png", dpi=150)
