"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph

from fairchem.core.common.utils import (
    cg_change_mat,
    compute_neighbors,
    get_pbc_distances,
    irreps_sum,
    radius_graph_pbc,
    scatter_det,
)

from .base_layers import Dense


class BaseModel(nn.Module):
    def __init__(
        self,
        output_targets: dict | None = None,
        node_embedding_dim=None,
        edge_embedding_dim=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.output_targets = output_targets or {}
        self.num_targets = len(output_targets)

        self.module_dict = nn.ModuleDict({})
        for target in output_targets:
            if self.output_targets[target].get("default_head", False):
                if "decomposition" in self.output_targets[target]:
                    # If decomposed, we only need heads for the individual components
                    continue

                if "irrep_dim" in self.output_targets[target]:
                    if edge_embedding_dim is None:
                        raise NotImplementedError(
                            "Model does not support SO(3) equivariant prediction without edge embeddings."
                        )
                    embedding_dim = edge_embedding_dim
                    output_shape = 1
                else:
                    embedding_dim = node_embedding_dim
                    output_shape = self.output_targets[target].get("shape", 1)

                layers = [
                    Dense(
                        embedding_dim,
                        embedding_dim,
                        activation="silu",
                    )
                    for _ in range(self.output_targets[target].get("num_layers", 2) - 1)
                ]

                layers.append(Dense(embedding_dim, output_shape, activation=None))

                self.module_dict[target] = nn.Sequential(*layers)

    def forward(self, data):
        batch = data.batch
        self.device = data.pos.device
        self.num_atoms = data.atomic_numbers.shape[0]
        self.num_systems = data.natoms.shape[0]

        # call declared model forward pass
        out = self._forward(data)
        results = {}

        for target in self.output_targets:
            # Skip parent decomposed targets. The components should exist separately
            if "decomposition" in self.output_targets[target]:
                continue

            ### for models that directly return desired property, add directly
            if target not in self.module_dict:
                pred = out[target]
                # squeeze if necessary
                if len(pred.shape) > 1:
                    pred = pred.squeeze(dim=1)

                results[target] = pred
                continue

            assert (
                target not in out
            ), f"{target} is explicitly returned, but also has a default head"

            # equivariant prediction
            if "irrep_dim" in self.output_targets[target]:
                pred = self.forward_irrep(out, target)
            # scalar prediction
            else:
                pred = self.module_dict[target](out["node_embedding"])

            # (batch, output_shape)
            if self.output_targets[target].get("level", "system") == "system":
                pred = scatter_det(
                    pred,
                    batch,
                    dim=0,
                    dim_size=self.num_systems,
                    reduce=self.output_targets[target].get("reduce", "add"),
                )

            results[target] = pred.squeeze(1)

        self.construct_parent_tensor(results)

        return results

    def forward_irrep(self, out, target):
        """
        For equivariant properties, make use of spherical harmonics to ensure
        SO(3) equivariance.
        """
        irrep = self.output_targets[target]["irrep_dim"]

        ### Compute spherical harmonics based on edge vectors
        assert "edge_vec" in out
        assert "edge_idx" in out
        assert "edge_embedding" in out

        edge_vec = out["edge_vec"]
        edge_idx = out["edge_idx"]

        # (nedges, (2*irrep_dim+1))
        sphharm = o3.spherical_harmonics(irrep, edge_vec, True).detach()
        # (nedges, 1)
        pred = self.module_dict[target](out["edge_embedding"])
        # (nedges, 2*irrep_dim+1)
        pred = pred * sphharm

        # aggregate edges per node
        # (nnodes, 2*irrep_dim+1)
        return scatter_det(pred, edge_idx, dim=0, dim_size=self.num_atoms, reduce="add")

    def construct_parent_tensor(self, results):
        parent_construction = defaultdict(dict)

        # Identify target properties that are decomposition of parent property
        for target in self.output_targets:
            if "parent" in self.output_targets[target]:
                parent_target = self.output_targets[target]["parent"]
                irrep_dim = self.output_targets[target]["irrep_dim"]
                ### NOTE: Only supports rank 2 tensors
                ### TODO: Remove dictionary structure when rank 3 tensors are supported
                parent_construction[parent_target][irrep_dim] = target

        # Construct parent tensors from predicted irreps
        for parent_target in parent_construction:
            rank = max(parent_construction[parent_target].keys())
            cg_matrix = cg_change_mat(rank, self.device)

            # TODO: handle per-atom vs per-system properties
            prediction_irreps = torch.zeros(
                (self.num_systems, irreps_sum(rank)), device=self.device
            )

            # Rank 2 support
            for irrep in range(rank + 1):
                if irrep in parent_construction[parent_target]:
                    # (batch, 2*irrep+1)
                    prediction_irreps[
                        :, max(0, irreps_sum(irrep - 1)) : irreps_sum(irrep)
                    ] = results[parent_construction[parent_target][irrep]].view(
                        -1, 2 * irrep + 1
                    )

            # NOTE: AMP will return this as a float-16 tensor
            parent_prediction = torch.mm(prediction_irreps, cg_matrix)

            results[parent_target] = parent_prediction

    @abstractmethod
    def _forward(self, data):
        """
        Derived models should implement this function. Expected output should
        be in the following format:

            out = {
                "output_property_1": pred_1,
                "output_property_2": pred_2,
            }

        Where `output_property` are the desired model outputs as defined in the
        `outputs` section of the model config.
        """

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
