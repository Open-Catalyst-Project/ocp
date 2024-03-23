"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)
from ocpmodels.models.scn.spherical_harmonics import SphericalHarmonicsHelper

try:
    from e3nn import o3
except ImportError:
    pass


@registry.register_model("scn")
class SphericalChannelNetwork(BaseModel):
    """Spherical Channel Network
    Paper: Spherical Channels for Modeling Atomic Interactions

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_num_neighbors (int): Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_interactions (int): Number of layers in the GNN
        lmax (int):             Maximum degree of the spherical harmonics (1 to 10)
        mmax (int):             Maximum order of the spherical harmonics (0 or 1)
        num_resolutions (int):  Number of resolutions used to compute messages, further away atoms has lower resolution (1 or 2)
        sphere_channels (int):  Number of spherical channels
        sphere_channels_reduce (int): Number of spherical channels used during message passing (downsample or upsample)
        hidden_channels (int):  Number of hidden units in message passing
        num_taps (int):         Number of taps or rotations used during message passing (1 or otherwise set automatically based on mmax)

        use_grid (bool):        Use non-linear pointwise convolution during aggregation
        num_bands (int):        Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)

        num_sphere_samples (int): Number of samples used to approximate the integration of the sphere in the output blocks
        num_basis_functions (int): Number of basis functions used for distance and atomic number blocks
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float): Width of distance basis function
        distance_resolution (float): Distance between distance basis functions in Angstroms

        show_timing_info (bool): Show timing and memory info
    """

    energy_fc1: nn.Linear
    energy_fc2: nn.Linear
    energy_fc3: nn.Linear
    force_fc1: nn.Linear
    force_fc2: nn.Linear
    force_fc3: nn.Linear

    def __init__(
        self,
        output_targets: dict,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_num_neighbors: int = 20,
        cutoff: float = 8.0,
        max_num_elements: int = 90,
        num_interactions: int = 8,
        lmax: int = 6,
        mmax: int = 1,
        num_resolutions: int = 2,
        sphere_channels: int = 128,
        sphere_channels_reduce: int = 128,
        hidden_channels: int = 256,
        num_taps: int = -1,
        use_grid: bool = True,
        num_bands: int = 1,
        num_sphere_samples: int = 128,
        num_basis_functions: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
        direct_forces: bool = True,
    ) -> None:
        super().__init__(
            output_targets=output_targets,
            node_embedding_dim=hidden_channels,
            edge_embedding_dim=hidden_channels,
        )

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.2.6 to use SCN.")
            raise ImportError

        self.output_targets = output_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.num_atoms = 0
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.sphere_channels_reduce = sphere_channels_reduce
        self.max_num_neighbors = self.max_neighbors = max_num_neighbors
        self.num_basis_functions = num_basis_functions
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax = lmax
        self.mmax = mmax
        self.basis_width_scalar = basis_width_scalar
        self.sphere_basis = (self.lmax + 1) ** 2
        self.use_grid = use_grid
        self.distance_function = distance_function

        # variables used for display purposes
        self.counter = 0

        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]

        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        if num_resolutions == 1:
            self.num_resolutions = 1
            self.hidden_channels_list = torch.tensor([self.hidden_channels])
            self.lmax_list = torch.tensor(
                [self.lmax, -1]
            )  # always end with -1
            self.cutoff_list = torch.tensor([self.max_num_neighbors - 0.01])
        if num_resolutions == 2:
            self.num_resolutions = 2
            self.hidden_channels_list = torch.tensor(
                [self.hidden_channels, self.hidden_channels // 4]
            )
            self.lmax_list = torch.tensor([self.lmax, max(4, self.lmax - 2)])
            self.cutoff_list = torch.tensor(
                [12 - 0.01, self.max_num_neighbors - 0.01]
            )

        self.sphharm_list = []
        for i in range(self.num_resolutions):
            self.sphharm_list.append(
                SphericalHarmonicsHelper(
                    self.lmax_list[i],
                    self.mmax,
                    num_taps,
                    num_bands,
                )
            )

        self.edge_blocks = nn.ModuleList()
        for _ in range(self.num_interactions):
            block = EdgeBlock(
                self.num_resolutions,
                self.sphere_channels_reduce,
                self.hidden_channels_list,
                self.cutoff_list,
                self.sphharm_list,
                self.sphere_channels,
                self.distance_expansion,
                self.max_num_elements,
                self.num_basis_functions,
                self.num_gaussians,
                self.use_grid,
                self.act,
            )
            self.edge_blocks.append(block)

        # Energy estimation
        self.energy_fc1 = nn.Linear(self.sphere_channels, self.sphere_channels)
        self.energy_fc2 = nn.Linear(
            self.sphere_channels, self.sphere_channels_reduce
        )
        self.energy_fc3 = nn.Linear(self.sphere_channels_reduce, 1)

        # Force estimation
        if self.regress_forces:
            self.force_fc1 = nn.Linear(
                self.sphere_channels, self.sphere_channels
            )
            self.force_fc2 = nn.Linear(
                self.sphere_channels, self.sphere_channels_reduce
            )
            self.force_fc3 = nn.Linear(self.sphere_channels_reduce, 1)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        self.device = data.pos.device
        self.num_atoms = len(data.batch)
        self.batch_size = len(data.natoms)

        start_time = time.time()

        outputs = self._forward_helper(
            data,
        )

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                "{} Time: {}\tMemory: {}\t{}".format(
                    self.counter,
                    time.time() - start_time,
                    len(data.pos),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        self.counter = self.counter + 1

        return outputs

    # restructure forward helper for conditional grad
    def _forward_helper(self, data):
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Calculate which message block each edge should use. Based on edge distance rank.
        edge_rank = self._rank_edge_distances(
            edge_distance, edge_index, self.max_num_neighbors
        )

        # Reorder edges so that they are grouped by distance rank (lowest to highest)
        last_cutoff = -0.1
        message_block_idx = torch.zeros(len(edge_distance), device=pos.device)
        edge_distance_reorder = torch.tensor([], device=self.device)
        edge_index_reorder = torch.tensor([], device=self.device)
        edge_distance_vec_reorder = torch.tensor([], device=self.device)
        cutoff_index = torch.tensor([0], device=self.device)
        for i in range(self.num_resolutions):
            mask = torch.logical_and(
                edge_rank.gt(last_cutoff), edge_rank.le(self.cutoff_list[i])
            )
            last_cutoff = self.cutoff_list[i]
            message_block_idx.masked_fill_(mask, i)
            edge_distance_reorder = torch.cat(
                [
                    edge_distance_reorder,
                    torch.masked_select(edge_distance, mask),
                ],
                dim=0,
            )
            edge_index_reorder = torch.cat(
                [
                    edge_index_reorder,
                    torch.masked_select(
                        edge_index, mask.view(1, -1).repeat(2, 1)
                    ).view(2, -1),
                ],
                dim=1,
            )
            edge_distance_vec_mask = torch.masked_select(
                edge_distance_vec, mask.view(-1, 1).repeat(1, 3)
            ).view(-1, 3)
            edge_distance_vec_reorder = torch.cat(
                [edge_distance_vec_reorder, edge_distance_vec_mask], dim=0
            )
            cutoff_index = torch.cat(
                [
                    cutoff_index,
                    torch.tensor(
                        [len(edge_distance_reorder)], device=self.device
                    ),
                ],
                dim=0,
            )

        edge_index = edge_index_reorder.long()
        edge_distance = edge_distance_reorder
        edge_distance_vec = edge_distance_vec_reorder

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.sphharm_list[i].InitWignerDMatrix(
                edge_rot_mat[cutoff_index[i] : cutoff_index[i + 1]],
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = torch.zeros(
            num_atoms,
            self.sphere_basis,
            self.sphere_channels,
            device=pos.device,
        )
        x[:, 0, :] = self.sphere_embedding(atomic_numbers)

        ###############################################################
        # Update spherical nodg embeddings
        ###############################################################
        for i, interaction in enumerate(self.edge_blocks):
            if i > 0:
                x = x + interaction(
                    x, atomic_numbers, edge_distance, edge_index, cutoff_index
                )
            else:
                x = interaction(
                    x, atomic_numbers, edge_distance, edge_index, cutoff_index
                )

        ###############################################################
        # Estimate energy and forces using the node embeddings
        ###############################################################

        # Create a roughly evenly distributed point sampling of the sphere
        sphere_points = CalcSpherePoints(
            self.num_sphere_samples, x.device
        ).detach()
        sphharm_weights = o3.spherical_harmonics(
            torch.arange(0, self.lmax + 1).tolist(), sphere_points, False
        ).detach()

        # Energy estimation
        node_energy = torch.einsum(
            "abc, pb->apc", x, sphharm_weights
        ).contiguous()
        node_energy = node_energy.view(-1, self.sphere_channels)
        node_energy = self.act(self.energy_fc1(node_energy))
        node_energy = self.act(self.energy_fc2(node_energy))
        node_energy = self.energy_fc3(node_energy)
        node_energy = node_energy.view(-1, self.num_sphere_samples, 1)
        node_energy = torch.sum(node_energy, dim=1) / self.num_sphere_samples
        energy = torch.zeros(len(data.natoms), device=pos.device)
        energy.index_add_(0, data.batch, node_energy.view(-1))

        outputs = {"energy": energy}

        # Force estimation
        if self.regress_forces:
            forces = torch.einsum(
                "abc, pb->apc", x, sphharm_weights
            ).contiguous()
            forces = forces.view(-1, self.sphere_channels)
            forces = self.act(self.force_fc1(forces))
            forces = self.act(self.force_fc2(forces))
            forces = self.force_fc3(forces)
            forces = forces.view(-1, self.num_sphere_samples, 1)
            forces = forces * sphere_points.view(1, self.num_sphere_samples, 3)
            forces = torch.sum(forces, dim=1) / self.num_sphere_samples
            outputs["forces"] = forces

        return outputs

    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        if torch.min(edge_vec_0_distance) < 0.0001:
            logging.error(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(edge_vec_0_distance)
                )
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            logging.error(
                "Error edge_vec_0_distance: {} {} {} {} {}".format(
                    minidx,
                    edge_index[0, minidx],
                    edge_index[1, minidx],
                    data.pos[edge_index[0, minidx]],
                    data.pos[edge_index[1, minidx]],
                )
            )

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
            -1, 1
        )
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
            -1, 1
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
        )
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
        )
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
        )
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
        )

        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    def _rank_edge_distances(
        self, edge_distance, edge_index, max_num_neighbors: int
    ) -> torch.Tensor:
        device = edge_distance.device
        # Create an index map to map distances from atom_distance to distance_sort
        # index_sort_map assumes index to be sorted
        output, num_neighbors = torch.unique(edge_index[1], return_counts=True)
        index_neighbor_offset = (
            torch.cumsum(num_neighbors, dim=0) - num_neighbors
        )
        index_neighbor_offset_expand = torch.repeat_interleave(
            index_neighbor_offset, num_neighbors
        )

        index_sort_map = (
            edge_index[1] * max_num_neighbors
            + torch.arange(len(edge_distance), device=device)
            - index_neighbor_offset_expand
        )

        num_atoms = int(torch.max(edge_index)) + 1
        distance_sort = torch.full(
            [num_atoms * max_num_neighbors], np.inf, device=device
        )
        distance_sort.index_copy_(0, index_sort_map, edge_distance)
        distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
        no_op, index_sort = torch.sort(distance_sort, dim=1)

        index_map = (
            torch.arange(max_num_neighbors, device=device)
            .view(1, -1)
            .repeat(num_atoms, 1)
            .view(-1)
        )
        index_sort = index_sort + (
            torch.arange(num_atoms, device=device) * max_num_neighbors
        ).view(-1, 1).repeat(1, max_num_neighbors)
        edge_rank = torch.zeros_like(index_map)
        edge_rank.index_copy_(0, index_sort.view(-1), index_map)
        edge_rank = edge_rank.view(num_atoms, max_num_neighbors)

        index_sort_mask = distance_sort.lt(1000.0)
        edge_rank = torch.masked_select(edge_rank, index_sort_mask)

        return edge_rank

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EdgeBlock(torch.nn.Module):
    def __init__(
        self,
        num_resolutions: int,
        sphere_channels_reduce,
        hidden_channels_list,
        cutoff_list,
        sphharm_list,
        sphere_channels,
        distance_expansion,
        max_num_elements: int,
        num_basis_functions: int,
        num_gaussians: int,
        use_grid: bool,
        act,
    ) -> None:
        super(EdgeBlock, self).__init__()
        self.num_resolutions = num_resolutions
        self.act = act
        self.hidden_channels_list = hidden_channels_list
        self.sphere_channels = sphere_channels
        self.sphere_channels_reduce = sphere_channels_reduce
        self.distance_expansion = distance_expansion
        self.cutoff_list = cutoff_list
        self.sphharm_list = sphharm_list
        self.max_num_elements = max_num_elements
        self.num_basis_functions = num_basis_functions
        self.use_grid = use_grid
        self.num_gaussians = num_gaussians

        # Edge features
        self.dist_block = DistanceBlock(
            self.num_gaussians,
            self.num_basis_functions,
            self.distance_expansion,
            self.max_num_elements,
            self.act,
        )

        # Create a message block for each cutoff
        self.message_blocks = nn.ModuleList()
        for i in range(self.num_resolutions):
            block = MessageBlock(
                self.sphere_channels_reduce,
                int(self.hidden_channels_list[i]),
                self.num_basis_functions,
                self.sphharm_list[i],
                self.act,
            )
            self.message_blocks.append(block)

        # Downsampling number of sphere channels
        # Make sure bias is false unless equivariance is lost
        if self.sphere_channels != self.sphere_channels_reduce:
            self.downsample = nn.Linear(
                self.sphere_channels,
                self.sphere_channels_reduce,
                bias=False,
            )
            self.upsample = nn.Linear(
                self.sphere_channels_reduce,
                self.sphere_channels,
                bias=False,
            )

        # Use non-linear message aggregation?
        if self.use_grid:
            # Network for each node to combine edge messages
            self.fc1_sphere = nn.Linear(
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
            )

            self.fc2_sphere = nn.Linear(
                self.sphharm_list[0].num_bands
                * 2
                * self.sphere_channels_reduce,
                2 * self.sphere_channels_reduce,
            )

            self.fc3_sphere = nn.Linear(
                2 * self.sphere_channels_reduce, self.sphere_channels_reduce
            )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        cutoff_index,
    ):
        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        x_edge = self.dist_block(
            edge_distance,
            atomic_numbers[edge_index[0]],
            atomic_numbers[edge_index[1]],
        )
        x_new = torch.zeros(
            len(x),
            self.sphharm_list[0].sphere_basis,
            self.sphere_channels_reduce,
            dtype=x.dtype,
            device=x.device,
        )

        if self.sphere_channels != self.sphere_channels_reduce:
            x_down = self.downsample(x.view(-1, self.sphere_channels))
        else:
            x_down = x
        x_down = x_down.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels_reduce
        )

        for i, interaction in enumerate(self.message_blocks):
            start_idx = cutoff_index[i]
            end_idx = cutoff_index[i + 1]

            x_message = interaction(
                x_down[:, 0 : self.sphharm_list[i].sphere_basis, :],
                x_edge[start_idx:end_idx],
                edge_index[:, start_idx:end_idx],
            )

            # Sum all incoming edges to the target nodes
            x_new[:, 0 : self.sphharm_list[i].sphere_basis, :].index_add_(
                0, edge_index[1, start_idx:end_idx], x_message.to(x_new.dtype)
            )

        if self.use_grid:
            # Feed in the spherical functions from the previous time step
            x_grid = self.sphharm_list[0].ToGrid(
                x_down, self.sphere_channels_reduce
            )
            x_grid = torch.cat(
                [
                    x_grid,
                    self.sphharm_list[0].ToGrid(
                        x_new, self.sphere_channels_reduce
                    ),
                ],
                dim=1,
            )

            x_grid = self.act(self.fc1_sphere(x_grid))
            x_grid = self.act(self.fc2_sphere(x_grid))
            x_grid = self.fc3_sphere(x_grid)
            x_new = self.sphharm_list[0].FromGrid(
                x_grid, self.sphere_channels_reduce
            )

        if self.sphere_channels != self.sphere_channels_reduce:
            x_new = x_new.view(-1, self.sphere_channels_reduce)
            x_new = self.upsample(x_new)
        x_new = x_new.view(
            -1, self.sphharm_list[0].sphere_basis, self.sphere_channels
        )

        return x_new


class MessageBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels_reduce,
        hidden_channels,
        num_basis_functions,
        sphharm,
        act,
    ) -> None:
        super(MessageBlock, self).__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels_reduce = sphere_channels_reduce
        self.sphharm = sphharm

        self.fc1_dist = nn.Linear(num_basis_functions, self.hidden_channels)

        # Network for each edge to compute edge messages
        self.fc1_edge_proj = nn.Linear(
            2 * self.sphharm.sphere_basis_reduce * self.sphere_channels_reduce,
            self.hidden_channels,
        )

        self.fc1_edge = nn.Linear(self.hidden_channels, self.hidden_channels)

        self.fc2_edge = nn.Linear(
            self.hidden_channels,
            self.sphharm.sphere_basis_reduce * self.sphere_channels_reduce,
        )

    def forward(
        self,
        x,
        x_edge,
        edge_index,
    ):
        ###############################################################
        # Compute messages
        ###############################################################
        x_edge = self.act(self.fc1_dist(x_edge))

        x_source = x[edge_index[0, :]]
        x_target = x[edge_index[1, :]]

        # Rotate the spherical harmonic basis functions to align with the edge
        x_msg_source = self.sphharm.Rotate(x_source)
        x_msg_target = self.sphharm.Rotate(x_target)

        # Compute messages
        x_message = torch.cat([x_msg_source, x_msg_target], dim=1)
        x_message = self.act(self.fc1_edge_proj(x_message))
        x_message = (
            x_message.view(
                -1, self.sphharm.num_y_rotations, self.hidden_channels
            )
        ) * x_edge.view(-1, 1, self.hidden_channels)
        x_message = x_message.view(-1, self.hidden_channels)

        x_message = self.act(self.fc1_edge(x_message))
        x_message = self.act(self.fc2_edge(x_message))

        # Combine the rotated versions of the messages
        x_message = x_message.view(-1, self.sphere_channels_reduce)
        x_message = self.sphharm.CombineYRotations(x_message)

        # Rotate the spherical harmonic basis functions back to global coordinate frame
        x_message = self.sphharm.RotateInv(x_message)

        return x_message


class DistanceBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_basis_functions: int,
        distance_expansion,
        max_num_elements: int,
        act,
    ) -> None:
        super(DistanceBlock, self).__init__()
        self.in_channels = in_channels
        self.distance_expansion = distance_expansion
        self.act = act
        self.num_basis_functions = num_basis_functions
        self.max_num_elements = max_num_elements
        self.num_edge_channels = self.num_basis_functions

        self.fc1_dist = nn.Linear(self.in_channels, self.num_basis_functions)

        self.source_embedding = nn.Embedding(
            self.max_num_elements, self.num_basis_functions
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements, self.num_basis_functions
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.fc1_edge_attr = nn.Linear(
            self.num_edge_channels,
            self.num_edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        x_edge = self.act(source_embedding + target_embedding + x_dist)
        x_edge = self.act(self.fc1_edge_attr(x_edge))

        return x_edge
