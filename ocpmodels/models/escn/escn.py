"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import sys
import time
from typing import List

import torch
import torch.nn as nn

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.escn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass


@registry.register_model("escn")
class eSCN(BaseModel):
    """Equivariant Spherical Channel Network
    Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        hidden_channels (int):        Number of hidden units in message passing
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels (int):          Number of channels for the edge invariant features
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float):   Width of distance basis function
        distance_resolution (float):  Distance between distance basis functions in Angstroms
        show_timing_info (bool):      Show timing and memory info
    """

    def __init__(
        self,
        output_targets: dict,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_neighbors: int = 40,
        cutoff: float = 8.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        lmax_list: List[int] = [6],
        mmax_list: List[int] = [2],
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        use_grid: bool = True,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
    ) -> None:
        super().__init__(
            output_targets=output_targets,
            node_embedding_dim=hidden_channels,
            edge_embedding_dim=edge_channels,
        )

        if "e3nn" not in sys.modules:
            logging.error(
                "You need to install the e3nn library to use the SCN model"
            )
            raise ImportError

        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_atoms = 0
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.max_neighbors = max_neighbors
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.grad_forces = False
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = (
            self.num_resolutions * self.sphere_channels
        )
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function

        # variables used for display purposes
        self.counter = 0

        # non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
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

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m))

            self.SO3_grid.append(SO3_m_grid)

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.distance_expansion,
                self.max_num_elements,
                self.SO3_grid,
                self.act,
            )
            self.layer_blocks.append(block)

        # Output blocks for energy and forces
        if "energy" in self.output_targets:
            self.energy_block = EnergyBlock(
                self.sphere_channels_all, self.num_sphere_samples, self.act
            )
        if "forces" in self.output_targets:
            if self.regress_forces:
                self.force_block = ForceBlock(
                    self.sphere_channels_all, self.num_sphere_samples, self.act
                )

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        # For each spherical point, compute the spherical harmonic coefficient weights
        sphharm_weights: List[nn.Parameter] = []
        for i in range(self.num_resolutions):
            sphharm_weights.append(
                nn.Parameter(
                    o3.spherical_harmonics(
                        torch.arange(0, self.lmax_list[i] + 1).tolist(),
                        self.sphere_points,
                        False,
                    ),
                    requires_grad=False,
                )
            )
        self.sphharm_weights = nn.ParameterList(sphharm_weights)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        device = data.pos.device
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        start_time = time.time()
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)

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

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(
                SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(
                atomic_numbers
            )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, device
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding

            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.tensor([], device=device)
        offset = 0
        # Compute the embedding values at every sampled point on the sphere
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt = torch.cat(
                [
                    x_pt,
                    torch.einsum(
                        "abc, pb->apc",
                        x.embedding[:, offset : offset + num_coefficients],
                        self.sphharm_weights[i],
                    ).contiguous(),
                ],
                dim=2,
            )
            offset = offset + num_coefficients

        x_pt = x_pt.view(-1, self.sphere_channels_all)

        outputs = {}
        outputs["sphere_values"] = x_pt
        outputs["sphere_points"] = self.sphere_points

        ###############################################################
        # Energy estimation
        ###############################################################
        if "energy" in self.output_targets:
            node_energy = self.energy_block(x_pt)
            energy = torch.zeros(len(data.natoms), device=device)
            energy.index_add_(0, data.batch, node_energy.view(-1))
            # Scale energy to help balance numerical precision w.r.t. forces
            energy = energy * 0.001
            outputs["energy"] = energy

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if "forces" in self.output_targets:
            if self.regress_forces:
                forces = self.force_block(x_pt, self.sphere_points)
                outputs["forces"] = forces

        if self.show_timing_info:
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

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # Make sure the atoms are far enough apart
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

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerBlock(torch.nn.Module):
    """
    Layer block: Perform one layer (message passing and aggregation) of the GNN

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super(LayerBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels = sphere_channels
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid

        # Message block
        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax_list,
            self.mmax_list,
            distance_expansion,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        # Compute messages by performing message block
        x_message = self.message_block(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        max_lmax = max(self.lmax_list)

        # Project to grid
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)

        # Return aggregated messages
        return x_message


class MessageBlock(torch.nn.Module):
    """
    Message block: Perform message passing

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super(MessageBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid
        self.num_resolutions = len(lmax_list)
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.edge_channels = edge_channels

        # Create edge scalar (invariant to rotations) features
        self.edge_block = EdgeBlock(
            self.edge_channels,
            distance_expansion,
            max_num_elements,
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        ###############################################################
        # Compute messages
        ###############################################################

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.edge_block(
            edge_distance,
            atomic_numbers[edge_index[0]],  # Source atom atomic number
            atomic_numbers[edge_index[1]],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        # Rotate the irreps to align with the edge
        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        # Add together the source and target results
        x_target.embedding = x_source.embedding + x_target.embedding

        # Point-wise spherical non-linearity
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)

        # Rotate back the irreps
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_target._reduce_edge(edge_index[1], len(x.embedding))

        return x_target


class SO2Block(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super(SO2Block, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(lmax_list)
        self.act = act

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficents = self.lmax_list[i] + 1
            num_channels_m0 = (
                num_channels_m0 + num_coefficents * self.sphere_channels
            )

        # SO(2) convolution for m=0
        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(
            num_channels_m0, self.hidden_channels, bias=False
        )
        self.fc2_m0 = nn.Linear(
            self.hidden_channels, num_channels_m0, bias=False
        )

        # SO(2) convolution for non-zero m
        self.so2_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            so2_conv = SO2Conv(
                m,
                self.sphere_channels,
                self.hidden_channels,
                edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.act,
            )
            self.so2_conv.append(so2_conv)

    def forward(
        self,
        x,
        x_edge,
        mappingReduced,
    ):
        num_edges = len(x_edge)

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(mappingReduced)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)

        # Compute edge scalar features for m=0
        x_edge_0 = self.act(self.fc1_dist0(x_edge))

        x_0 = x.embedding[:, 0 : mappingReduced.m_size[0]].contiguous()
        x_0 = x_0.view(num_edges, -1)

        x_0 = self.fc1_m0(x_0)
        x_0 = x_0 * x_edge_0
        x_0 = self.fc2_m0(x_0)
        x_0 = x_0.view(num_edges, -1, x.num_channels)

        # Update the m=0 coefficients
        x.embedding[:, 0 : mappingReduced.m_size[0]] = x_0

        # Compute the values for the m > 0 coefficients
        offset = mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ].contiguous()
            x_m = x_m.view(num_edges, 2, -1)
            # Perform SO(2) convolution
            x_m = self.so2_conv[m - 1](x_m, x_edge)
            x_m = x_m.view(num_edges, -1, x.num_channels)
            x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ] = x_m

            offset = offset + 2 * mappingReduced.m_size[m]

        # Reshape the spherical harmonics based on l (degree)
        x._l_primary(mappingReduced)

        return x


class SO2Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super(SO2Conv, self).__init__()
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions: int = len(self.lmax_list)
        self.m = m
        self.act = act

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= m:
                num_coefficents = self.lmax_list[i] - m + 1

            num_channels = (
                num_channels + num_coefficents * self.sphere_channels
            )

        assert num_channels > 0

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)

        # Real weights of SO(2) convolution
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)

        # Imaginary weights of SO(2) convolution
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_i = nn.Linear(self.hidden_channels, num_channels, bias=False)

    def forward(self, x_m, x_edge) -> torch.Tensor:
        # Compute edge scalar features
        x_edge = self.act(self.fc1_dist(x_edge))
        x_edge = x_edge.view(-1, 2, self.hidden_channels)

        # Perform the complex weight multiplication
        x_r = self.fc1_r(x_m)
        x_r = x_r * x_edge[:, 0:1, :]
        x_r = self.fc2_r(x_r)

        x_i = self.fc1_i(x_m)
        x_i = x_i * x_edge[:, 1:2, :]
        x_i = self.fc2_i(x_i)

        x_m_r = x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r[:, 1] + x_i[:, 0]

        return torch.stack((x_m_r, x_m_i), dim=1).contiguous()


class EdgeBlock(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        edge_channels,
        distance_expansion,
        max_num_elements,
        act,
    ) -> None:
        super(EdgeBlock, self).__init__()
        self.in_channels = distance_expansion.num_output
        self.distance_expansion = distance_expansion
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        # Compute distance embedding
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        # Compute atomic number embeddings
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Compute invariant edge embedding
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        x_edge = self.act(self.fc1_edge_attr(x_edge))

        return x_edge


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super(EnergyBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        node_energy = torch.sum(x_pt, dim=1) / self.num_sphere_samples

        return node_energy


class ForceBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels: int,
        num_sphere_samples: int,
        act,
    ) -> None:
        super(ForceBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points) -> torch.Tensor:
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        forces = torch.sum(forces, dim=1) / self.num_sphere_samples

        return forces
