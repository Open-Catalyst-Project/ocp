from __future__ import annotations

import contextlib
import logging
import math

import torch
import torch.nn as nn

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BaseModel
from fairchem.core.models.scn.smearing import GaussianSmearing

with contextlib.suppress(ImportError):
    pass


from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v2")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        avg_num_nodes (float):      Average number of nodes per graph
        avg_degree (float):         Average degree of nodes in the graph

        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
    ):
        if mmax_list is None:
            mmax_list = [2]
        if lmax_list is None:
            lmax_list = [6]
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        if self.load_energy_lin_ref:
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def _init_gp_partitions(
        self,
        atomic_numbers_full,
        data_batch_full,
        edge_index,
        edge_distance,
        edge_distance_vec,
    ):
        """ Graph Parallel
        This creates the required partial tensors for each rank given the full tensors.
        The tensors are split on the dimension along the node index using node_partition.
        """
        node_partition = gp_utils.scatter_to_model_parallel_region(
            torch.arange(len(atomic_numbers_full)).to(self.device)
        )
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]
        edge_index = edge_index[:, edge_partition]
        edge_distance = edge_distance[edge_partition]
        edge_distance_vec = edge_distance_vec[edge_partition]
        atomic_numbers = atomic_numbers_full[node_partition]
        data_batch = data_batch_full[node_partition]
        node_offset = node_partition.min().item()
        return (
            atomic_numbers,
            data_batch,
            node_offset,
            edge_index,
            edge_distance,
            edge_distance_vec,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        data_batch_full = data.batch
        data_batch = data.batch
        atomic_numbers_full = atomic_numbers
        node_offset = 0
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                atomic_numbers_full,
                data_batch_full,
                edge_index,
                edge_distance,
                edge_distance_vec,
            )
        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers_full[
                edge_index[0]
            ]  # Source atom atomic number
            target_element = atomic_numbers_full[
                edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers_full,
            edge_distance,
            edge_index,
            len(atomic_numbers),
            node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers_full,
                edge_distance,
                edge_index,
                batch=data_batch,  # for GraphDropPath
                node_offset=node_offset,
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        if gp_utils.initialized():
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        energy.index_add_(0, data_batch_full, node_energy.view(-1))
        energy = energy / self.avg_num_nodes

        # Add the per-atom linear references to the energy.
        if self.use_energy_lin_ref and self.load_energy_lin_ref:
            # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
            # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
            # where
            #
            # E_DFT = raw DFT energy,
            # E_ref = reference energy,
            # E_mean = normalizer mean,
            # E_std = normalizer std,
            # \hat{E} = predicted energy,
            # \hat{E_DFT} = predicted DFT energy.
            #
            # We can also write this as
            # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
            # which is why we save E_ref / E_std as the linear reference.
            with torch.cuda.amp.autocast(False):
                energy = energy.to(self.energy_lin_ref.dtype).index_add(
                    0,
                    data_batch_full,
                    self.energy_lin_ref[atomic_numbers_full],
                )

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(
                x,
                atomic_numbers_full,
                edge_distance,
                edge_index,
                node_offset=node_offset,
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3).contiguous()
            if gp_utils.initialized():
                forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)
            outputs["forces"] = forces

        return outputs

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_LinearV2))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)
