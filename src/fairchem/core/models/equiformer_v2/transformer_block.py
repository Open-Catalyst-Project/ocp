from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch_geometric

from fairchem.core.common import gp_utils

from .activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from .drop import EquivariantDropoutArraySphericalHarmonics, GraphDropPath
from .layer_norm import get_normalization_layer
from .radial_function import RadialFunction
from .so2_ops import SO2_Convolution
from .so3 import SO3_Embedding, SO3_LinearV2


class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements: int,
        edge_channels_list,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        activation="scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_LinearV2(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax_list[0],
        )

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers,
        edge_distance: torch.Tensor,
        edge_index,
        node_offset: int = 0,
    ):
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region(x.embedding, dim=0)
            x_source.set_embedding(x_full)
            x_target.set_embedding(x_full)
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            # x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1] - node_offset, len(x.embedding))

        # Project
        return self.proj(x_message)


class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_grid,
        activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.max_lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.max_lmax, self.max_lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.max_lmax
        )

    def forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(
                    input_embedding.embedding.narrow(1, 0, 1)
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(
                self.SO3_grid, lmax=self.max_lmax
            )
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(
                input_embedding_grid, self.SO3_grid, lmax=self.max_lmax
            )

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.embedding.narrow(
                            1, 1, input_embedding.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(
                    gating_scalars, input_embedding.embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(
                        gating_scalars,
                        input_embedding.embedding,
                        self.SO3_grid,
                    )
                else:
                    input_embedding.embedding = self.s2_act(
                        input_embedding.embedding, self.SO3_grid
                    )

        return self.so3_linear_2(input_embedding)


class TransBlockV2(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN
    """

    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        norm_type: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = (
            EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False)
            if proj_drop > 0.0
            else None
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
        )

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(
                sphere_channels, output_channels, lmax=max_lmax
            )
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x,  # SO3_Embedding
        atomic_numbers,
        edge_distance,
        edge_index,
        batch,  # for GraphDropPath
        node_offset: int = 0,
    ):
        output_embedding = x

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(
            output_embedding, atomic_numbers, edge_distance, edge_index, node_offset
        )

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype,
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(
                output_embedding.lmax_list.copy(),
                output_embedding.lmax_list.copy(),
            )
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding
