"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch

from fairchem.core.common import gp_utils
from fairchem.core.models.base_layers import Dense, ResidualLayer
from fairchem.core.modules.scaling import ScaleFactor

from .atom_update_block import AtomUpdateBlock
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding


class InteractionBlockTripletsOnly(torch.nn.Module):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        activation: str | None = None,
        name: str = "Interaction",
    ) -> None:
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for _ in range(num_before_skip)
            ]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for _ in range(num_after_skip)
            ]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, activation=activation)
                for _ in range(num_concat)
            ]
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        rbf_h,
        idx_s,
        idx_t,
        edge_offset,
        Kmax,
        nAtoms,
    ):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: torch.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
            Node: h
            Edge: m, rbf3, id_swap, rbf_h, idx_s, idx_t, cbf3[0], cbf3[1] (dense)
            Triplet: id3_ragged_idx, id3_ba, id3_ca
        """
        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x3 = self.trip_interaction(
            m,
            rbf3,
            cbf3,
            id3_ragged_idx,
            id_swap,
            id3_ba,
            id3_ca,
            edge_offset,
            Kmax,
        )

        ## ----------------------------- Merge Embeddings after Triplet Interaction ------------------------------ ##
        x = x_ca_skip + x3  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_2

        ## ---------------------------------------- Update Edge Embeddings --------------------------------------- ##
        # Transformations before skip connection
        for _, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for _, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## ---------------------------------------- Update Atom Embeddings --------------------------------------- ##
        h2 = self.atom_update(nAtoms, m, rbf_h, idx_t)

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, idx_s, idx_t)  # (nEdges, emb_size_edge)

        for _, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class TripletInteraction(torch.nn.Module):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
    """

    def __init__(
        self,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_bilinear: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        activation: str | None = None,
        name: str = "TripletInteraction",
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_ba = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            emb_size_rbf,
            emb_size_edge,
            activation=None,
            bias=False,
        )
        self.scale_rbf = ScaleFactor(name + "_had_rbf")

        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear
        )

        # combines scaling for bilinear layer and summation
        self.scale_cbf_sum = ScaleFactor(name + "_sum_cbf")

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m: torch.Tensor,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        edge_offset,
        Kmax,
    ):
        """
        Returns
        -------
            m: torch.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """

        # Dense transformation
        x_ba = self.dense_ba(m)  # (nEdges, emb_size_edge)

        # Transform via radial bessel basis
        rbf_emb = self.mlp_rbf(rbf3)  # (nEdges, emb_size_edge)
        x_ba2 = x_ba * rbf_emb
        x_ba = self.scale_rbf(x_ba2, ref=x_ba)

        x_ba = self.down_projection(x_ba)  # (nEdges, emb_size_trip)

        # Graph Parallel: Gather x_ba from all nodes
        x_ba = gp_utils.gather_from_model_parallel_region(x_ba, dim=0)

        # Transform via circular spherical basis
        x_ba = x_ba[id3_ba]

        # Efficient bilinear layer
        x = self.mlp_cbf(cbf3, x_ba, id3_ca, id3_ragged_idx, edge_offset, Kmax)
        # (nEdges, emb_size_quad)
        x = self.scale_cbf_sum(x, ref=x_ba)

        # =>
        # rbf(d_ba)
        # cbf(d_ca, angle_cab)

        # Up project embeddings
        x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
        x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

        # Graph Parallel: Gather x_ac from all nodes
        x_ac = gp_utils.gather_from_model_parallel_region(x_ac, dim=0)

        # Merge interaction of c->a and a->c
        x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
        x_ac = gp_utils.scatter_to_model_parallel_region(x_ac, dim=0)

        x3 = x_ca + x_ac
        return x3 * self.inv_sqrt_2
