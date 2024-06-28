"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.initializers import he_orthogonal_init


class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ) -> None:
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = torch.nn.Parameter(
            torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)),
            requires_grad=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, rbf, sph, id_ca, id_ragged_idx):
        """

        Arguments
        ---------
        rbf: torch.Tensor, shape=(1, nEdges, num_radial)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
        id_ca
        id_ragged_idx

        Returns
        -------
        rbf_W1: torch.Tensor, shape=(nEdges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rbf.shape[1]

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf, self.weight)
        # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)
        # (nEdges, emb_size_interm, num_spherical)

        # Zero padded dense matrix
        # maximum number of neighbors, catch empty id_ca with maximum
        if sph.shape[0] == 0:
            Kmax = 0
        else:
            Kmax = torch.max(
                torch.max(id_ragged_idx + 1),
                torch.tensor(0).to(id_ragged_idx.device),
            )

        sph2 = sph.new_zeros(num_edges, Kmax, self.num_spherical)
        sph2[id_ca, id_ragged_idx] = sph

        sph2 = torch.transpose(sph2, 1, 2)
        # (nEdges, num_spherical/emb_size_interm, Kmax)

        return rbf_W1, sph2


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.emb_size, self.emb_size_interm, self.units_out),
                requires_grad=True,
            )
        )
        he_orthogonal_init(self.weight)

    def forward(
        self,
        basis,
        m,
        id_reduce,
        id_ragged_idx,
    ) -> torch.Tensor:
        """

        Arguments
        ---------
        basis
        m: quadruplets: m = m_db , triplets: m = m_ba
        id_reduce
        id_ragged_idx

        Returns
        -------
            m_ca: torch.Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        # num_spherical is actually num_spherical**2 for quadruplets
        (rbf_W1, sph) = basis
        # (nEdges, emb_size_interm, num_spherical), (nEdges, num_spherical, Kmax)
        nEdges = rbf_W1.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = torch.max(
            torch.max(id_ragged_idx) + 1,
            torch.tensor(0).to(id_ragged_idx.device),
        )
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        m2 = m.new_zeros(nEdges, Kmax, self.emb_size)
        m2[id_reduce, id_ragged_idx] = m
        # (num_quadruplets or num_triplets, emb_size) -> (nEdges, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (nEdges, num_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)
        # (nEdges, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_ca = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)
        # (emb_size, nEdges, units_out)
        return torch.sum(m_ca, dim=0)
        # (nEdges, units_out)
