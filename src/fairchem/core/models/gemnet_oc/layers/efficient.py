"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.base_layers import Dense
from fairchem.core.models.initializers import he_orthogonal_init


class BasisEmbedding(torch.nn.Module):
    """
    Embed a basis (CBF, SBF), optionally using the efficient reformulation.

    Arguments
    ---------
    num_radial: int
        Number of radial basis functions.
    emb_size_interm: int
        Intermediate embedding size of triplets/quadruplets.
    num_spherical: int
        Number of circular/spherical basis functions.
        Only required if there is a circular/spherical basis.
    """

    weight: torch.nn.Parameter

    def __init__(
        self,
        num_radial: int,
        emb_size_interm: int,
        num_spherical: int | None = None,
    ) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        if num_spherical is None:
            self.weight = torch.nn.Parameter(
                torch.empty(emb_size_interm, num_radial),
                requires_grad=True,
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(num_radial, num_spherical, emb_size_interm),
                requires_grad=True,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.weight)

    def forward(
        self,
        rad_basis,
        sph_basis=None,
        idx_rad_outer=None,
        idx_rad_inner=None,
        idx_sph_outer=None,
        idx_sph_inner=None,
        num_atoms=None,
    ):
        """

        Arguments
        ---------
        rad_basis: torch.Tensor, shape=(num_edges, num_radial or num_orders * num_radial)
            Raw radial basis.
        sph_basis: torch.Tensor, shape=(num_triplets or num_quadruplets, num_spherical)
            Raw spherical or circular basis.
        idx_rad_outer: torch.Tensor, shape=(num_edges)
            Atom associated with each radial basis value.
            Optional, used for efficient edge aggregation.
        idx_rad_inner: torch.Tensor, shape=(num_edges)
            Enumerates radial basis values per atom.
            Optional, used for efficient edge aggregation.
        idx_sph_outer: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Edge associated with each circular/spherical basis value.
            Optional, used for efficient triplet/quadruplet aggregation.
        idx_sph_inner: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Enumerates circular/spherical basis values per edge.
            Optional, used for efficient triplet/quadruplet aggregation.
        num_atoms: int
            Total number of atoms.
            Optional, used for efficient edge aggregation.

        Returns
        -------
        rad_W1: torch.Tensor, shape=(num_edges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(num_edges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rad_basis.shape[0]

        if self.num_spherical is not None:
            # MatMul: mul + sum over num_radial
            rad_W1 = rad_basis @ self.weight.reshape(self.weight.shape[0], -1)
            # (num_edges, emb_size_interm * num_spherical)
            rad_W1 = rad_W1.reshape(num_edges, -1, sph_basis.shape[-1])
            # (num_edges, emb_size_interm, num_spherical)
        else:
            # MatMul: mul + sum over num_radial
            rad_W1 = rad_basis @ self.weight.T
            # (num_edges, emb_size_interm)

        if idx_rad_inner is not None:
            # Zero padded dense matrix
            # maximum number of neighbors

            Kmax = 0 if idx_rad_outer.shape[0] == 0 else torch.max(idx_rad_inner) + 1

            rad_W1_padded = rad_W1.new_zeros([num_atoms, Kmax, *list(rad_W1.shape[1:])])
            rad_W1_padded[idx_rad_outer, idx_rad_inner] = rad_W1
            # (num_atoms, Kmax, emb_size_interm, ...)
            rad_W1_padded = torch.transpose(rad_W1_padded, 1, 2)
            # (num_atoms, emb_size_interm, Kmax, ...)
            rad_W1_padded = rad_W1_padded.reshape(num_atoms, rad_W1.shape[1], -1)
            # (num_atoms, emb_size_interm, Kmax2 * ...)
            rad_W1 = rad_W1_padded

        if idx_sph_inner is not None:
            # Zero padded dense matrix
            # maximum number of neighbors
            Kmax = 0 if idx_sph_outer.shape[0] == 0 else torch.max(idx_sph_inner) + 1

            sph2 = sph_basis.new_zeros(num_edges, Kmax, sph_basis.shape[-1])
            sph2[idx_sph_outer, idx_sph_inner] = sph_basis
            # (num_edges, Kmax, num_spherical)
            sph2 = torch.transpose(sph2, 1, 2)
            # (num_edges, num_spherical, Kmax)

        if sph_basis is None:
            return rad_W1
        else:
            if idx_sph_inner is None:
                rad_W1 = rad_W1[idx_sph_outer]
                # (num_triplets, emb_size_interm, num_spherical)

                sph_W1 = rad_W1 @ sph_basis[:, :, None]
                # (num_triplets, emb_size_interm, num_spherical)
                return sph_W1.squeeze(-1)
            else:
                return rad_W1, sph2


class EfficientInteractionBilinear(torch.nn.Module):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Arguments
    ---------
    emb_size_in: int
        Embedding size of input triplets/quadruplets.
    emb_size_interm: int
        Intermediate embedding size of the basis transformation.
    emb_size_out: int
        Embedding size of output triplets/quadruplets.
    """

    def __init__(
        self,
        emb_size_in: int,
        emb_size_interm: int,
        emb_size_out: int,
    ) -> None:
        super().__init__()
        self.emb_size_in = emb_size_in
        self.emb_size_interm = emb_size_interm
        self.emb_size_out = emb_size_out

        self.bilinear = Dense(
            self.emb_size_in * self.emb_size_interm,
            self.emb_size_out,
            bias=False,
            activation=None,
        )

    def forward(
        self,
        basis,
        m,
        idx_agg_outer,
        idx_agg_inner,
        idx_agg2_outer=None,
        idx_agg2_inner=None,
        agg2_out_size=None,
    ):
        """

        Arguments
        ---------
        basis: Tuple (torch.Tensor, torch.Tensor),
            shapes=((num_edges, emb_size_interm, num_spherical),
                    (num_edges, num_spherical, Kmax))
            First element: Radial basis multiplied with weight matrix
            Second element: Circular/spherical basis
        m: torch.Tensor, shape=(num_edges, emb_size_in)
            Input edge embeddings
        idx_agg_outer: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Output edge aggregating this intermediate triplet/quadruplet edge.
        idx_agg_inner: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Enumerates intermediate edges per output edge.
        idx_agg2_outer: torch.Tensor, shape=(num_edges)
            Output atom aggregating this edge.
        idx_agg2_inner: torch.Tensor, shape=(num_edges)
            Enumerates edges per output atom.
        agg2_out_size: int
            Number of output embeddings when aggregating twice. Typically
            the number of atoms.

        Returns
        -------
        m_ca: torch.Tensor, shape=(num_edges, emb_size)
            Aggregated edge/atom embeddings.
        """
        # num_spherical is actually num_spherical**2 for quadruplets
        (rad_W1, sph) = basis
        # (num_edges, emb_size_interm, num_spherical),
        # (num_edges, num_spherical, Kmax)
        num_edges = sph.shape[0]

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = torch.max(idx_agg_inner) + 1
        m_padded = m.new_zeros(num_edges, Kmax, self.emb_size_in)
        m_padded[idx_agg_outer, idx_agg_inner] = m
        # (num_quadruplets/num_triplets, emb_size_in) -> (num_edges, Kmax, emb_size_in)

        sph_m = torch.matmul(sph, m_padded)
        # (num_edges, num_spherical, emb_size_in)

        if idx_agg2_outer is not None:
            Kmax2 = torch.max(idx_agg2_inner) + 1
            sph_m_padded = sph_m.new_zeros(
                agg2_out_size, Kmax2, sph_m.shape[1], sph_m.shape[2]
            )
            sph_m_padded[idx_agg2_outer, idx_agg2_inner] = sph_m
            # (num_atoms, Kmax2, num_spherical, emb_size_in)
            sph_m_padded = sph_m_padded.reshape(agg2_out_size, -1, sph_m.shape[-1])
            # (num_atoms, Kmax2 * num_spherical, emb_size_in)

            rad_W1_sph_m = rad_W1 @ sph_m_padded
            # (num_atoms, emb_size_interm, emb_size_in)
        else:
            # MatMul: mul + sum over num_spherical
            rad_W1_sph_m = torch.matmul(rad_W1, sph_m)
            # (num_edges, emb_size_interm, emb_size_in)

        # Bilinear: Sum over emb_size_interm and emb_size_in
        return self.bilinear(rad_W1_sph_m.reshape(-1, rad_W1_sph_m.shape[1:].numel()))
        # (num_edges/num_atoms, emb_size_out)
