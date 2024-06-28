"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch

from fairchem.core.common import gp_utils
from fairchem.core.models.base_layers import Dense


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size: int) -> None:
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(83, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z) -> torch.Tensor:
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        return self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        num_out_features: int,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(
            in_features, num_out_features, activation=activation, bias=False
        )

    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ) -> torch.Tensor:
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h = gp_utils.gather_from_model_parallel_region(h, dim=0)
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat([h_s, h_t, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        return self.dense(m_st)  # (nEdges, emb_size)
