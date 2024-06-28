"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.modules.scaling import ScaleFactor

from .basis_utils import get_sph_harm_basis
from .radial_basis import GaussianBasis, RadialBasis


class CircularBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis

    Arguments
    ---------
    num_spherical: int
        Number of basis functions. Controls the maximum frequency.
    radial_basis: RadialBasis
        Radial basis function.
    cbf: dict
        Name and hyperparameters of the circular basis function.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: dict,
        scale_basis: bool = False,
    ) -> None:
        super().__init__()

        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_cbf = ScaleFactor()

        cbf_name = cbf["name"].lower()
        cbf_hparams = cbf.copy()
        del cbf_hparams["name"]

        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianBasis(
                start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams
            )
        elif cbf_name == "spherical_harmonics":
            self.cosφ_basis = get_sph_harm_basis(num_spherical, zero_m_only=True)
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca, cosφ_cab):
        rad_basis = self.radial_basis(D_ca)  # (num_edges, num_radial)
        cir_basis = self.cosφ_basis(cosφ_cab)  # (num_triplets, num_spherical)

        if self.scale_basis:
            cir_basis = self.scale_cbf(cir_basis)

        return rad_basis, cir_basis
        # (num_edges, num_radial), (num_triplets, num_spherical)


class SphericalBasisLayer(torch.nn.Module):
    """
    3D Fourier Bessel Basis

    Arguments
    ---------
    num_spherical: int
        Number of basis functions. Controls the maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions.
    sbf: dict
        Name and hyperparameters of the spherical basis function.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        sbf: dict,
        scale_basis: bool = False,
    ) -> None:
        super().__init__()

        self.num_spherical = num_spherical
        self.radial_basis = radial_basis

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_sbf = ScaleFactor()

        sbf_name = sbf["name"].lower()
        sbf_hparams = sbf.copy()
        del sbf_hparams["name"]

        if sbf_name == "spherical_harmonics":
            self.spherical_basis = get_sph_harm_basis(num_spherical, zero_m_only=False)

        elif sbf_name == "legendre_outer":
            circular_basis = get_sph_harm_basis(num_spherical, zero_m_only=True)
            self.spherical_basis = lambda cosφ, ϑ: (
                circular_basis(cosφ)[:, :, None]
                * circular_basis(torch.cos(ϑ))[:, None, :]
            ).reshape(cosφ.shape[0], -1)

        elif sbf_name == "gaussian_outer":
            self.circular_basis = GaussianBasis(
                start=-1, stop=1, num_gaussians=num_spherical, **sbf_hparams
            )
            self.spherical_basis = lambda cosφ, ϑ: (
                self.circular_basis(cosφ)[:, :, None]
                * self.circular_basis(torch.cos(ϑ))[:, None, :]
            ).reshape(cosφ.shape[0], -1)

        else:
            raise ValueError(f"Unknown spherical basis function '{sbf_name}'.")

    def forward(self, D_ca, cosφ_cab, θ_cabd):
        rad_basis = self.radial_basis(D_ca)
        sph_basis = self.spherical_basis(cosφ_cab, θ_cabd)
        # (num_quadruplets, num_spherical**2)

        if self.scale_basis:
            sph_basis = self.scale_sbf(sph_basis)

        return rad_basis, sph_basis
        # (num_edges, num_radial), (num_quadruplets, num_spherical**2)
