"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch

from .initializers import he_orthogonal_init


class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        num_in_features: int
            Input embedding size.
        num_out_features: int
            Output embedding size.
        bias: bool
            True if use bias.
        activation: str
            Name of the activation function to use.
    """

    def __init__(
        self,
        num_in_features: int,
        num_out_features: int,
        bias: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(num_in_features, num_out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init) -> None:
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self._activation(x)


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._activation(x)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer: torch.nn.Module
            Class for the layers inside the residual block.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs
    ) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(
                    num_in_features=units,
                    num_out_features=units,
                    bias=False,
                    **layer_kwargs,
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.dense_mlp(input)
        x = input + x
        return x * self.inv_sqrt_2
