"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copied (and improved) from:
https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py (MIT license)
"""

from __future__ import annotations

import copy
import weakref
from typing import TYPE_CHECKING

import torch

from fairchem.core.common.typing import none_throws

if TYPE_CHECKING:
    from collections.abc import Iterable


# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        use_num_updates: bool = False,
    ) -> None:
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates: int | None = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params: list[torch.nn.Parameter] = []
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters if p.requires_grad]

    def _get_parameters(
        self, parameters: Iterable[torch.nn.Parameter] | None
    ) -> Iterable[torch.nn.Parameter]:
        none_msg = (
            "(One of) the parameters with which this "
            "ExponentialMovingAverage "
            "was initialized no longer exists (was garbage collected);"
            " please either provide `parameters` explicitly or keep "
            "the model to which they belong from being garbage "
            "collected."
        )
        if parameters is None:
            return [none_throws(p(), none_msg) for p in self._params_refs]
        else:
            return [p for p in parameters if p.requires_grad]

    def update(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = param - s_param
                s_param.add_(tmp, alpha=one_minus_decay)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored. If `None`, the parameters of with which this
            `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter] | None = None) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters. If `None`, the
            parameters with which this `ExponentialMovingAverage` was
            initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(
            self.num_updates, int
        ), "Invalid num_updates"

        assert isinstance(
            state_dict["shadow_params"], list
        ), "shadow_params must be a list"
        self.shadow_params = [
            p.to(self.shadow_params[i].device)
            for i, p in enumerate(state_dict["shadow_params"])
        ]
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        assert isinstance(
            state_dict["collected_params"], list
        ), "collected_params must be a list"
        # collected_params is empty at initialization,
        # so use shadow_params for device instead
        self.collected_params = [
            p.to(self.shadow_params[i].device)
            for i, p in enumerate(state_dict["collected_params"])
        ]
        assert all(
            isinstance(p, torch.Tensor) for p in self.collected_params
        ), "collected_params must all be Tensors"
