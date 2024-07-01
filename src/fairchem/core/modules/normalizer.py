"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.datasets import data_list_collater

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class Normalizer(nn.Module):
    """Normalize/denormalize a tensor and optionally add a atom reference offset."""

    def __init__(
        self,
        mean: float | torch.Tensor = 0.0,
        std: float | torch.Tensor = 1.0,
    ):
        """tensor is taken as a sample to calculate the mean and std"""
        super().__init__()

        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(std, float):
            std = torch.tensor(std)

        self.register_buffer(name="mean", tensor=mean)
        self.register_buffer(name="std", tensor=std)

    @torch.autocast(device_type="cuda", enabled=False)
    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    @torch.autocast(device_type="cuda", enabled=False)
    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def forward(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return self.denorm(normed_tensor)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # check if state dict is legacy state dicts
        if isinstance(state_dict["mean"], float):
            state_dict.update({k: torch.tensor(state_dict[k]) for k in ("mean", "std")})

        super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_normalizer(
    file: str | Path | None = None,
    state_dict: dict | None = None,
    tensor: torch.Tensor | None = None,
    mean: float | torch.Tensor | None = None,
    std: float | torch.Tensor | None = None,
) -> Normalizer:
    """Build a target data normalizers with optional atom ref

    Args:
        file (str or Path): path to pt or npz file.
        state_dict (dict): a state dict for Normalizer module
        tensor (Tensor): a tensor with target values used to compute mean and std
        mean (float | Tensor): mean of target data
        std (float | Tensor): std of target data
        data_loader (DataLoader): a data loader used to estimate mean and std
        num_batches (int): the number of batches used. If not given all batches are used.

    Returns:
        Normalizer
    """
    # path takes priority if given
    if file is not None:
        try:
            # try to load a Normalizer pt file
            state_dict = torch.load(file)
        except RuntimeError:  # try to read an npz file
            # try to load an NPZ file
            values = np.load(file)
            mean = values.get("mean")
            std = values.get("std")

    if state_dict is not None:
        normalizer = Normalizer()
        normalizer.load_state_dict(state_dict)
        return normalizer

    # if not then read target value tensor
    if tensor is not None and mean is None and std is None:
        mean = torch.mean(tensor)
        std = torch.std(tensor)
    elif mean is not None and std is not None:
        mean = torch.tensor(mean)
        std = torch.tensor(std)

    # if mean and std are still None than raise an error
    if mean is None or std is None:
        raise ValueError(
            "Incorrect inputs. One of the following sets of inputs must be given: ",
            "a file path to a .pt or .npz file, or mean and std values, or a tensor of target values",
        )

    return Normalizer(mean=mean, std=std)


def fit_normalizers(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    element_references: dict | None = None,
    num_batches: int | None = None,
    num_workers: int = 1,
    shuffle: bool = True,
) -> dict[str, Normalizer]:
    """Estimate mean and std from data to create normalizers

    Args:
        targets: list of target names
        dataset: data set to fit linear references with
        batch_size: size of batch
        element_references:
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader
        shuffle: whether to shuffle when loading the dataset

    Returns:
        dict of normalizer objects
    """
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        pin_memory=True,
    )

    num_batches = num_batches if num_batches is not None else len(data_loader)
    if num_batches > len(data_loader):
        logging.warning(
            f"The give num_batches {num_batches} is larger than total batches of size {batch_size} in dataset. "
            f"Will ignore num_batches and use the whole dataset."
        )
        num_batches = len(data_loader)

    element_references = element_references or {}
    target_vectors = defaultdict(list)

    logging.info(
        f"Estimating mean and std for normalization using {num_batches * batch_size} samples in {num_batches} batches of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Estimating mean and std"
    ):
        if i == num_batches:
            break

        for target in targets:
            target_vector = batch[target]
            if target in element_references:
                target_vector = element_references[target].dereference(
                    target_vector, batch, reshaped=False
                )
            target_vectors[target].append(target_vector)

    normalizers = {}
    for target in targets:
        target_vector = torch.cat(target_vectors[target], dim=0)
        normalizers[target] = create_normalizer(tensor=target_vector)

    return normalizers
