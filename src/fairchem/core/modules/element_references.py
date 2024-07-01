"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.datasets import data_list_collater

if TYPE_CHECKING:
    from pathlib import Path

    from torch_geometric.data import Batch


class LinearReference(nn.Module):
    """Compute a linear reference for target scalar properties"""

    def __init__(
        self,
        element_references: torch.Tensor | None = None,
        max_num_elements: int = 118,
    ):
        """
        Args:
            element_references (Tensor): tensor with linear reference values
            max_num_elements (int): max number of elements - 118 is a stretch
        """
        super().__init__()
        self.register_buffer(
            name="element_references",
            tensor=element_references
            if element_references is not None
            else torch.zeros(max_num_elements + 1),
        )

    def _apply_refs(
        self, target: torch.Tensor, batch: Batch, sign: int, reshaped: bool = True
    ) -> torch.Tensor:
        """Apply references batch-wise"""
        indices = batch.atomic_numbers.to(
            dtype=torch.int, device=self.element_references.device
        )
        elemrefs = sign * self.element_references[indices].to(dtype=target.dtype)
        # this option should not exist, all tensors should have compatible shapes in dataset and trainer outputs
        if reshaped:
            elemrefs = elemrefs.view(batch.natoms.sum(), -1)

        return target.index_add(0, batch.batch, elemrefs)

    @torch.autocast(device_type="cuda", enabled=False)
    def dereference(
        self, target: torch.Tensor, batch: Batch, reshaped: bool = True
    ) -> torch.Tensor:
        """Remove linear references"""
        return self._apply_refs(target, batch, -1, reshaped=reshaped)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self, target: torch.Tensor, batch: Batch, reshaped: bool = True
    ) -> torch.Tensor:
        """Add linear references"""
        return self._apply_refs(target, batch, 1, reshaped=reshaped)


def create_element_references(
    file: str | Path | None = None,
    state_dict: dict | None = None,
) -> LinearReference:
    """Create an element reference module.

    Currently only linear references are supported.

    Args:
        type (str): type of reference (only linear implemented)
        file (str or Path): path to pt or npz file
        state_dict (dict): a state dict of a element reference module

    Returns:
        LinearReference
    """

    # path takes priority if given
    if file is not None:
        try:
            # try to load a pt file
            state_dict = torch.load(file)
        except RuntimeError:  # try to read an npz file
            state_dict = {}
            with np.load(file) as values:
                # legacy linref files:
                if "coeff" in values:
                    state_dict["element_references"] = torch.tensor(values["coeff"])
                else:
                    state_dict["element_references"] = torch.tensor(
                        values["element_references"]
                    )

    if "element_references" not in state_dict:
        raise RuntimeError("Unable to load linear element references!")

    return LinearReference(element_references=state_dict["element_references"])


def fit_linear_references(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    num_batches: int | None = None,
    num_workers: int = 1,
    max_num_elements: int = 118,
    driver: str | None = None,
    shuffle: bool = True,
) -> dict[str, LinearReference]:
    """Fit a set linear references for a list of targets using a given number of batches.

    Args:
        targets: list of target names
        dataset: data set to fit linear references with
        batch_size: size of batch
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader
        max_num_elements: max number of elements in dataset. If not given will use an ambitious value of 118
        driver: backend used to solve linear system. See torch.linalg.lstsq docs.
        shuffle: whether to shuffle when loading the dataset

    Returns:
        dict of fitted LinearReference objects
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
            f"Will ignore num_batches and use the whole dataset"
        )
        num_batches = len(data_loader)

    max_num_elements += 1  # + 1 since H starts at index 1
    # solving linear system happens on CPU, which allows handling poorly conditioned and
    # rank deficient matrices, unlike torch lstsq on GPU
    composition_matrix = torch.zeros(
        num_batches * batch_size,
        max_num_elements,
    )

    # This only works with scalar properties
    target_vectors = {
        target: torch.zeros(num_batches * batch_size) for target in targets
    }

    logging.info(
        f"Fitting linear references using {num_batches * batch_size} samples in {num_batches} batches of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Fitting linear references"
    ):
        if i == num_batches:
            break

        next_batch_size = (
            batch.energy.shape[0] if i == len(data_loader) - 1 else batch_size
        )
        for target in targets:
            target_vectors[target][
                i * batch_size : i * batch_size + next_batch_size
            ] = batch[target].to(torch.float64)
        for j, data in enumerate(batch.to_data_list()):
            composition_matrix[i * batch_size + j] = torch.bincount(
                data.atomic_numbers.int(),
                minlength=max_num_elements,
            ).to(torch.float64)

    # reduce the composition matrix to only features that are non-zero to improve rank
    mask = composition_matrix.sum(axis=0) != 0.0
    reduced_composition_matrix = composition_matrix[:, mask]
    elementrefs = {}
    for target in targets:
        coeffs = torch.zeros(max_num_elements)
        lstsq = torch.linalg.lstsq(
            reduced_composition_matrix, target_vectors[target], driver=driver
        )
        coeffs[mask] = lstsq.solution
        elementrefs[target] = LinearReference(coeffs)

    return elementrefs
