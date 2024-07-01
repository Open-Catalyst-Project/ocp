"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import torch
from torch_geometric.data import Batch

from fairchem.core.common.typing import assert_is_instance
from fairchem.core.datasets.lmdb_dataset import data_list_collater

from .optimizers.lbfgs_torch import LBFGS
from .optimizers.optimizable import OptimizableBatch, OptimizableUnitCellBatch


def ml_relax(
    batch,
    model,
    steps: int,
    fmax: float,
    relax_opt: dict[str],
    relax_cell: bool = False,
    relax_volume: bool = False,
    save_full_traj: bool = True,
    device: str = "cuda:0",
    transform: torch.nn.Module | None = None,
    mask_converged: bool = True,
    cumulative_mask: bool = True,
):
    """Runs ML-based relaxations.

    Args:
        batch: a data batch object.
        model: a trainer object with model.q
        steps: Max number of steps in the structure relaxation.
        fmax: Structure relaxation terminates when the max force of the system is no bigger than fmax.
        relax_opt: Optimizer and corresponding parameters to be used for structure relaxations.
        relax_cell: if true will use stress predictions to relax crystallographic cell.
            The model given must predict stress
        relax_volume: if true will relax the cell isotropically. the given model must predict stress.
        save_full_traj: Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
        mask_converged: whether to mask batches where all atoms are below convergence threshold
        cumulative_mask: if true, once system is masked then it remains masked even if new predictions give forces
                above threshold, ie. once masked always masked. Note if this is used make sure to check convergence with
                the same fmax always
    """
    # if not pbc is set, ignore it when comparing batches
    if not hasattr(batch, "pbc"):
        OptimizableBatch.ignored_changes = {"pbc"}

    batches = deque([batch])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid

        if relax_cell or relax_volume:
            optimizable = OptimizableUnitCellBatch(
                batch,
                trainer=model,
                transform=transform,
                mask_converged=mask_converged,
                cumulative_mask=cumulative_mask,
                hydrostatic_strain=relax_volume,
            )
        else:
            optimizable = OptimizableBatch(
                batch,
                trainer=model,
                transform=transform,
                mask_converged=mask_converged,
                cumulative_mask=cumulative_mask,
            )

        # Run ML-based relaxation
        traj_dir = relax_opt.get("traj_dir")
        optimizer = LBFGS(
            optimizable_batch=optimizable,
            maxstep=relax_opt.get("maxstep", 0.2),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.2),
            alpha=relax_opt.get("alpha", 80.0),
            device=device,
            save_full_traj=save_full_traj,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
        )

        e: RuntimeError | None = None
        try:
            optimizer.run(fmax=fmax, steps=steps)
            relaxed_batches.append(batch)
        except RuntimeError as err:
            e = err
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside of except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise assert_is_instance(e, RuntimeError)
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    # reset for good measure
    OptimizableBatch.ignored_changes = {}

    return Batch.from_data_list(relaxed_batches)
