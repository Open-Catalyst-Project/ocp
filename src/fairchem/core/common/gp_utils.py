"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import distributed as dist

"""
Functions to support graph parallel training.
This is based on the Megatron-LM implementation:
https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/initialize.py
"""

########## INITIALIZATION ##########

_GRAPH_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None


def ensure_div(a: int, b: int) -> None:
    assert a % b == 0


def divide_and_check_no_remainder(a: int, b: int) -> int:
    ensure_div(a, b)
    return a // b


def setup_gp(config) -> None:
    gp_size = config["gp_gpus"]
    backend = config["distributed_backend"]
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()

    gp_size = min(gp_size, world_size)
    ensure_div(world_size, gp_size)
    dp_size = world_size // gp_size
    rank = dist.get_rank()

    if rank == 0:
        logging.info(f"> initializing graph parallel with size {gp_size}")
        logging.info(f"> initializing ddp with size {dp_size}")

    groups = torch.arange(world_size).reshape(dp_size, gp_size)
    found = [x.item() for x in torch.where(groups == rank)]

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(gp_size):
        group = dist.new_group(groups[:, j].tolist(), backend=backend)
        if j == found[1]:
            _DATA_PARALLEL_GROUP = group
    global _GRAPH_PARALLEL_GROUP
    assert _GRAPH_PARALLEL_GROUP is None, "graph parallel group is already initialized"
    for i in range(dp_size):
        group = dist.new_group(groups[i, :].tolist(), backend=backend)
        if i == found[0]:
            _GRAPH_PARALLEL_GROUP = group


def cleanup_gp() -> None:
    dist.destroy_process_group(_DATA_PARALLEL_GROUP)
    dist.destroy_process_group(_GRAPH_PARALLEL_GROUP)


def initialized() -> bool:
    return _GRAPH_PARALLEL_GROUP is not None


def get_dp_group():
    return _DATA_PARALLEL_GROUP


def get_gp_group():
    return _GRAPH_PARALLEL_GROUP


def get_dp_rank() -> int:
    return dist.get_rank(group=get_dp_group())


def get_gp_rank() -> int:
    return dist.get_rank(group=get_gp_group())


def get_dp_world_size() -> int:
    return dist.get_world_size(group=get_dp_group())


def get_gp_world_size() -> int:
    return 1 if not initialized() else dist.get_world_size(group=get_gp_group())


########## DIST METHODS ##########


def pad_tensor(
    tensor: torch.Tensor, dim: int = -1, target_size: int | None = None
) -> torch.Tensor:
    size = tensor.size(dim)
    if target_size is None:
        world_size = get_gp_world_size()
        pad_size = 0 if size % world_size == 0 else world_size - size % world_size
    else:
        pad_size = target_size - size
    if pad_size == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.empty(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)


def trim_tensor(tensor: torch.Tensor, sizes: torch.Tensor | None = None, dim: int = 0):
    size = tensor.size(dim)
    world_size = get_gp_world_size()
    if size % world_size == 0:
        return tensor, sizes
    trim_size = size - size % world_size
    if dim == 0:
        tensor = tensor[:trim_size]
    elif dim == 1:
        tensor = tensor[:, :trim_size]
    else:
        raise ValueError
    if sizes is not None:
        sizes[-1] = sizes[-1] - size % world_size
    return tensor, sizes


def _split_tensor(
    tensor: torch.Tensor,
    num_parts: int,
    dim: int = -1,
    contiguous_chunks: bool = False,
):
    part_sizes = [
        len(part) for part in np.array_split(np.zeros(tensor.size(dim)), num_parts)
    ]
    tensor_list = torch.split(tensor, part_sizes, dim=dim)
    if contiguous_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _reduce(ctx: Any, input: torch.Tensor) -> torch.Tensor:
    group = get_gp_group()
    if ctx:
        ctx.mark_dirty(input)
    if dist.get_world_size(group) == 1:
        return input
    dist.all_reduce(input, group=group)
    return input


def _split(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input
    input_list = _split_tensor(input, world_size, dim=dim)
    return input_list[rank].contiguous()


def _gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input
    dist.all_gather(tensor_list, input, group=group)
    return torch.cat(tensor_list, dim=dim).contiguous()


def _gather_with_padding(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input

    # Gather sizes
    size_list = [
        torch.empty(1, device=input.device, dtype=torch.long) for _ in range(world_size)
    ]
    size = torch.tensor([input.size(dim)], device=input.device, dtype=torch.long)
    size_list[rank] = size
    dist.all_gather(size_list, size, group=group)

    # Gather the inputs
    max_size = int(max([size.item() for size in size_list]))
    input = pad_tensor(input, dim, max_size)
    shape = list(input.shape)
    shape[dim] = max_size
    tensor_list = [
        torch.empty(shape, device=input.device, dtype=input.dtype)
        for _ in range(world_size)
    ]
    tensor_list[rank] = input
    dist.all_gather(tensor_list, input, group=group)

    # Trim and cat
    if dim == 0:
        tensor_list = [tensor[:size] for tensor, size in zip(tensor_list, size_list)]
    elif dim == 1:
        tensor_list = [tensor[:, :size] for tensor, size in zip(tensor_list, size_list)]
    else:
        raise ValueError
    return torch.cat(tensor_list, dim=dim).contiguous()


class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _reduce(None, grad_output)


class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return _reduce(ctx, input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        world_size = 1
        return grad_output.mul_(world_size)


class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        result = _split(input, dim)
        ctx.save_for_backward(torch.tensor(dim))
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        world_size = 1
        return (
            _gather_with_padding(grad_output, dim.item()).div_(world_size),
            None,
        )


class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(dim))
        return _gather_with_padding(input, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        result = _split(grad_output, dim.item())
        world_size = 1
        return result.mul_(world_size), None


def copy_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return CopyToModelParallelRegion.apply(input)


def reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ReduceFromModelParallelRegion.apply(input)


def scatter_to_model_parallel_region(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScatterToModelParallelRegion.apply(input, dim)


def gather_from_model_parallel_region(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegion.apply(input, dim)
