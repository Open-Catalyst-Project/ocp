from __future__ import annotations

import pytest
import torch

from fairchem.core.common.gp_utils import (
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from fairchem.core.common.test_utils import PGConfig, spawn_multi_process


def _dummy_call(x):
    return x

@pytest.mark.parametrize("world_size, input, expected_output", [(1, 5, [5]), (3, 0, [0, 0, 0])]) # noqa: PT006
def test_basic_setup(world_size: int, input: torch.Tensor, expected_output: list):
    config = PGConfig(backend="gloo", world_size=world_size, gp_group_size=1, use_gp=True)
    output = spawn_multi_process(config, _dummy_call, input)
    assert output == expected_output

@pytest.mark.parametrize("world_size, gp_size, input, expected_output", # noqa: PT006
                         [(2, 1, torch.Tensor([0,1,2,3]), [torch.Tensor([0,1,2,3]), torch.Tensor([0,1,2,3])]),
                          (2, 2, torch.Tensor([0,1,2,3]), [torch.Tensor([0,1]), torch.Tensor([2,3])]),
                          (2, 2, torch.Tensor([0,1,2]), [torch.Tensor([0,1]), torch.Tensor([2])]),
                          (3, 3, torch.Tensor([0,1,2]), [torch.Tensor([0]), torch.Tensor([1]), torch.Tensor([2])])]
)
def test_scatter_tensors(world_size: int, gp_size: int, input: torch.Tesnor, expected_output: list):
    config = PGConfig(backend="gloo", world_size=world_size, gp_group_size=gp_size, use_gp=True)
    output = spawn_multi_process(config, scatter_to_model_parallel_region, input)
    for out, expected_out in zip(output, expected_output):
        assert torch.equal(out, expected_out)

def scatter_gather_fn(input: torch.Tensor, dim: int = 0):
    x = scatter_to_model_parallel_region(input, dim)
    return gather_from_model_parallel_region(x, dim)

@pytest.mark.parametrize("world_size, gp_size, input, expected_output", # noqa: PT006
                         [(2, 1, torch.Tensor([0,1,2,3]), [torch.Tensor([0,1,2,3]), torch.Tensor([0,1,2,3])]),
                          (2, 2, torch.Tensor([0,1,2,3]), [torch.Tensor([0,1,2,3]), torch.Tensor([0,1,2,3])]),
                          (2, 2, torch.Tensor([0,1,2]), [torch.Tensor([0,1,2]), torch.Tensor([0,1,2])]),
                          (3, 3, torch.Tensor([0,1,2]), [torch.Tensor([0,1,2]), torch.Tensor([0,1,2]), torch.Tensor([0,1,2])])]
)
def test_gather_tensors(world_size: int, gp_size: int, input: torch.Tesnor, expected_output: list):
    config = PGConfig(backend="gloo", world_size=world_size, gp_group_size=gp_size, use_gp=True)
    output = spawn_multi_process(config, scatter_gather_fn, input)
    for out, expected_out in zip(output, expected_output):
        assert torch.equal(out, expected_out)
