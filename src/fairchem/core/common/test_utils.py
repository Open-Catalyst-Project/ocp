from __future__ import annotations

import multiprocessing
import os
import pdb
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port

from fairchem.core.common.gp_utils import setup_gp


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess/23654936#23654936

    example usage to debug a torch distributed run on rank 0:
    if torch.distributed.get_rank() == 0:
        from fairchem.core.common.test_utils import ForkedPdb
        ForkedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin") as f:
                sys.stdin = f
                pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@dataclass
class PGConfig:
    backend: str
    world_size: int
    gp_group_size: int = 1
    port: str = "12345"
    use_gp: bool = True

def spawn_multi_process(
    config: PGConfig,
    test_method: callable,
    *test_method_args: Any,
    **test_method_kwargs: Any,
) -> list[Any]:
    """
    Spawn single node, multi-rank function.
    Uses localhost and free port to communicate.

    Args:
        world_size: number of processes
        backend: backend to use. for example, "nccl", "gloo", etc
        test_method: callable to spawn. first 3 arguments are rank, world_size and mp output dict
        test_method_args: args for the test method
        test_method_kwargs: kwargs for the test method

    Returns:
        A list, l, where l[i] is the return value of test_method on rank i
    """
    manager = multiprocessing.Manager()
    mp_output_dict = manager.dict()

    port = str(get_free_port())
    config.port = port
    torch.multiprocessing.spawn(
        # torch.multiprocessing.spawn sends rank as the first param
        # https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
        _init_pg_and_rank_and_launch_test,
        args=(
            config,
            mp_output_dict,
            test_method,
            test_method_args,
            test_method_kwargs,
        ),
        nprocs=config.world_size,
    )

    return [mp_output_dict[i] for i in range(config.world_size)]


def _init_pg_and_rank_and_launch_test(
    rank: int,
    pg_setup_params: PGConfig,
    mp_output_dict: dict[int, object],
    test_method: callable,
    args: list[object],
    kwargs: dict[str, object],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = pg_setup_params.port
    os.environ["WORLD_SIZE"] = str(pg_setup_params.world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # setup default process group
    dist.init_process_group(
        rank=rank,
        world_size=pg_setup_params.world_size,
        backend=pg_setup_params.backend,
        timeout=timedelta(seconds=10),  # setting up timeout for distributed collectives
    )
    # setup gp
    if pg_setup_params.use_gp:
        config = {"gp_gpus": pg_setup_params.gp_group_size, "distributed_backend": pg_setup_params.backend}
        setup_gp(config)
    mp_output_dict[rank] = test_method(*args, **kwargs)  # pyre-fixme
