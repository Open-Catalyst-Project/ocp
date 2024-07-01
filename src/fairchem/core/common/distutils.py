"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from fairchem.core.common.typing import none_throws

T = TypeVar("T")
TIMEOUT = timedelta(minutes=30)


def os_environ_get_or_throw(x: str) -> str:
    if x not in os.environ:
        raise RuntimeError(f"Could not find {x} in ENV variables")
    return none_throws(os.environ.get(x))


def setup(config) -> None:
    if config["submit"]:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config["init_method"] = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config["distributed_port"],
                )
                nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os_environ_get_or_throw("SLURM_NTASKS"))
                    nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config["world_size"] % nnodes == 0
                    gpus_per_node = config["world_size"] // nnodes
                    node_id = int(os_environ_get_or_throw("SLURM_NODEID"))
                    config["rank"] = node_id * gpus_per_node
                    config["local_rank"] = 0
                else:
                    assert ntasks_per_node == config["world_size"] // nnodes
                    config["rank"] = int(os_environ_get_or_throw("SLURM_PROCID"))
                    config["local_rank"] = int(os_environ_get_or_throw("SLURM_LOCALID"))

                logging.info(
                    f"Init: {config['init_method']}, {config['world_size']}, {config['rank']}"
                )

                # ensures GPU0 does not have extra context/higher peak memory
                torch.cuda.set_device(config["local_rank"])

                dist.init_process_group(
                    backend=config["distributed_backend"],
                    init_method=config["init_method"],
                    world_size=config["world_size"],
                    rank=config["rank"],
                    timeout=TIMEOUT,
                )
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass
    elif config["summit"]:
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        get_master = (
            "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)"
        ).format(os.environ["LSB_DJOB_HOSTFILE"])
        os.environ["MASTER_ADDR"] = str(
            subprocess.check_output(get_master, shell=True)
        )[2:-3]
        os.environ["MASTER_PORT"] = "23456"
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        # NCCL and MPI initialization
        dist.init_process_group(
            backend="nccl",
            rank=world_rank,
            world_size=world_size,
            init_method="env://",
            timeout=TIMEOUT,
        )
    else:
        config["local_rank"] = int(os.environ.get("LOCAL_RANK", config["local_rank"]))
        dist.init_process_group(backend="nccl", timeout=TIMEOUT)


def cleanup() -> None:
    dist.destroy_process_group()


def initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(
    tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def broadcast_object_list(
    object_list: list[Any], src: int, group=dist.group.WORLD, device: str | None = None
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast_object_list(object_list, src, group, device)


def all_reduce(
    data, group=dist.group.WORLD, average: bool = False, device=None
) -> torch.Tensor:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(data, group=dist.group.WORLD, device=None) -> list[torch.Tensor]:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.cuda(device)
    tensor_list = [tensor.new_zeros(tensor.shape) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor, group=group)
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        result = tensor_list
    return result


def gather_objects(data: T, group: dist.ProcessGroup = dist.group.WORLD) -> list[T]:
    """Gather a list of pickleable objects into rank 0"""
    if get_world_size() == 1:
        return [data]

    output = [None for _ in range(get_world_size())] if is_master() else None
    dist.gather_object(data, output, group=group, dst=0)
    return output
