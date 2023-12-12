"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from itertools import chain
from typing import List

import torch
from torch_geometric.data.data import BaseData

from ocpmodels.datasets import data_list_collater


class OCPDataParallel(torch.nn.DataParallel):
    use_cpu: bool

    def __init__(
        self, module, output_device: torch.device, num_gpus: int
    ) -> None:
        if num_gpus < 0:
            raise ValueError("# GPUs must be positive.")
        if num_gpus > torch.cuda.device_count():
            raise ValueError("# GPUs specified larger than available")

        self.src_device = torch.device(output_device)

        self.use_cpu = False
        if num_gpus == 0:
            self.use_cpu = True
        elif num_gpus == 1:
            device_ids = [self.src_device]
        else:
            if (
                self.src_device.type == "cuda"
                and self.src_device.index >= num_gpus
            ):
                raise ValueError("Main device must be less than # of GPUs")
            device_ids = list(range(num_gpus))

        if self.use_cpu:
            super(torch.nn.DataParallel, self).__init__()
            self.module = module

        else:
            super(OCPDataParallel, self).__init__(
                module=module,
                device_ids=device_ids,
                output_device=self.src_device,
            )

    def forward(self, batch_list, **kwargs):
        if self.use_cpu:
            return self.module(batch_list[0])

        if len(self.device_ids) == 1:
            return self.module(
                batch_list[0].to(f"cuda:{self.device_ids[0]}"), **kwargs
            )

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    (
                        "Module must have its parameters and buffers on device "
                        "{} but found one of them on device {}."
                    ).format(self.src_device, t.device)
                )

        inputs = [
            batch.to(f"cuda:{self.device_ids[i]}")
            for i, batch in enumerate(batch_list)
        ]
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class ParallelCollater:
    def __init__(self, num_gpus: int, otf_graph: bool = False) -> None:
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph

    def __call__(self, data_list: List[BaseData]) -> List[BaseData]:
        if self.num_gpus in [0, 1]:  # adds cpu-only case
            batch = data_list_collater(data_list, otf_graph=self.otf_graph)
            return [batch]

        else:
            num_devices = min(self.num_gpus, len(data_list))

            count = torch.tensor([data.num_nodes for data in data_list])
            cumsum = count.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = (
                num_devices * cumsum.to(torch.float) / cumsum[-1].item()
            )
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                data_list_collater(data_list[split[i] : split[i + 1]])
                for i in range(len(split) - 1)
            ]
