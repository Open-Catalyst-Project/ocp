"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import heapq
import logging
from pathlib import Path
from typing import List, Literal, Protocol, Tuple, Union, runtime_checkable

import numba
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import BatchSampler, DistributedSampler, Sampler
from torch_geometric.data import Batch, Data

from ocpmodels.common import distutils, gp_utils
from ocpmodels.datasets import data_list_collater


class OCPCollater:
    def __init__(self, otf_graph: bool = False) -> None:
        self.otf_graph = otf_graph

    def __call__(self, data_list: List[Data]) -> Batch:
        batch = data_list_collater(data_list, otf_graph=self.otf_graph)
        return batch


@numba.njit
def balanced_partition(sizes: npt.NDArray[np.int_], num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap: List[Tuple[List[int], List[int]]] = []
    for idx in sort_idx[:num_parts]:
        heap.append((sizes[idx], [idx]))
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + sizes[idx]
        new_idx = smallest_part[1] + [idx]
        heapq.heappush(heap, (new_size, new_idx))
    idx_balanced = [part[1] for part in heap]
    return idx_balanced


@runtime_checkable
class _HasMetadata(Protocol):
    @property
    def metadata_path(self) -> Path:
        ...


# mostly from https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset=dataset, **kwargs)

        self.start_iter = 0
        self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size not set for the sampler"
        logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        # also consider if we are dropping samples to make evenly divisble by
        # num_replicas or not
        common_chunk_size = len(self.dataset) // self.num_replicas
        num_samples = common_chunk_size + (
            self.rank < len(self.dataset) % self.num_replicas
        ) * (not self.drop_last)
        start_idx = common_chunk_size * self.rank + min(
            len(self.dataset) % self.num_replicas, self.rank
        ) * (not self.drop_last)

        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(num_samples, generator=g).tolist()
        indices = np.arange(start_idx, start_idx + num_samples)[
            shuffling
        ].tolist()
        # make sure we have correct number of samples per replica
        assert len(indices) == num_samples

        # resume the sampler
        return iter(indices[self.start_iter * self.batch_size :])

    def set_epoch_and_start_iteration(self, epoch, start_iter):
        self.set_epoch(epoch)
        self.start_iter = start_iter


class BalancedBatchSampler(Sampler):
    def _load_dataset(self, dataset, mode: Literal["atoms", "neighbors"]):
        errors: List[str] = []
        if not isinstance(dataset, _HasMetadata):
            errors.append(
                f"Dataset {dataset} does not have a metadata_path attribute."
            )
            return None, errors
        if not dataset.metadata_path.exists():
            errors.append(
                f"Metadata file {dataset.metadata_path} does not exist."
            )
            return None, errors

        key = {"atoms": "natoms", "neighbors": "neighbors"}[mode]
        sizes = np.load(dataset.metadata_path)[key]

        return sizes, errors

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: torch.device,
        mode: Union[str, bool] = "atoms",
        shuffle: bool = True,
        drop_last: bool = False,
        force_balancing: bool = False,
        throw_on_error: bool = False,
    ) -> None:
        if mode is True:
            mode = "atoms"

        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ("atoms", "neighbors"):
                raise ValueError(
                    f"Invalid mode {mode}. Must be one of 'atoms', 'neighbors', or a boolean."
                )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.device = device
        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.single_sampler = StatefulDistributedSampler(
            self.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            batch_size=batch_size,
        )
        self.batch_sampler = BatchSampler(
            self.single_sampler,
            batch_size,
            drop_last=drop_last,
        )

        self.sizes = None
        self.balance_batches = False

        if self.num_replicas <= 1:
            logging.info(
                "Batch balancing is disabled for single GPU training."
            )
            return

        if self.mode is False:
            logging.info(
                "Batch balancing is disabled because `optim.load_balancing` is `False`"
            )
            return

        self.sizes, errors = self._load_dataset(dataset, self.mode)
        if self.sizes is None:
            self.balance_batches = force_balancing
            if force_balancing:
                errors.append(
                    "BalancedBatchSampler has to load the data to  determine batch sizes, which incurs significant overhead! "
                    "You can disable balancing by setting `optim.load_balancing` to `False`."
                )
            else:
                errors.append(
                    "Batches will not be balanced, which can incur significant overhead!"
                )
        else:
            self.balance_batches = True

        if errors:
            msg = "BalancedBatchSampler: " + " ".join(errors)
            if throw_on_error:
                raise RuntimeError(msg)
            else:
                logging.warning(msg)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def set_epoch_and_start_iteration(
        self, epoch: int, start_iteration: int
    ) -> None:
        self.single_sampler.set_epoch_and_start_iteration(
            epoch, start_iteration
        )

    def __iter__(self):
        if not self.balance_batches:
            yield from self.batch_sampler
            return

        for batch_idx in self.batch_sampler:
            if self.sizes is None:
                # Unfortunately, we need to load the data to know the image sizes
                data_list = [self.dataset[idx] for idx in batch_idx]

                if self.mode == "atoms":
                    sizes = [data.num_nodes for data in data_list]
                elif self.mode == "neighbors":
                    sizes = [data.edge_index.shape[1] for data in data_list]
                else:
                    raise NotImplementedError(
                        f"Unknown load balancing mode: {self.mode}"
                    )
            else:
                sizes = [self.sizes[idx] for idx in batch_idx]

            idx_sizes = torch.stack(
                [torch.tensor(batch_idx), torch.tensor(sizes)]
            )
            idx_sizes_all = distutils.all_gather(idx_sizes, device=self.device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            if gp_utils.initialized():
                idx_sizes_all = torch.unique(input=idx_sizes_all, dim=1)
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = balanced_partition(
                sizes_all.numpy(), num_parts=self.num_replicas
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.rank]]
