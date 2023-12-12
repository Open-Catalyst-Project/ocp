import heapq
from logging import getLogger
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numba
import numpy as np
import torch
import torch.distributed
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
from typing_extensions import override

log = getLogger(__name__)


def _all_gather(tensor: torch.Tensor, device: Optional[torch.device] = None):
    gathered = [
        torch.zeros_like(tensor, device=device)
        for _ in range(torch.distributed.get_world_size())
    ]
    _ = torch.distributed.all_gather(gathered, tensor)
    return gathered


@numba.njit
def _balanced_partition(sizes: np.ndarray, num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap = []
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


class UnsupportedDatasetError(ValueError):
    pass


@runtime_checkable
class DatasetWithSizes(Protocol):
    def data_sizes(self, indices: List[int]) -> np.ndarray:
        ...


def _ensure_supported(dataset: Any):
    if not isinstance(dataset, Dataset):
        raise UnsupportedDatasetError(
            "BalancedBatchSampler requires a dataset."
        )

    if not isinstance(dataset, DatasetWithSizes):
        raise UnsupportedDatasetError(
            "BalancedBatchSampler requires a dataset that implements the `data_sizes` method."
        )

    log.debug(f"BalancedBatchSampler: Resolved dataset to {type(dataset)}")
    return dataset


class BalancedBatchSampler(BatchSampler):
    @property
    def distributed_sampler(self):
        if not isinstance(self.sampler, DistributedSampler):
            raise ValueError(
                f"Sampler must be a DistributedSampler, got {type(self.sampler)}"
            )
        return self.sampler

    def _data_sizes(self, batch_idx: List[int]):
        dataset = self.distributed_sampler.dataset
        try:
            dataset = _ensure_supported(dataset)
            return dataset.data_sizes(batch_idx)
        except UnsupportedDatasetError as e:
            if self.on_error == "raise":
                raise e
            elif self.on_error == "warn_and_balance":
                log.warning(
                    f"Failed to get data sizes from metadata, loading data to get sizes (THIS IS SLOW). {e}"
                )
                return np.array([dataset[idx].num_nodes for idx in batch_idx])
            elif self.on_error == "warn_and_no_balance":
                log.warning(
                    f"Failed to get data sizes, falling back to uniform partitioning. {e}"
                )
                return None
            else:
                raise ValueError(f"Unknown on_error={self.on_error}") from e

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: torch.device,
        mode: Union[str, bool] = "atoms",
        shuffle: bool = True,
        on_error: Literal[
            "warn_and_balance", "warn_and_no_balance", "raise"
        ] = "raise",
        drop_last: bool = False,
    ):
        """
        Initializes a BalancedBatchSampler object.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of each batch.
            num_replicas (int): The number of processes participating in distributed training.
            rank (int): The rank of the current process in distributed training.
            device (torch.device): The device to use for the batches.
            mode (Union[str, bool], optional): The mode to use for balancing the batches. Defaults to "atoms".
            shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.
            on_error (Literal["warn_and_balance", "warn_and_no_balance", "raise"], optional): The action to take when an error occurs (i.e., when we have an invalid dataset). Defaults to "raise".
                - "warn_and_balance": Raise a warning and balance the batch by manually loading the data samples and counting the number of nodes (this is slow).
                - "warn_and_no_balance": Raise a warning and do not do any balancing.
                - "raise": Raise an error.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        """
        self.disabled = False

        if mode is False:
            log.warning(f"Disabled BalancedBatchSampler because {mode=}.")
            self.disabled = True

        if num_replicas == 1:
            log.warning(
                f"Disabled BalancedBatchSampler because {num_replicas=}."
            )
            self.disabled = True

        if isinstance(mode, str) and mode != "atoms":
            raise ValueError(f"Only mode='atoms' is supported, got {mode=}.")

        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)

        self._device = device
        self.on_error = on_error

        log.info(
            f"Created BalancedBatchSampler with {sampler=}, {batch_size=}, {drop_last=}"
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    @staticmethod
    def _dist_enabled():
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    def _should_disable(self):
        return self.disabled or not self._dist_enabled()

    @override
    def __iter__(self):
        if self._should_disable():
            yield from super().__iter__()
            return

        for batch_idx in super().__iter__():
            sizes = self._data_sizes(batch_idx)
            if sizes is None:  # on_error == "warn_and_no_balance" is set
                yield batch_idx
                continue

            idx_sizes = torch.stack(
                [
                    torch.tensor(batch_idx, device=self._device),
                    torch.tensor(sizes, device=self._device),
                ]
            )
            idx_sizes_all = _all_gather(idx_sizes, device=self._device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = _balanced_partition(
                sizes_all.numpy(),
                num_parts=self.distributed_sampler.num_replicas,
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[
                local_idx_balanced[self.distributed_sampler.rank]
            ].tolist()
