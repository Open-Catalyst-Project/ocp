from typing import List, TypeVar

import numpy as np
import pytest
from torch.utils.data import Dataset

from ocpmodels.common.balanced_batch_sampler import (
    BalancedBatchSampler,
    UnsupportedDatasetError,
)

DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_ATOMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

T_co = TypeVar("T_co", covariant=True)


@pytest.fixture
def valid_dataset():
    class _Dataset(Dataset[T_co]):
        def data_sizes(self, batch_idx: List[int]) -> np.ndarray:
            return np.array(SIZE_ATOMS)[batch_idx]

        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    yield _Dataset(DATA)


@pytest.fixture
def invalid_dataset():
    class _Dataset(Dataset):
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return _Dataset(DATA)


def test_invalid_mode(invalid_dataset) -> None:
    with pytest.raises(ValueError, match="Only mode='atoms' is supported"):
        _ = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="natoms",
            on_error="raise",
        )

    with pytest.raises(ValueError, match="Only mode='atoms' is supported"):
        _ = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="neighbors",
            on_error="raise",
        )


def test_invalid_dataset(invalid_dataset) -> None:
    with pytest.raises(UnsupportedDatasetError):
        sampler = BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            on_error="raise",
        )
        _ = sampler._data_sizes(list(range(len(SIZE_ATOMS))))


def test_valid_dataset(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="atoms",
        on_error="raise",
    )
    assert (
        sampler._data_sizes(list(range(len(SIZE_ATOMS))))
        == np.array(SIZE_ATOMS)
    ).all()


def test_disabled(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode=False,
        on_error="raise",
    )
    assert sampler._should_disable()


def test_single_node(valid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=1,
        device=None,
        mode="atoms",
        on_error="raise",
    )
    assert sampler._should_disable()
