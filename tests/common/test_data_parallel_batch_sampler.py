import functools
import itertools
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from torch.utils.data import BatchSampler, Dataset, DistributedSampler, Sampler

from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    StatefulDistributedSampler,
)

DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SIZE_ATOMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SIZE_NEIGHBORS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

T_co = TypeVar("T_co", covariant=True)


@contextmanager
def _temp_file(name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / name


@pytest.fixture
def valid_path_dataset():
    class _Dataset(Dataset[T_co]):
        def __init__(self, data, fpath: Path) -> None:
            self.data = data
            self.metadata_path = fpath

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    with _temp_file("metadata.npz") as file:
        np.savez(
            natoms=np.array(SIZE_ATOMS),
            neighbors=np.array(SIZE_NEIGHBORS),
            file=file,
        )
        yield _Dataset(DATA, file)


@pytest.fixture
def invalid_path_dataset():
    class _Dataset(Dataset):
        def __init__(self, data) -> None:
            self.data = data
            self.metadata_path = Path("/tmp/does/not/exist.np")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return _Dataset(DATA)


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


def test_lowercase(invalid_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=invalid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="ATOMS",
        throw_on_error=False,
    )
    assert sampler.mode == "atoms"

    sampler = BalancedBatchSampler(
        dataset=invalid_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="NEIGHBORS",
        throw_on_error=False,
    )
    assert sampler.mode == "neighbors"


def test_invalid_mode(invalid_dataset) -> None:
    with pytest.raises(
        ValueError, match="Must be one of 'atoms', 'neighbors', or a boolean."
    ):
        BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="natoms",
            throw_on_error=True,
        )

    with pytest.raises(
        ValueError, match="Must be one of 'atoms', 'neighbors', or a boolean."
    ):
        BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="nneighbors",
            throw_on_error=True,
        )


def test_invalid_dataset(invalid_dataset) -> None:
    with pytest.raises(
        RuntimeError,
        match="does not have a metadata_path attribute. BalancedBatchSampler has to load the data to  determine batch sizes, which incurs significant overhead!",
    ):
        BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            throw_on_error=True,
            force_balancing=True,
        )
    with pytest.raises(
        RuntimeError,
        match="does not have a metadata_path attribute. Batches will not be balanced, which can incur significant overhead!",
    ):
        BalancedBatchSampler(
            dataset=invalid_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            throw_on_error=True,
            force_balancing=False,
        )


def test_invalid_path_dataset(invalid_path_dataset) -> None:
    with pytest.raises(
        RuntimeError,
        match="Metadata file .+ does not exist. BalancedBatchSampler has to load the data to  determine batch sizes, which incurs significant overhead!",
    ):
        BalancedBatchSampler(
            dataset=invalid_path_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            throw_on_error=True,
            force_balancing=True,
        )
    with pytest.raises(
        RuntimeError,
        match="Metadata file .+ does not exist. Batches will not be balanced, which can incur significant overhead!",
    ):
        BalancedBatchSampler(
            dataset=invalid_path_dataset,
            batch_size=1,
            rank=0,
            num_replicas=2,
            device=None,
            mode="atoms",
            throw_on_error=True,
            force_balancing=False,
        )


def test_valid_dataset(valid_path_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_path_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="atoms",
        throw_on_error=True,
    )
    assert (sampler.sizes == np.array(SIZE_ATOMS)).all()

    sampler = BalancedBatchSampler(
        dataset=valid_path_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode="neighbors",
        throw_on_error=True,
    )
    assert (sampler.sizes == np.array(SIZE_NEIGHBORS)).all()


def test_disabled(valid_path_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_path_dataset,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device=None,
        mode=False,
        throw_on_error=True,
    )
    assert sampler.balance_batches is False


def test_single_node(valid_path_dataset) -> None:
    sampler = BalancedBatchSampler(
        dataset=valid_path_dataset,
        batch_size=1,
        rank=0,
        num_replicas=1,
        device=None,
        mode="atoms",
        throw_on_error=True,
    )
    assert sampler.balance_batches is False


def test_stateful_distributed_sampler(valid_path_dataset) -> None:
    for batch_size in range(1, 4):
        sampler = StatefulDistributedSampler(
            dataset=valid_path_dataset,
            batch_size=batch_size,
            rank=0,
            num_replicas=1,
            seed=0,
        )
        original_order = list(sampler)

        offset_step = 2
        loaded_sampler = StatefulDistributedSampler(
            dataset=valid_path_dataset,
            batch_size=batch_size,
            rank=0,
            seed=0,
            num_replicas=1,
        )
        loaded_sampler.set_epoch_and_start_iteration(0, offset_step)
        assert (
            list(loaded_sampler) == original_order[offset_step * batch_size :]
        )

        diff_sampler = StatefulDistributedSampler(
            dataset=valid_path_dataset,
            batch_size=batch_size,
            rank=0,
            num_replicas=1,
            seed=1,
        )
        assert list(diff_sampler) != original_order


def test_stateful_distributed_sampler_numreplicas(valid_path_dataset) -> None:
    fullset = set(range(len(valid_path_dataset)))
    for num_replicas in range(1, 4):
        for batch_size in range(1, 4):
            samplers = [
                StatefulDistributedSampler(
                    dataset=valid_path_dataset,
                    batch_size=batch_size,
                    rank=rank,
                    seed=0,
                    num_replicas=num_replicas,
                )
                for rank in range(num_replicas)
            ]

            # make sure each subset only differs by at most one element in size
            len_samplers = np.array(
                [len(list(sampler)) for sampler in samplers]
            )
            assert ((len_samplers - len_samplers.min()) <= 1).all()

            # make sure each subset is mutually exclusive and union covers the fullset
            concat_idxs = functools.reduce(
                lambda x, y: x + y, [list(sampler) for sampler in samplers]
            )
            assert len(concat_idxs) == len(np.unique(concat_idxs))
            assert set(concat_idxs) == fullset


def test_stateful_distributed_sampler_numreplicas_drop_last(
    valid_path_dataset,
) -> None:
    fullset = set(range(len(valid_path_dataset)))
    for num_replicas in range(1, 4):
        for batch_size in range(1, 4):
            samplers = [
                StatefulDistributedSampler(
                    dataset=valid_path_dataset,
                    batch_size=batch_size,
                    rank=rank,
                    seed=0,
                    num_replicas=num_replicas,
                    drop_last=True,
                )
                for rank in range(num_replicas)
            ]

            # make sure each subset only differs by at most one element in size
            len_samplers = np.array(
                [len(list(sampler)) for sampler in samplers]
            )
            assert ((len_samplers - len_samplers.min()) <= 1).all()

            # make sure each subset is mutually exclusive and union covers the fullset
            concat_idxs = functools.reduce(
                lambda x, y: x + y, [list(sampler) for sampler in samplers]
            )
            assert len(concat_idxs) == len(np.unique(concat_idxs))
            assert (
                len(concat_idxs)
                == (len(fullset) // num_replicas) * num_replicas
            )
