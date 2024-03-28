"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, TypeVar

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.datasets._utils import rename_data_object_keys
from ocpmodels.datasets.target_metadata_guesser import guess_property_metadata
from ocpmodels.modules.transforms import DataTransforms

T_co = TypeVar("T_co", covariant=True)


@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
class LmdbDataset(Dataset[T_co]):
    metadata_path: Path
    sharded: bool

    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    The keys in the LMDB must be integers (stored as ascii objects) starting
    from 0 through the length of the LMDB. For historical reasons any key named
    "length" is ignored since that was used to infer length of many lmdbs in the same
    folder, but lmdb lengths are now calculated directly from the number of keys.
    Args:
            config (dict): Dataset configuration
    """

    def __init__(self, config) -> None:
        super(LmdbDataset, self).__init__()
        self.config = config

        assert not self.config.get(
            "train_on_oc20_total_energies", False
        ), "For training on total energies set dataset=oc22_lmdb"

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries
                    # in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)

            # If "length" encoded as ascii is present, use that
            length_entry = self.env.begin().get("length".encode("ascii"))
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = assert_is_instance(
                    self.env.stat()["entries"], int
                )

            self._keys = list(range(num_entries))
            self.num_samples = num_entries

        # If specified, limit dataset to only a portion of the entire dataset
        # total_shards: defines total chunks to partition dataset
        # shard: defines dataset shard to make visible
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> T_co:
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(
                data_object, self.key_mapping
            )

        data_object = self.transforms(data_object)

        return data_object

    def connect_db(self, lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self) -> None:
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

    def get_metadata(self, num_samples: int = 100):
        # This will interogate the classic OCP LMDB format to determine
        # which properties are present and attempt to guess their shapes
        # and whether they are intensive or extensive.

        # Grab an example data point
        example_pyg_data = self.__getitem__(0)

        # Check for all properties we've used for OCP datasets in the past
        props = []
        for potential_prop in [
            "y",
            "y_relaxed",
            "stress",
            "stresses",
            "force",
            "forces",
        ]:
            if hasattr(example_pyg_data, potential_prop):
                props.append(potential_prop)

        # Get a bunch of random data samples and the number of atoms
        sample_pyg = [
            self[i]
            for i in np.random.choice(
                self.__len__(), size=(num_samples,), replace=False
            )
        ]
        atoms_lens = [data.natoms for data in sample_pyg]

        # Guess the metadata for targets for each found property
        metadata = {
            "targets": {
                prop: guess_property_metadata(
                    atoms_lens, [getattr(data, prop) for data in sample_pyg]
                )
                for prop in props
            }
        }

        return metadata


class SinglePointLmdbDataset(LmdbDataset[BaseData]):
    def __init__(self, config, transform=None) -> None:
        super(SinglePointLmdbDataset, self).__init__(config, transform)
        warnings.warn(
            "SinglePointLmdbDataset is deprecated and will be removed in the future."
            "Please use 'LmdbDataset' instead.",
            stacklevel=3,
        )


class TrajectoryLmdbDataset(LmdbDataset[BaseData]):
    def __init__(self, config, transform=None) -> None:
        super(TrajectoryLmdbDataset, self).__init__(config, transform)
        warnings.warn(
            "TrajectoryLmdbDataset is deprecated and will be removed in the future."
            "Please use 'LmdbDataset' instead.",
            stacklevel=3,
        )


def data_list_collater(
    data_list: List[BaseData], otf_graph: bool = False
) -> BaseData:
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for _, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch
