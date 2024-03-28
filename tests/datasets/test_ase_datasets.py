import os

import numpy as np
import pytest
from ase import build, db
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, write

from ocpmodels.datasets import (
    AseDBDataset,
    AseReadDataset,
    AseReadMultiStructureDataset,
)
from ocpmodels.datasets.lmdb_database import LMDBDatabase

structures = [
    build.molecule("H2O", vacuum=4),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
]
for atoms in structures:
    calc = SinglePointCalculator(
        atoms,
        energy=1,
        forces=atoms.positions,
        # there is an issue with ASE db when writing a db with 3x3 stress it is flattened to (9,) and then
        # errors when trying to read it
        stress=np.random.random((6,)),
    )
    atoms.calc = calc
    atoms.info["extensive_property"] = 3 * len(atoms)
    atoms.info["tensor_property"] = np.random.random((6, 6))

structures[2].set_pbc(True)


@pytest.fixture(
    scope="function",
    params=[
        "db_dataset",
        "db_dataset_folder",
        "db_dataset_list",
        "db_dataset_path_list",
        "lmdb_dataset",
        "aselmdb_dataset",
    ],
)
def ase_dataset(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dataset")
    mult = 1
    a2g_args = {
        "r_energy": True,
        "r_forces": True,
        "r_stress": True,
        "r_data_keys": ["extensive_property", "tensor_property"],
    }
    if request.param == "db_dataset":
        with db.connect(tmp_path / "asedb.db") as database:
            for i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)
        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.db"), "a2g_args": a2g_args}
        )
    elif (
        request.param == "db_dataset_folder"
        or request.param == "db_dataset_list"
    ):
        for db_name in ("asedb1.db", "asedb2.db"):
            with db.connect(tmp_path / db_name) as database:
                for i, atoms in enumerate(structures):
                    database.write(atoms, data=atoms.info)
        mult = 2
        src = (
            str(tmp_path)
            if request.param == "db_dataset_folder"
            else [str(tmp_path / "asedb1.db"), str(tmp_path / "asedb2.db")]
        )
        dataset = AseDBDataset(config={"src": src, "a2g_args": a2g_args})
    elif request.param == "db_dataset_path_list":
        os.mkdir(tmp_path / "dir1")
        os.mkdir(tmp_path / "dir2")

        for dir_name in ("dir1", "dir2"):
            for db_name in ("asedb1.db", "asedb2.db"):
                with db.connect(tmp_path / dir_name / db_name) as database:
                    for i, atoms in enumerate(structures):
                        database.write(atoms, data=atoms.info)
        mult = 4
        dataset = AseDBDataset(
            config={
                "src": [str(tmp_path / "dir1"), str(tmp_path / "dir2")],
                "a2g_args": a2g_args,
            }
        )
    elif request.param == "lmbd_dataset":
        with LMDBDatabase(str(tmp_path / "asedb.lmdb")) as database:
            for i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)

        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.lmdb"), "a2g_args": a2g_args}
        )
    else:  # "aselmbd_dataset" with .aselmdb file extension
        with LMDBDatabase(str(tmp_path / "asedb.lmdb")) as database:
            for i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)

        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.lmdb"), "a2g_args": a2g_args}
        )

    return dataset, mult


def test_ase_dataset(ase_dataset):
    dataset, mult = ase_dataset
    assert len(dataset) == mult * len(structures)
    for data in dataset:
        assert hasattr(data, "y")
        assert data.forces.shape == (data.natoms, 3)
        assert data.stress.shape == (3, 3)
        assert data.tensor_property.shape == (6, 6)
        assert isinstance(data.extensive_property, int)


def test_ase_read_dataset(tmp_path) -> None:
    # unfortunately there is currently no clean (already implemented) way to save atoms.info when saving
    # individual structures - so test separately
    for i, structure in enumerate(structures):
        write(tmp_path / f"{i}.cif", structure)

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.cif",
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data
    dataset.close_db()


def test_ase_metadata_guesser(ase_dataset) -> None:
    dataset, _ = ase_dataset

    metadata = dataset.get_metadata()

    # Confirm energy metadata guessed properly
    assert metadata["targets"]["energy"]["extensive"] is False
    assert metadata["targets"]["energy"]["shape"] == ()
    assert metadata["targets"]["energy"]["type"] == "per-image"

    # Confirm forces metadata guessed properly
    assert metadata["targets"]["forces"]["shape"] == (3,)
    assert metadata["targets"]["forces"]["extensive"] is True
    assert metadata["targets"]["forces"]["type"] == "per-atom"

    # Confirm stress metadata guessed properly
    assert metadata["targets"]["stress"]["shape"] == (3, 3)
    assert metadata["targets"]["stress"]["extensive"] is False
    assert metadata["targets"]["stress"]["type"] == "per-image"

    # Confirm extensive_property metadata guessed properly
    assert metadata["targets"]["info.extensive_property"]["extensive"] is True
    assert metadata["targets"]["info.extensive_property"]["shape"] == ()
    assert (
        metadata["targets"]["info.extensive_property"]["type"] == "per-image"
    )

    # Confirm tensor_property metadata guessed properly
    assert metadata["targets"]["info.tensor_property"]["extensive"] is False
    assert metadata["targets"]["info.tensor_property"]["shape"] == (6, 6)
    assert metadata["targets"]["info.tensor_property"]["type"] == "per-image"


def test_db_add_delete(tmp_path) -> None:
    database = db.connect(tmp_path / "asedb.db")
    for i, atoms in enumerate(structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == len(structures)
    orig_len = len(dataset)

    database.delete([1])

    new_structures = [
        build.molecule("CH3COOH", vacuum=4),
        build.bulk("Al"),
    ]

    for i, atoms in enumerate(new_structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == orig_len + len(new_structures) - 1
    dataset.close_db()


def test_ase_multiread_dataset(tmp_path) -> None:
    atoms_objects = [build.bulk("Cu", a=a) for a in np.linspace(3.5, 3.7, 10)]

    energies = np.linspace(1, 0, len(atoms_objects))

    traj = Trajectory(tmp_path / "test.traj", mode="w")

    for atoms, energy in zip(atoms_objects, energies):
        calc = SinglePointCalculator(
            atoms, energy=energy, forces=atoms.positions
        )
        atoms.calc = calc
        traj.write(atoms)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "keep_in_memory": True,
            "atoms_transform_args": {
                "skip_always": True,
            },
        }
    )

    assert len(dataset) == len(atoms_objects)

    with open(tmp_path / "test_index_file", "w") as f:
        f.write(f"{tmp_path / 'test.traj'} {len(atoms_objects)}")

    dataset = AseReadMultiStructureDataset(
        config={"index_file": str(tmp_path / "test_index_file")},
    )

    assert len(dataset) == len(atoms_objects)

    dataset = AseReadMultiStructureDataset(
        config={
            "index_file": str(tmp_path / "test_index_file"),
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert len(dataset) == len(atoms_objects)

    assert hasattr(dataset[0], "y_relaxed")
    assert dataset[0].y_relaxed != dataset[0].energy
    assert dataset[-1].y_relaxed == dataset[-1].energy

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "ase_read_args": {
                "index": "0",
            },
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert hasattr(dataset[0], "y_relaxed")
    assert dataset[0].y_relaxed != dataset[0].energy


def test_empty_dataset(tmp_path):
    # raises error on empty dataset
    with pytest.raises(ValueError):
        AseReadMultiStructureDataset(config={"src": str(tmp_path)})

    with pytest.raises(ValueError):
        AseDBDataset(config={"src": str(tmp_path)})
