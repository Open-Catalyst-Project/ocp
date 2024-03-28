import numpy as np
import pytest
from ase import build
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.db.row import AtomsRow

from ocpmodels.datasets.lmdb_database import LMDBDatabase

N_WRITES = 100
N_READS = 200


test_structures = [
    build.molecule("H2O", vacuum=4),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
]

test_structures[2].set_constraint(FixAtoms(indices=[0, 1]))


def generate_random_structure():
    # Make base slab
    slab = build.fcc111("Cu", size=(4, 4, 3), vacuum=10.0)

    # Randomly set some elements
    slab.set_chemical_symbols(
        np.random.choice(["Cu", "Ag", "Au", "Pd"], size=(len(slab)))
    )

    # Randomly set some positions
    slab.positions = np.random.normal(size=slab.positions.shape)

    # Add entries for energy/forces/stress/magmom/etc.
    # Property must be one of the ASE core properties to
    # go in to a singlepointcalculator and get stored as
    # fields correctly
    spc = SinglePointCalculator(
        slab,
        energy=np.random.normal(),
        forces=np.random.normal(size=slab.positions.shape),
        stress=np.random.normal(size=(3, 3)),
        magmom=np.random.normal(size=(len(slab))),
    )
    slab.set_calculator(spc)

    # Make up some other properties to show how we can include arbitrary outputs
    slab.info["test_info_property_1"] = np.random.normal(size=(3, 3))
    slab.info["test_info_property_2"] = np.random.normal(size=(len(slab), 3))

    return slab


@pytest.fixture(scope="function")
def ase_lmbd_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dataset")
    with LMDBDatabase(tmp_path / "ase_lmdb.lmdb") as db:
        for structure in test_structures:
            db.write(structure)

        for _ in range(N_WRITES):
            slab = generate_random_structure()
            # Save the slab info, and make sure the info gets put in as data
            db.write(slab, data=slab.info)
    return tmp_path / "ase_lmdb.lmdb"


def test_aselmdb_write(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path, readonly=True) as db:
        for i, structure in enumerate(test_structures):
            assert str(structure) == str(db._get_row_by_index(i).toatoms())


def test_aselmdb_count(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path, readonly=True) as db:
        assert db.count() == N_WRITES + len(test_structures)


def test_aselmdb_delete(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path) as db:
        for i in range(5):
            # Note the available ids list is updating
            # but the ids themselves are fixed.
            db.delete([db.ids[0]])
    assert db.count() == N_WRITES + len(test_structures) - 5


def test_aselmdb_randomreads(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path, readonly=True) as db:
        for _ in range(N_READS):
            total_size = db.count()
            assert isinstance(
                db._get_row_by_index(np.random.choice(total_size)), AtomsRow
            )


def test_aselmdb_constraintread(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path, readonly=True) as db:
        atoms = db._get_row_by_index(2).toatoms()

    assert isinstance(atoms.constraints[0], FixAtoms)


def test_update_keyvalue_pair(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path) as db:
        db.update(1, test=5)

    with LMDBDatabase(ase_lmbd_path) as db:
        row = db._get_row(1)
        assert row.test == 5


def test_update_atoms(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path) as db:
        db.update(40, atoms=test_structures[-1])

    with LMDBDatabase(ase_lmbd_path) as db:
        row = db._get_row(40)
        assert str(row.toatoms()) == str(test_structures[-1])


def test_metadata(ase_lmbd_path) -> None:
    with LMDBDatabase(ase_lmbd_path) as db:
        db.metadata = {"test": True}

    with LMDBDatabase(ase_lmbd_path, readonly=True) as db:
        assert db.metadata["test"] is True
