from itertools import product
from random import choice
import pytest
import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure

from fairchem.core.datasets import LMDBDatabase, AseDBDataset


@pytest.fixture(scope="session")
def dummy_element_refs():
    # create some dummy elemental energies from ionic radii (ignore deuterium and tritium included in pmg)
    return np.concatenate(
        [[0], [e.average_ionic_radius for e in Element if e.name not in ("D", "T")]]
    )


@pytest.fixture(scope="session")
def max_num_elements(dummy_element_refs):
    return len(dummy_element_refs) - 1


@pytest.fixture(scope="session")
def dataset(tmpdir_factory, dummy_element_refs):
    # a dummy dataset with binaries with energy that depends on composition only plus noise
    all_binaries = list(product(list(Element), repeat=2))
    rng = np.random.default_rng(seed=0)

    tmpdir = tmpdir_factory.mktemp("dataset")
    with LMDBDatabase(tmpdir / "dummy.aselmdb") as db:
        for _ in range(1000):
            elements = choice(all_binaries)
            structure = Structure.from_prototype("cscl", species=elements, a=2.0)
            energy = (
                sum(e.average_ionic_radius for e in elements)
                + 0.05 * rng.random() * dummy_element_refs.mean()
            )
            atoms = structure.to_ase_atoms()
            db.write(atoms, data={"energy": energy, "forces": rng.random((2, 3))})

    dataset = AseDBDataset(
        config={
            "src": str(tmpdir / "dummy.aselmdb"),
            "a2g_args": {"r_data_keys": ["energy", "forces"]},
        }
    )
    return dataset
