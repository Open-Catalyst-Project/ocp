"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import pytest
from ase.io import read
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from ocpmodels.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def atoms_to_graphs_internals(request) -> None:
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    atoms.info["stiffness_tensor"] = np.array(
        [
            [293, 121, 121, 0, 0, 0],
            [121, 293, 121, 0, 0, 0],
            [121, 121, 293, 0, 0, 0],
            [0, 0, 0, 146, 0, 0],
            [0, 0, 0, 0, 146, 0],
            [0, 0, 0, 0, 0, 146],
        ],
        dtype=float,
    )
    test_object = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_distances=True,
        r_data_keys=["stiffness_tensor"],
    )
    request.cls.atg = test_object
    request.cls.atoms = atoms


@pytest.mark.usefixtures("atoms_to_graphs_internals")
class TestAtomsToGraphs:
    def test_gen_neighbors_pymatgen(self) -> None:
        # call the internal function
        (
            c_index,
            n_index,
            n_distances,
            offsets,
        ) = self.atg._get_neighbors_pymatgen(self.atoms)
        edge_index, edge_distances, cell_offsets = self.atg._reshape_features(
            c_index, n_index, n_distances, offsets
        )

        # use ase to compare distances and indices
        n = NeighborList(
            cutoffs=[self.atg.radius / 2.0] * len(self.atoms),
            self_interaction=False,
            skin=0,
            bothways=True,
            primitive=NewPrimitiveNeighborList,
        )
        n.update(self.atoms)
        ase_neighbors = [
            n.get_neighbors(index) for index in range(len(self.atoms))
        ]
        ase_s_index = []
        ase_n_index = []
        ase_offsets = []
        for i, n in enumerate(ase_neighbors):
            nidx = n[0]
            ncount = len(nidx)
            ase_s_index += [i] * ncount
            ase_n_index += nidx.tolist()
            ase_offsets.append(n[1])
        ase_s_index = np.array(ase_s_index)
        ase_n_index = np.array(ase_n_index)
        ase_offsets = np.concatenate(ase_offsets)
        # compute ase distance
        cell = self.atoms.cell
        positions = self.atoms.positions
        distance_vec = positions[ase_s_index] - positions[ase_n_index]
        _offsets = np.dot(ase_offsets, cell)
        distance_vec -= _offsets
        act_dist = np.linalg.norm(distance_vec, axis=-1)

        act_dist = np.sort(act_dist)
        act_index = np.sort(ase_n_index)
        test_dist = np.sort(edge_distances)
        test_index = np.sort(edge_index[0, :])
        # check that the distance and neighbor index values are correct
        np.testing.assert_allclose(act_dist, test_dist)
        np.testing.assert_array_equal(act_index, test_index)

    def test_convert(self) -> None:
        # run convert on a single atoms obj
        data = self.atg.convert(self.atoms)
        # atomic numbers
        act_atomic_numbers = self.atoms.get_atomic_numbers()
        atomic_numbers = data.atomic_numbers.numpy()
        np.testing.assert_equal(act_atomic_numbers, atomic_numbers)
        # positions
        act_positions = self.atoms.get_positions()
        positions = data.pos.numpy()
        np.testing.assert_allclose(act_positions, positions)
        # check energy value
        act_energy = self.atoms.get_potential_energy(apply_constraint=False)
        test_energy = data.energy
        np.testing.assert_equal(act_energy, test_energy)
        # forces
        act_forces = self.atoms.get_forces(apply_constraint=False)
        forces = data.forces.numpy()
        np.testing.assert_allclose(act_forces, forces)
        # stress
        act_stress = self.atoms.get_stress(apply_constraint=False, voigt=False)
        stress = data.stress.numpy()
        np.testing.assert_allclose(act_stress, stress)
        # additional data (ie stiffness_tensor)
        stiffness_tensor = data.stiffness_tensor.numpy()
        np.testing.assert_allclose(
            self.atoms.info["stiffness_tensor"], stiffness_tensor
        )

    def test_convert_all(self) -> None:
        # run convert_all on a list with one atoms object
        # this does not test the atoms.db functionality
        atoms_list = [self.atoms]
        data_list = self.atg.convert_all(atoms_list)
        # check shape/values of features
        # atomic numbers
        act_atomic_nubmers = self.atoms.get_atomic_numbers()
        atomic_numbers = data_list[0].atomic_numbers.numpy()
        np.testing.assert_equal(act_atomic_nubmers, atomic_numbers)
        # positions
        act_positions = self.atoms.get_positions()
        positions = data_list[0].pos.numpy()
        np.testing.assert_allclose(act_positions, positions)
        # check energy value
        act_energy = self.atoms.get_potential_energy(apply_constraint=False)
        test_energy = data_list[0].energy
        np.testing.assert_equal(act_energy, test_energy)
        # forces
        act_forces = self.atoms.get_forces(apply_constraint=False)
        forces = data_list[0].forces.numpy()
        np.testing.assert_allclose(act_forces, forces)
        # stress
        act_stress = self.atoms.get_stress(apply_constraint=False, voigt=False)
        stress = data_list[0].stress.numpy()
        np.testing.assert_allclose(act_stress, stress)
        # additional data (ie stiffness_tensor)
        stiffness_tensor = data_list[0].stiffness_tensor.numpy()
        np.testing.assert_allclose(
            self.atoms.info["stiffness_tensor"], stiffness_tensor
        )
