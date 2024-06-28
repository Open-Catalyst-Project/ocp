"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest
import torch
from ase.io import read

from fairchem.core.common.registry import registry
from fairchem.core.common.transforms import RandomRotate
from fairchem.core.common.utils import setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def load_data(request) -> None:
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


@pytest.fixture(scope="class")
def load_model(request) -> None:
    torch.manual_seed(4)
    setup_imports()

    model = registry.get_model_class("dimenetplusplus")(
        output_targets={"energy": {}, "forces": {}},
        cutoff=6.0,
        regress_forces=True,
        use_pbc=False,
    )
    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestDimeNet:
    def test_rotation_invariance(self) -> None:
        random.seed(1)
        data = self.data

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_rotated, rot, inv_rot = transform(data.clone())
        assert not np.array_equal(data.pos, data_rotated.pos)

        # Pass it through the model.
        batch = data_list_collater([data, data_rotated])
        out = self.model(batch)

        # Compare predicted energies and forces (after inv-rotation).
        energies = out["energy"].detach()
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=5)

        forces = out["forces"].detach()
        logging.info(forces)
        np.testing.assert_array_almost_equal(
            forces[: forces.shape[0] // 2],
            torch.matmul(forces[forces.shape[0] // 2 :], inv_rot),
            decimal=5,
        )

    def test_energy_force_shape(self, snapshot) -> None:
        # Recreate the Data object to only keep the necessary features.
        data = self.data

        # Pass it through the model.
        outputs = self.model(data_list_collater([data]))
        energy, forces = outputs["energy"], outputs["forces"]

        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())

        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach())
