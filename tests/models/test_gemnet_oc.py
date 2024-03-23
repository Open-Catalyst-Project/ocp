"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import io
import logging
import os
import random

import numpy as np
import pytest
import requests
import torch
from ase.io import read

from ocpmodels.common.registry import registry
from ocpmodels.common.transforms import RandomRotate
from ocpmodels.common.utils import load_state_dict, setup_imports
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


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

    # download and load weights.
    checkpoint_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_07/s2ef/gemnet_oc_base_s2ef_all.pt"

    # load buffer into memory as a stream
    # and then load it with torch.load
    r = requests.get(checkpoint_url, stream=True)
    r.raise_for_status()
    checkpoint = torch.load(
        io.BytesIO(r.content), map_location=torch.device("cpu")
    )

    model = registry.get_model_class("gemnet_oc")(
        output_targets={"energy": {}, "forces": {}},
        num_spherical=7,
        num_radial=128,
        num_blocks=4,
        emb_size_atom=256,
        emb_size_edge=512,
        emb_size_trip_in=64,
        emb_size_trip_out=64,
        emb_size_quad_in=32,
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=16,
        emb_size_cbf=16,
        emb_size_sbf=32,
        num_before_skip=2,
        num_after_skip=2,
        num_concat=1,
        num_atom=3,
        num_output_afteratom=3,
        num_atom_emb_layers=2,
        num_global_out_layers=2,
        regress_forces=True,
        direct_forces=True,
        use_pbc=True,
        cutoff=12.0,
        cutoff_qint=12.0,
        cutoff_aeaint=12.0,
        cutoff_aint=12.0,
        max_neighbors=30,
        max_neighbors_qint=8,
        max_neighbors_aeaint=20,
        max_neighbors_aint=1000,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "legendre_outer"},
        extensive=True,
        forces_coupled=False,
        output_init="HeOrthogonal",
        activation="silu",
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
        qint_tags=[1, 2],
        scale_file=checkpoint["scale_dict"],
    )

    new_dict = {
        k[len("module.") * 2 :]: v for k, v in checkpoint["state_dict"].items()
    }
    load_state_dict(model, new_dict)

    request.cls.model = model


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestGemNetOC:
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
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=3)

        forces = out["forces"].detach()
        logging.info(forces)
        np.testing.assert_array_almost_equal(
            forces[: forces.shape[0] // 2],
            torch.matmul(forces[forces.shape[0] // 2 :], inv_rot),
            decimal=3,
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
