"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import io
import os

import pytest
import requests
import torch
from ase.io import read
from torch.nn.parallel.distributed import DistributedDataParallel

from fairchem.core.common.registry import registry
from fairchem.core.common.test_utils import PGConfig, spawn_multi_process
from fairchem.core.common.utils import load_state_dict, setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
)
from fairchem.core.preprocessing import AtomsToGraphs


@pytest.fixture(scope="class")
def load_data(request):
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


def _load_model():
    torch.manual_seed(4)
    setup_imports()

    # download and load weights.
    checkpoint_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt"

    # load buffer into memory as a stream
    # and then load it with torch.load
    r = requests.get(checkpoint_url, stream=True)
    r.raise_for_status()
    checkpoint = torch.load(io.BytesIO(r.content), map_location=torch.device("cpu"))

    model = registry.get_model_class("equiformer_v2")(
        None,
        -1,
        1,
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=12.0,
        max_num_elements=90,
        num_layers=8,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type="layer_norm_sh",
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="silu",
        use_s2_act_attn=False,
        ffn_activation="silu",
        use_gate_act=False,
        use_grid_mlp=True,
        alpha_drop=0.1,
        drop_path_rate=0.1,
        proj_drop=0.0,
        weight_init="uniform",
    )

    new_dict = {k[len("module.") * 2 :]: v for k, v in checkpoint["state_dict"].items()}
    load_state_dict(model, new_dict)

    # Precision errors between mac vs. linux compound with multiple layers,
    # so we explicitly set the number of layers to 1 (instead of all 8).
    # The other alternative is to have different snapshots for mac vs. linux.
    model.num_layers = 1
    return model


@pytest.fixture(scope="class")
def load_model(request):
    request.cls.model = _load_model()


def _runner(data):
    # serializing the model through python multiprocess results in precision errors, so we get a fresh model here
    model = _load_model()
    ddp_model = DistributedDataParallel(model)
    outputs = ddp_model(data_list_collater([data]))
    return {k: v.detach() for k, v in outputs.items()}


@pytest.mark.usefixtures("load_data")
@pytest.mark.usefixtures("load_model")
class TestEquiformerV2:
    def test_energy_force_shape(self, snapshot):
        # Recreate the Data object to only keep the necessary features.
        data = self.data
        model = copy.deepcopy(self.model)

        # Pass it through the model.
        outputs = model(data_list_collater([data]))
        print(outputs)
        energy, forces = outputs["energy"], outputs["forces"]

        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())

        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))

    def test_ddp(self, snapshot):
        data_dist = self.data.clone().detach()
        config = PGConfig(backend="gloo", world_size=1, gp_group_size=1, use_gp=False)
        output = spawn_multi_process(config, _runner, data_dist)
        assert len(output) == 1
        energy, forces = output[0]["energy"], output[0]["forces"]
        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())
        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))

    def test_gp(self, snapshot):
        data_dist = self.data.clone().detach()
        config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
        output = spawn_multi_process(config, _runner, data_dist)
        assert len(output) == 2
        energy, forces = output[0]["energy"], output[0]["forces"]
        assert snapshot == energy.shape
        assert snapshot == pytest.approx(energy.detach())
        assert snapshot == forces.shape
        assert snapshot == pytest.approx(forces.detach().mean(0))


class TestMPrimaryLPrimary:
    def test_mprimary_lprimary_mappings(self):
        def sign(x):
            return 1 if x >= 0 else -1

        device = torch.device("cpu")
        lmax_list = [6, 8]
        mmax_list = [3, 6]
        for lmax in lmax_list:
            for mmax in mmax_list:
                c = CoefficientMappingModule([lmax], [mmax])

                embedding = SO3_Embedding(
                    length=1,
                    lmax_list=[lmax],
                    num_channels=1,
                    device=device,
                    dtype=torch.float32,
                )

                """
                Generate L_primary matrix
                L0: 0.00 ~ L0M0
                L1: -1.01 1.00 1.01 ~ L1M(-1),L1M0,L1M1
                L2: -2.02 -2.01 2.00 2.01 2.02 ~ L2M(-2),L2M(-1),L2M0,L2M1,L2M2
                """
                test_matrix_lp = []
                for l in range(lmax + 1):
                    max_m = min(l, mmax)
                    for m in range(-max_m, max_m + 1):
                        v = l * sign(m) + 0.01 * m  # +/- l . 00 m
                        test_matrix_lp.append(v)

                test_matrix_lp = (
                    torch.tensor(test_matrix_lp).reshape(1, -1, 1).to(torch.float32)
                )

                """
                Generate M_primary matrix
                M0: 0.00 , 1.00, 2.00, ... , LMax ~ M0L0, M0L1, .., M0L(LMax)
                M1: 1.01, 2.01, .., LMax.01, -1.01, -2.01, -LMax.01 ~ L1M1, L2M1, .., L(LMax)M1, L1M(-1), L2M(-1), ... , L(LMax)M(-1)
                """
                test_matrix_mp = []
                for m in range(max_m + 1):
                    for l in range(m, lmax + 1):
                        v = l + 0.01 * m  # +/- l . 00 m
                        test_matrix_mp.append(v)
                    if m > 0:
                        for l in range(m, lmax + 1):
                            v = -(l + 0.01 * m)  # +/- l . 00 m
                            test_matrix_mp.append(v)

                test_matrix_mp = (
                    torch.tensor(test_matrix_mp).reshape(1, -1, 1).to(torch.float32)
                )

                embedding.embedding = test_matrix_lp.clone()

                embedding._m_primary(c)
                mp = embedding.embedding.clone()
                (test_matrix_mp == mp).all()

                embedding._l_primary(c)
                lp = embedding.embedding.clone()
                (test_matrix_lp == lp).all()

