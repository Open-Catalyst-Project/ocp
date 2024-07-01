"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import torch
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file

if TYPE_CHECKING:
    from ase import Atoms


# Following the OCP tutorial: https://github.com/Open-Catalyst-Project/tutorial
@pytest.fixture()
def atoms() -> Atoms:
    atoms = fcc111("Pt", size=(2, 2, 5), vacuum=10.0)
    add_adsorbate(atoms, "O", height=1.2, position="fcc")
    return atoms


@pytest.fixture(
    params=[
        "SchNet-S2EF-OC20-All",
        "DimeNet++-S2EF-OC20-All",
        "GemNet-dT-S2EF-OC20-All",
        "PaiNN-S2EF-OC20-All",
        "GemNet-OC-Large-S2EF-OC20-All+MD",
        "SCN-S2EF-OC20-All+MD",
        # Equiformer v2  # already tested in test_relaxation_final_energy
        # "EquiformerV2-153M-S2EF-OC20-All+MD"
        # eSCNm # already tested in test_random_seed_final_energy
        # "eSCN-L4-M2-Lay12-S2EF-OC20-2M"
    ]
)
def checkpoint_path(request, tmp_path):
    return model_name_to_local_file(request.param, tmp_path)


# First let's just make sure all checkpoints are being loaded without any
# errors as part of the ASE calculator setup.
def test_calculator_setup(checkpoint_path):
    _ = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)


# test relaxation with EqV2
def test_relaxation_final_energy(atoms, tmp_path, snapshot) -> None:
    random.seed(1)
    torch.manual_seed(1)
    calc = OCPCalculator(
        checkpoint_path=model_name_to_local_file(
            "EquiformerV2-153M-S2EF-OC20-All+MD", tmp_path
        ),
        cpu=True,
    )

    assert "energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties

    atoms.set_calculator(calc)
    opt = BFGS(atoms)
    opt.run(fmax=0.05, steps=100)

    assert snapshot == round(atoms.get_potential_energy(), 2)


# test random seed final energy with eSCN
def test_random_seed_final_energy(atoms, tmp_path):
    # too big to run on CircleCI on github
    seeds = [100, 200, 100]
    results_by_seed = {}
    # compute the value for each seed , make sure repeated seeds have the exact same output

    checkpoint_path = model_name_to_local_file(
        "eSCN-L4-M2-Lay12-S2EF-OC20-2M", tmp_path
    )

    for seed in seeds:
        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True, seed=seed)

        atoms.set_calculator(calc)

        energy = atoms.get_potential_energy()
        if seed in results_by_seed:
            assert results_by_seed[seed] == energy
        else:
            results_by_seed[seed] = energy
    # make sure different seeds give slightly different results , expected due to discretization error in grid
    for seed_a in set(seeds):
        for seed_b in set(seeds) - {seed_a}:
            assert results_by_seed[seed_a] != results_by_seed[seed_b]
