import pytest

import numpy as np
import numpy.testing as npt

from ase import build
from ase.optimize import LBFGS, FIRE, BFGS

try:
    from ase.filters import UnitCellFilter
except ModuleNotFoundError:
    # older ase version, import UnitCellFilterOld
    from ase.constraints import UnitCellFilter

from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing.atoms_to_graphs import AtomsToGraphs
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation.optimizers import LBFGS as LBFGS_torch
from fairchem.core.common.relaxation import OptimizableBatch, OptimizableUnitCellBatch


@pytest.fixture
def calculator(tmpdir):
    calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=tmpdir, seed=0
    )
    # TODO debug this
    # removing amp so that we always get float32 predictions
    calc.trainer.scaler = None
    return calc


@pytest.fixture(scope="function")
def atoms_list():
    atoms_list = [
        build.bulk("Cu", "fcc", a=3.8, cubic=True),
        build.bulk("NaCl", crystalstructure="rocksalt", a=5.8),
    ]
    for atoms in atoms_list:
        atoms.rattle(stdev=0.05, seed=0)
    return atoms_list


@pytest.fixture(scope="function")
def batch(atoms_list):
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    return data_list_collater([a2g.convert(atoms) for atoms in atoms_list])


@pytest.fixture(params=[FIRE, BFGS])
def optimizer_cls(request):
    return request.param


def test_lbfgs_relaxation(atoms_list, batch, calculator):
    """Tests batch relaxation using ASE optimizers."""
    obatch = OptimizableBatch(batch, trainer=calculator.trainer, numpy=False)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = LBFGS(atoms, damping=0.8, alpha=70.0)
        opt.run(0.01, 20)

    # optimize atoms in batch using ASE
    batch_optimizer = LBFGS_torch(obatch, damping=0.8, alpha=70.0)
    batch_optimizer.run(0.01, 20)

    # compare energy and atom positions, this needs pretty slack tols but that should be ok
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )
        pnorm1 = np.linalg.norm(a1.positions, axis=1)
        pnorm2 = np.linalg.norm(a2.positions, axis=1)
        npt.assert_allclose(pnorm1, pnorm2, atol=0.01)
        npt.assert_allclose(a1.positions, a2.positions, rtol=0.01, atol=0.05)


def test_ase_relaxation(atoms_list, batch, calculator, optimizer_cls):
    """Tests batch relaxation using ASE optimizers."""
    obatch = OptimizableBatch(batch, trainer=calculator.trainer, numpy=True)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = optimizer_cls(atoms)
        opt.run(0.01, 20)

    # TODO relaxations with LBFGS give nans for positions which leads to a cryptic
    #  error in edge_rot_mat (since edges are empty)
    # optimize atoms in batch using ASE
    batch_optimizer = optimizer_cls(obatch)
    batch_optimizer.run(0.01, 20)

    # compare energy and atom positions, this needs pretty slack tols but that should be ok
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )
        pnorm1 = np.linalg.norm(a1.positions, axis=1)
        pnorm2 = np.linalg.norm(a2.positions, axis=1)
        npt.assert_allclose(pnorm1, pnorm2, atol=0.01)
        npt.assert_allclose(a1.positions, a2.positions, rtol=0.01, atol=0.05)


@pytest.mark.parametrize("mask_converged", [False, True])
def test_batch_relaxation_mask(atoms_list, calculator, mask_converged):
    """Test that masking is working as intended!"""
    # relax only the first atom in list
    atoms = atoms_list[0]
    atoms.calc = calculator
    opt = LBFGS(atoms)
    opt.run(0.01, 50)
    assert ((atoms.get_forces() ** 2).sum(axis=1) ** 0.5 <= 0.01).all()

    # now create a batch
    batch = data_list_collater([calculator.a2g.convert(atoms) for atoms in atoms_list])
    obatch = OptimizableBatch(
        batch, trainer=calculator.trainer, numpy=True, mask_converged=mask_converged
    )

    npt.assert_allclose(batch.pos[batch.batch == 0].cpu().numpy(), atoms.positions)
    batch_opt = LBFGS(obatch)
    batch_opt.run(0.01, 20)

    if mask_converged:
        # assert preconverged structure was not changed at all
        npt.assert_allclose(batch.pos[batch.batch == 0].cpu().numpy(), atoms.positions)
        assert not np.allclose(
            batch.pos[batch.batch == 1].cpu().numpy(), atoms_list[1].positions
        )
    else:
        # assert that it was changed
        assert not np.allclose(
            batch.pos[batch.batch == 0].cpu().numpy(), atoms.positions
        )


@pytest.mark.skip("Skip until we have a test model that can predict stress")
def test_ase_cell_relaxation(atoms_list, batch, calculator, optimizer_cls):
    """Tests batch relaxation using ASE optimizers."""
    cell_factor = batch.natoms.cpu().numpy().mean()
    obatch = OptimizableUnitCellBatch(
        batch, trainer=calculator.trainer, numpy=True, cell_factor=cell_factor
    )

    # optimize atoms in batch using ASE
    batch_optimizer = optimizer_cls(obatch)
    batch_optimizer.run(0.01, 20)

    # optimize atoms one-by-one
    for atoms in atoms_list:
        atoms.calc = calculator
        opt = optimizer_cls(UnitCellFilter(atoms, cell_factor=cell_factor))
        opt.run(0.01, 20)

    # compare energy, atom positions and cell
    for a1, a2 in zip(atoms_list, obatch.get_atoms_list()):
        assert a1.get_potential_energy() / len(a1) == pytest.approx(
            a2.get_potential_energy() / len(a2), abs=0.05
        )

        pnorm1 = np.linalg.norm(a1.positions, axis=1)
        pnorm2 = np.linalg.norm(a2.positions, axis=1)
        npt.assert_allclose(pnorm1, pnorm2, atol=0.01)
        npt.assert_allclose(a1.positions, a2.positions, rtol=0.01, atol=0.05)

        cnorm1 = np.linalg.norm(a1.cell.array, axis=1)
        cnorm2 = np.linalg.norm(a2.cell.array, axis=1)
        npt.assert_allclose(cnorm1, cnorm2, atol=0.01)
        npt.assert_allclose(
            a1.cell.array.T / cnorm1, a2.cell.array.T / cnorm2, rtol=0.01
        )
