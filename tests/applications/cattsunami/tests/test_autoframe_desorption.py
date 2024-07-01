from fairchem.applications.cattsunami.core.autoframe import (
    AutoFrameDesorption,
    interpolate_and_correct_frames,
)
from fairchem.applications.cattsunami.core import Reaction
from fairchem.core import FAIRChemCalculator 
from fairchem.core.models.model_registry import model_name_to_local_file
import numpy as np
import pytest
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH
from fairchem.applications.cattsunami.databases import DESORPTION_REACTION_DB_PATH


@pytest.mark.usefixtures("desorption_inputs")
class TestAutoframe:
    def test_overall_functionality(self, tmp_path):
        inputs = self.inputs
        num_frames = 10
        reactant_systems = inputs["reactant_systems"]
        reactant_energies = inputs["reactant_energies"]

        reaction = Reaction(
            reaction_db_path=DESORPTION_REACTION_DB_PATH,
            reaction_id_from_db=0,
            adsorbate_db_path=ADSORBATE_PKL_PATH,
        )
        checkpoint_path = model_name_to_local_file(
            "EquiformerV2-31M-S2EF-OC20-All+MD",
            local_cache=tmp_path / "ocp_checkpoints",
        )
        calc1 = FAIRChemCalculator (checkpoint_path=checkpoint_path, cpu=False)
        af = AutoFrameDesorption(reaction, reactant_systems, reactant_energies, 3)
        neb_frames_sets = af.get_neb_frames(
            calc1,
            n_frames=num_frames,
            n_systems=5,
            fmax=0.5,
        )
        neb_frames_len = [len(neb_set) == num_frames for neb_set in neb_frames_sets]

        assert all(neb_frames_len)
        assert len(neb_frames_sets) == 5

        neb_frames_sets = af.get_neb_frames(
            calc1,
            n_frames=num_frames,
            n_systems=2,
            fmax=0.5,
        )
        assert len(neb_frames_sets) == 2

    def test_additional_failure_cases(self, tmp_path):
        inputs = self.inputs
        num_frames = 10
        reactant_systems = inputs["reactant_systems"]
        reactant_energies = inputs["reactant_energies"]
        dissociated_adsorbate = reactant_systems[0].copy()
        dissociated_adsorbate.positions[-1] = dissociated_adsorbate.positions[
            -1
        ] + np.array([1, 2, -1.25])
        reactant_systems.append(dissociated_adsorbate)

        dis_des_adsorbate = reactant_systems[0].copy()
        dis_des_adsorbate.positions[-1] = dis_des_adsorbate.positions[-1] + np.array(
            [0, 0, 3]
        )
        dis_des_adsorbate.positions[-2] = dis_des_adsorbate.positions[-2] + np.array(
            [1, 2, 3]
        )
        with pytest.raises(Exception):
            interpolate_and_correct_frames(
                reactant_systems[0], dis_des_adsorbate, 10, reaction, 0
            )
