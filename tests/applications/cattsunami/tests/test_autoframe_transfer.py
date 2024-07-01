import pytest
from fairchem.applications.cattsunami.core.autoframe import (
    AutoFrameTransfer,
    interpolate_and_correct_frames,
)
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.applications.cattsunami.core.reaction import Reaction
from fairchem.core import FAIRChemCalculator 
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH
from fairchem.applications.cattsunami.databases import TRANSFER_REACTION_DB_PATH


def get_ads_syms(adslab):
    adsorbate = adslab[[idx for idx, tag in enumerate(adslab.get_tags()) if tag == 2]]
    syms = adsorbate.get_chemical_symbols()
    syms_str = ""
    for sym in syms:
        syms_str += sym
    return syms_str


@pytest.mark.usefixtures("transfer_inputs")
class TestAutoframe:
    def test_overall_functionality(self):
        inputs = self.inputs

        reaction = Reaction(
            reaction_db_path=TRANSFER_REACTION_DB_PATH,
            reaction_id_from_db=2,
            adsorbate_db_path=ADSORBATE_PKL_PATH,
        )

        num_frames = 10
        reactant1_systems = inputs["reactant1_systems"]
        reactant2_systems = inputs["reactant2_systems"]
        product1_systems = inputs["product1_systems"]
        product2_systems = inputs["product2_systems"]

        reactant1_energies = inputs["reactant1_energies"]
        reactant2_energies = inputs["reactant2_energies"]
        product1_energies = inputs["product1_energies"]
        product2_energies = inputs["product2_energies"]

        checkpoint_path = model_name_to_local_file(
            "EquiformerV2-31M-S2EF-OC20-All+MD", local_cache="/tmp/ocp_checkpoints/"
        )
        calc1 = FAIRChemCalculator (checkpoint_path=checkpoint_path, cpu=True)
        af = AutoFrameTransfer(
            reaction,
            reactant1_systems,
            reactant2_systems,
            reactant1_energies,
            reactant2_energies,
            product1_systems,
            product1_energies,
            product2_systems,
            product2_energies,
            5,
            4,
            1.5,
        )
        neb_frames_sets, map_idx_list = af.get_neb_frames(
            calc1,
            n_frames=num_frames,
            n_initial_frames=3,
            n_final_frames_per_initial=3,
            fmax=0.5,
        )
        neb_frames_len = [len(neb_set) == num_frames for neb_set in neb_frames_sets]

        syms_str_agree = [
            get_ads_syms(frame_set[0]) == get_ads_syms(frame_set[-1])
            for frame_set in neb_frames_sets
        ]

        # Test interpolate and correct frames
        new_frames = interpolate_and_correct_frames(
            neb_frames_sets[0][0],
            neb_frames_sets[0][-1],
            5,
            reaction,
            map_idx_list[0],
        )

        assert len(new_frames) == 5
        assert all(syms_str_agree)
        assert all(neb_frames_len)
        assert len(neb_frames_sets) == 9
