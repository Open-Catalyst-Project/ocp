"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import io

import pytest
import requests
import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict, setup_imports
from ocpmodels.modules.scaling import ScaleFactor
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted


class TestGemNetOC:
    def test_no_scaling_mismatch(self) -> None:
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
            k[len("module.") * 2 :]: v
            for k, v in checkpoint["state_dict"].items()
        }

        try:
            load_state_dict(model, new_dict)
        except ValueError as e:
            assert False, f"'load_state_dict' raised an exception {e}"

    def test_scaling_mismatch(self) -> None:
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

        for key in checkpoint["scale_dict"]:
            for submodule in model.modules():
                if not isinstance(submodule, ScaleFactor):
                    continue

                submodule.reset_()

            load_scales_compat(model, checkpoint["scale_dict"])

            new_dict = {
                k[len("module.") * 2 :]: v
                for k, v in checkpoint["state_dict"].items()
            }
            param_key = f"{key}.scale_factor"
            new_dict[param_key] = checkpoint["scale_dict"][key] - 10.0

            with pytest.raises(
                ValueError,
                match=f"Scale factor parameter {param_key} is inconsistent with the loaded state dict.",
            ):
                load_state_dict(model, new_dict)

    def test_no_file_exists(self) -> None:
        torch.manual_seed(4)
        setup_imports()

        with pytest.raises(ValueError):
            registry.get_model_class("gemnet_oc")(
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
                scale_file="/tmp/this/file/does/not/exist.pt",
            )

    def test_not_fitted(self) -> None:
        torch.manual_seed(4)
        setup_imports()

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
            scale_file=None,
        )

        with pytest.raises(ValueError):
            ensure_fitted(model)
