from __future__ import annotations

import collections.abc
import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fairchem.core._cli import Runner
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, setup_logging

setup_logging()


@pytest.fixture()
def configs():
    return {
        "escn": Path("tests/core/models/test_configs/test_escn.yml"),
        "gemnet": Path("tests/core/models/test_configs/test_gemnet.yml"),
        "equiformer_v2": Path("tests/core/models/test_configs/test_equiformerv2.yml"),
    }


@pytest.fixture()
def tutorial_train_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/train_100"


@pytest.fixture()
def tutorial_val_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/val_20"


def oc20_lmdb_train_and_val_from_paths(train_src, val_src, test_src=None):
    datasets = {}
    if train_src is not None:
        datasets["train"] = {
            "src": train_src,
            "normalize_labels": True,
            "target_mean": -0.7554450631141663,
            "target_std": 2.887317180633545,
            "grad_target_mean": 0.0,
            "grad_target_std": 2.887317180633545,
        }
    if val_src is not None:
        datasets["val"] = {"src": val_src}
    if test_src is not None:
        datasets["test"] = {"src": test_src}
    return datasets


def get_tensorboard_log_files(logdir):
    return glob.glob(f"{logdir}/tensorboard/*/events.out*")


def get_tensorboard_log_values(logdir):
    tf_event_files = get_tensorboard_log_files(logdir)
    assert len(tf_event_files) == 1
    tf_event_file = tf_event_files[0]
    acc = EventAccumulator(tf_event_file)
    acc.Reload()
    return acc


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _run_main(
    rundir,
    input_yaml,
    update_dict_with=None,
    update_run_args_with=None,
    save_checkpoint_to=None,
    save_predictions_to=None,
):
    config_yaml = Path(rundir) / "train_and_val_on_val.yml"

    with open(input_yaml) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    if update_dict_with is not None:
        yaml_config = merge_dictionary(yaml_config, update_dict_with)
    with open(str(config_yaml), "w") as yaml_file:
        yaml.dump(yaml_config, yaml_file)

    run_args = {
        "run_dir": rundir,
        "logdir": f"{rundir}/logs",
        "config_yml": config_yaml,
    }
    if update_run_args_with is not None:
        run_args.update(update_run_args_with)

    # run
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args(
        ["--mode", "train", "--seed", "100", "--config-yml", "config.yml", "--cpu"]
    )
    for arg_name, arg_value in run_args.items():
        setattr(args, arg_name, arg_value)
    config = build_config(args, override_args)
    Runner()(config)

    if save_checkpoint_to is not None:
        checkpoints = glob.glob(f"{rundir}/checkpoints/*/checkpoint.pt")
        assert len(checkpoints) == 1
        os.rename(checkpoints[0], save_checkpoint_to)
    if save_predictions_to is not None:
        predictions_filenames = glob.glob(f"{rundir}/results/*/s2ef_predictions.npz")
        assert len(predictions_filenames) == 1
        os.rename(predictions_filenames[0], save_predictions_to)
    return get_tensorboard_log_values(
        f"{rundir}/logs",
    )


@pytest.fixture(scope="class")
def torch_tempdir(tmpdir_factory):
    return tmpdir_factory.mktemp("torch_tempdir")


"""
These tests are intended to be as quick as possible and test only that the network is runnable and outputs training+validation to tensorboard output
These should catch errors such as shape mismatches or otherways to code wise break a network
"""


class TestSmoke:
    def smoke_test_train(
        self,
        model_name,
        input_yaml,
        tutorial_val_src,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            # first train a very simple model, checkpoint
            train_rundir = Path(tempdirname) / "train"
            train_rundir.mkdir()
            checkpoint_path = str(train_rundir / "checkpoint.pt")
            training_predictions_filename = str(train_rundir / "train_predictions.npz")
            acc = _run_main(
                rundir=str(train_rundir),
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {"max_epochs": 2, "eval_every": 8},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                save_checkpoint_to=checkpoint_path,
                save_predictions_to=training_predictions_filename,
            )
            assert "train/energy_mae" in acc.Tags()["scalars"]
            assert "val/energy_mae" in acc.Tags()["scalars"]

            # second load the checkpoint and predict
            predictions_rundir = Path(tempdirname) / "predict"
            predictions_rundir.mkdir()
            predictions_filename = str(predictions_rundir / "predictions.npz")
            _run_main(
                rundir=str(predictions_rundir),
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {"max_epochs": 2, "eval_every": 8},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={
                    "mode": "predict",
                    "checkpoint": checkpoint_path,
                },
                save_predictions_to=predictions_filename,
            )

            # verify predictions from train and predict are identical
            energy_from_train = np.load(training_predictions_filename)["energy"]
            energy_from_checkpoint = np.load(predictions_filename)["energy"]
            assert np.isclose(energy_from_train, energy_from_checkpoint).all()

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("gemnet", id="gemnet"),
            pytest.param("escn", id="escn"),
            pytest.param("equiformer_v2", id="equiformer_v2"),
        ],
    )
    def test_train_and_predict(
        self,
        model_name,
        configs,
        tutorial_val_src,
    ):
        self.smoke_test_train(
            model_name=model_name,
            input_yaml=configs[model_name],
            tutorial_val_src=tutorial_val_src,
        )

    # train for a few steps and confirm same seeds get same results
    def test_different_seeds(
        self,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)

            seed0_take1_rundir = tempdir / "seed0take1"
            seed0_take1_rundir.mkdir()
            seed0_take1_acc = _run_main(
                rundir=str(seed0_take1_rundir),
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 0},
                input_yaml=configs["escn"],
            )

            seed1000_rundir = tempdir / "seed1000"
            seed1000_rundir.mkdir()
            seed1000_acc = _run_main(
                rundir=str(seed1000_rundir),
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 1000},
                input_yaml=configs["escn"],
            )

            seed0_take2_rundir = tempdir / "seed0_take2"
            seed0_take2_rundir.mkdir()
            seed0_take2_acc = _run_main(
                rundir=str(seed0_take2_rundir),
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 0},
                input_yaml=configs["escn"],
            )

            assert not np.isclose(
                seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
                seed1000_acc.Scalars("train/energy_mae")[-1].value,
            )
            assert np.isclose(
                seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
                seed0_take2_acc.Scalars("train/energy_mae")[-1].value,
            )


"""
These tests intend to test if optimization is not obviously broken on a time scale of a few minutes
"""


class TestSmallDatasetOptim:
    @pytest.mark.parametrize(
        ("model_name", "expected_energy_mae", "expected_force_mae"),
        [
            pytest.param("gemnet", 0.45, 0.06, id="gemnet"),
            pytest.param("escn", 0.4, 0.06, id="escn"),
            pytest.param("equiformer_v2", 0.4, 0.06, id="equiformer_v2"),
        ],
    )
    def test_train_optimization(
        self,
        model_name,
        expected_energy_mae,
        expected_force_mae,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            acc = _run_main(
                rundir=tempdirname,
                input_yaml=configs[model_name],
                update_dict_with={
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                    ),
                },
            )
            assert acc.Scalars("train/energy_mae")[-1].value < expected_energy_mae
            assert acc.Scalars("train/forces_mae")[-1].value < expected_force_mae
