"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_config, save_checkpoint
from fairchem.core.modules.element_references import create_element_references
from fairchem.core.modules.normalizer import fit_normalizers


def fit_norms(
    config: dict,
    output_path: str | Path,
    linref_file: str | Path | None = None,
    linref_target: str = "energy",
) -> None:
    """Fit dataset mean and std using the standard config

    Args:
        config: config
        output_path: output path
        linref_file: path to fitted linear references. IF these are used in training they must be used to compute mean/std
        linref_target: target using linear references, basically always energy.
    """
    output_path = Path(output_path).resolve()
    elementrefs = (
        {linref_target: create_element_references(linref_file)}
        if linref_file is not None
        else {}
    )

    try:
        # load the training dataset
        train_dataset = registry.get_dataset_class(
            config["dataset"]["train"].get("format", "lmdb")
        )(config["dataset"]["train"])
    except KeyError as err:
        raise ValueError("Train dataset is not specified in config!") from err

    try:
        norm_config = config["dataset"]["train"]["transforms"]["normalizer"]["fit"]
    except KeyError as err:
        raise ValueError(
            "The provided config does not specify a 'fit' block for 'normalizer'!"
        ) from err

    normalizers = fit_normalizers(
        targets=norm_config["targets"],
        element_references=elementrefs,
        dataset=train_dataset,
        batch_size=norm_config.get("batch_size", 32),
        num_batches=norm_config.get("num_batches"),
        num_workers=config["optim"].get("num_workers", 16),
    )
    path = save_checkpoint(
        normalizers,
        output_path,
        "normalizers.pt",
    )
    logging.info(f"normalizers have been saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configuration yaml file",
    )
    parser.add_argument(
        "--out-path",
        default=".",
        type=str,
        help="Output path to save normalizers",
    )
    parser.add_argument(
        "--linref-path",
        type=str,
        help="Path to linear references used.",
    )
    parser.add_argument(
        "--linref-target",
        default="energy",
        type=str,
        help="target for which linear references are used.",
    )
    args = parser.parse_args()
    config, dup_warning, dup_error = load_config(args.config)

    if len(dup_warning) > 0:
        logging.warning(
            f"The following keys in the given config have duplicates: {dup_warning}."
        )
    if len(dup_error) > 0:
        raise RuntimeError(
            f"The following include entries in the config have duplicates: {dup_error}"
        )

    fit_norms(config, args.out_path, args.linref_path)
