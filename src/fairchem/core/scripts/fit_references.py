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
from fairchem.core.modules.element_references import fit_linear_references


def fit_linref(config: dict, output_path: str | Path) -> None:
    """Fit linear references using the standard config

    Args:
        config: config
        output_path: output path
    """
    # load the training dataset
    output_path = Path(output_path).resolve()

    try:
        # load the training dataset
        train_dataset = registry.get_dataset_class(
            config["dataset"]["train"].get("format", "lmdb")
        )(config["dataset"]["train"])
    except KeyError as err:
        raise ValueError("Train dataset is not specified in config!") from err

    try:
        elementref_config = config["dataset"]["train"]["transforms"][
            "element_references"
        ]["fit"]
    except KeyError as err:
        raise ValueError(
            "The provided config does not specify a 'fit' block for 'element_refereces'!"
        ) from err

    element_refs = fit_linear_references(
        targets=elementref_config["targets"],
        dataset=train_dataset,
        batch_size=elementref_config.get("batch_size", 32),
        num_batches=elementref_config.get("num_batches"),
        num_workers=config["optim"].get("num_workers", 16),
        max_num_elements=elementref_config.get("max_num_elements", 118),
        driver=elementref_config.get("driver", None),
    )

    for target, references in element_refs.items():
        path = save_checkpoint(
            references.state_dict(),
            output_path,
            f"{target}_linref.pt",
        )
        logging.info(f"{target} linear references have been saved to: {path}")


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
        help="Output path to save linear references",
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

    fit_linref(config, args.out_path)
