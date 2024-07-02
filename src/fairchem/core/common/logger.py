"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from fairchem.core.common.registry import registry


class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def watch(self, model, log_freq: int = 1000):
        """
        Monitor parameters and gradients.
        """

    def log(self, update_dict, step: int, split: str = ""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict
        return update_dict

    @abstractmethod
    def log_plots(self, plots) -> None:
        pass

    @abstractmethod
    def mark_preempting(self) -> None:
        pass

    @abstractmethod
    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        pass


@registry.register_logger("wandb")
class WandBLogger(Logger):
    def __init__(self, config) -> None:
        super().__init__(config)
        project = (
            self.config["logger"].get("project", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        entity = (
            self.config["logger"].get("entity", None)
            if isinstance(self.config["logger"], dict)
            else None
        )

        wandb.init(
            config=self.config,
            id=self.config["cmd"]["timestamp_id"],
            name=self.config["cmd"]["identifier"],
            dir=self.config["cmd"]["logs_dir"],
            project=project,
            entity=entity,
            resume="allow",
        )

    def watch(self, model, log_freq: int = 1000) -> None:
        wandb.watch(model, log_freq = log_freq)

    def log(self, update_dict, step: int, split: str = "") -> None:
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption: str = "") -> None:
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()


@registry.register_logger("tensorboard")
class TensorboardLogger(Logger):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.writer = SummaryWriter(self.config["cmd"]["logs_dir"])

    # TODO: add a model hook for watching gradients.
    def watch(self, model, log_freq: int = 1000) -> bool:
        logging.warning("Model gradient logging to tensorboard not yet supported.")
        return False

    def log(self, update_dict, step: int, split: str = ""):
        update_dict = super().log(update_dict, step, split)
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], (int, float))
                self.writer.add_scalar(key, update_dict[key], step)

    def mark_preempting(self) -> None:
        logging.warning("mark_preempting for Tensorboard not supported")

    def log_plots(self, plots) -> None:
        logging.warning("log_plots for Tensorboard not supported")

    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        logging.warning("log_summary for Tensorboard not supported")
