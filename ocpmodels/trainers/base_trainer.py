"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import ocpmodels
from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.balanced_batch_sampler import BalancedBatchSampler
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.common.utils import load_state_dict, save_checkpoint
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import AtomwiseL2Loss, DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.scheduler import LRScheduler


@registry.register_trainer("base")
class BaseTrainer(ABC):
    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, (OCPDataParallel, DistributedDataParallel)):
            module = module.module
        return module

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir=None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed=None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        name: str = "base_trainer",
        slurm={},
        noddp: bool = False,
    ) -> None:
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0

        self.device: torch.device
        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        self.timestamp_id: str
        if timestamp_id is None:
            timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
                self.device
            )
            # create directories from master rank only
            distutils.broadcast(timestamp, 0)
            timestamp = datetime.datetime.fromtimestamp(
                timestamp.float().item()
            ).strftime("%Y-%m-%d-%H-%M-%S")
            if identifier:
                self.timestamp_id = f"{timestamp}-{identifier}"
            else:
                self.timestamp_id = timestamp
        else:
            self.timestamp_id = timestamp_id

        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        assert_is_instance(ocpmodels.__path__[0], str),
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "trainer": "forces" if name == "s2ef" else "energy",
            "model": assert_is_instance(model.pop("name"), str),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "noddp": noddp,
        }
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "%s_%s" % (
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", None)
            self.config["val_dataset"] = dataset.get("val", None)
            self.config["test_dataset"] = dataset.get("test", None)
        else:
            self.config["dataset"] = dataset

        self.normalizer = normalizer
        # This supports the legacy way of providing norm parameters in dataset
        if self.config.get("dataset", None) is not None and normalizer is None:
            self.normalizer = self.config["dataset"]

        if not is_debug and distutils.is_master() and not is_hpo:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_hpo = is_hpo

        if self.is_hpo:
            # conditional import is necessary for checkpointing

            # sets the hpo checkpoint frequency
            # default is no checkpointing
            self.hpo_checkpoint_every = self.config["optim"].get(
                "checkpoint_every", -1
            )

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))
        self.load()

        self.evaluator = Evaluator(task=name)

    def load(self) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()

    def load_seed_from_config(self) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self) -> None:
        self.logger = None
        if not self.is_debug and distutils.is_master() and not self.is_hpo:
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            self.logger = registry.get_logger_class(logger_name)(self.config)

    def get_sampler(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
    ) -> BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
        )
        return sampler

    def get_dataloader(self, dataset, sampler) -> DataLoader:
        loader = DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader

    def load_datasets(self) -> None:
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.config.get("dataset", None):
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            if self.config.get("val_dataset", None):
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["val_dataset"])
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            if self.config.get("test_dataset", None):
                self.test_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["test_dataset"])
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                self.normalizers["target"] = Normalizer(
                    tensor=self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    device=self.device,
                )

    @abstractmethod
    def load_task(self):
        """Initialize task-specific information. Derived classes should implement this function."""

    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        self.model = registry.get_model_class(self.config["model"])(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: Dict = {}
    ) -> None:
        if not checkpoint:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            else:
                logging.info(f"Loading checkpoint from: {checkpoint_path}")
                map_location = torch.device("cpu") if self.cpu else self.device
                checkpoint = torch.load(
                    checkpoint_path, map_location=map_location
                )

        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self) -> None:
        self.loss_fn: Dict[str, str] = {
            "energy": self.config["optim"].get("loss_energy", "mae"),
            "force": self.config["optim"].get("loss_force", "mae"),
        }
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ["l1", "mae"]:
                self.loss_fn[loss] = nn.L1Loss()
            elif loss_name == "mse":
                self.loss_fn[loss] = nn.MSELoss()
            elif loss_name == "l2mae":
                self.loss_fn[loss] = L2MAELoss()
            elif loss_name == "atomwisel2":
                self.loss_fn[loss] = AtomwiseL2Loss()
            else:
                raise NotImplementedError(
                    f"Unknown loss function name: {loss_name}"
                )
            self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])

    def load_optimizer(self) -> None:
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        if self.config["optim"].get("weight_decay", 0) > 0:
            # Do not regularize bias etc.
            params_decay = []
            params_no_decay = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        params_no_decay += [param]
                    elif "frequencies" in name:
                        params_no_decay += [param]
                    elif "bias" in name:
                        params_no_decay += [param]
                    else:
                        params_decay += [param]

            self.optimizer = optimizer(
                [
                    {"params": params_no_decay, "weight_decay": 0},
                    {
                        "params": params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

    def load_extras(self) -> None:
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
        self,
        metrics=None,
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ):
        if not self.is_debug and distutils.is_master():
            if training_state:
                return save_checkpoint(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.scheduler.state_dict()
                        if self.scheduler.scheduler_type != "Null"
                        else None,
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.config["task"].get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
                return ckpt_path
        return None

    def save_hpo(self, epoch, step: int, metrics, checkpoint_every: int):
        # default is no checkpointing
        # checkpointing frequency can be adjusted by setting checkpoint_every in steps
        # to checkpoint every time results are communicated to Ray Tune set checkpoint_every=1
        if checkpoint_every != -1 and step % checkpoint_every == 0:
            with tune.checkpoint_dir(  # noqa: F821
                step=step
            ) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(self.save_state(epoch, step, metrics), path)

    def hpo_update(
        self, epoch, step, train_metrics, val_metrics, test_metrics=None
    ):
        progress = {
            "steps": step,
            "epochs": epoch,
            "act_lr": self.optimizer.param_groups[0]["lr"],
        }
        # checkpointing must occur before reporter
        # default is no checkpointing
        self.save_hpo(
            epoch,
            step,
            val_metrics,
            self.hpo_checkpoint_every,
        )
        # report metrics to tune
        tune_reporter(  # noqa: F821
            iters=progress,
            train_metrics={
                k: train_metrics[k]["metric"] for k in self.metrics
            },
            val_metrics={k: val_metrics[k]["metric"] for k in val_metrics},
            test_metrics=test_metrics,
        )

    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    @abstractmethod
    def _forward(self, batch_list):
        """Derived classes should implement this function."""

    @abstractmethod
    def _compute_loss(self, out, batch_list):
        """Derived classes should implement this function."""

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

    def save_results(
        self, predictions, results_file: Optional[str], keys
    ) -> None:
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                f"{self.name}_{results_file}.npz",
            )

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(
                gather_results["ids"],
            )[idx]
            for k in keys:
                if k == "forces":
                    gather_results[k] = np.concatenate(
                        np.array(gather_results[k], dtype=object)[idx]
                    )
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(
                        np.array(
                            gather_results[k],
                        )[idx]
                    )[:-1]
                else:
                    gather_results[k] = np.array(
                        gather_results[k],
                    )[idx]

            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)
