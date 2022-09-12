"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import time

import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("energy")
class EnergyTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_. # noqa: E501


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a
            SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable
            for distributed training. (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model_attributes,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        frame_averaging=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        new_gnn=True,
        data_split=None,
        test_rotation_invariance=False,
        note="",
        wandb_tag=None,
    ):
        super().__init__(
            task=task,
            model_attributes=model_attributes,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
            slurm=slurm,
            new_gnn=new_gnn,
            data_split=data_split,
            note=note,
            frame_averaging=frame_averaging,
            test_rotation_invariance=test_rotation_invariance,
            wandb_tag=wandb_tag,
        )

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    @torch.no_grad()
    def predict(self, loader, per_image=True, results_file=None, disable_tqdm=False):
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        predictions = {"id": [], "energy": []}

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out, _ = self._forward(batch, self.frame_averaging)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])

            if per_image:
                predictions["id"].extend([str(i) for i in batch[0].sid.tolist()])
                predictions["energy"].extend(out["energy"].tolist())
            else:
                predictions["energy"] = out["energy"].detach()
                return predictions

        self.save_results(predictions, results_file, keys=["energy"])

        if self.ema:
            self.ema.restore()

        return predictions

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        self.config["cmd"]["print_every"] = eval_every  # Temporary
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_mae = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        start_time = time.time()
        print("---Beginning of Training---")

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)
            print("Epoch: ", epoch_int)
            if epoch_int == 1 and self.logger is not None:
                self.logger.log({"Epoch time": time.time() - start_time})

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out, pooling_loss = self._forward(batch, self.frame_averaging)
                    loss = self._compute_loss(out, batch)
                    if pooling_loss is not None:
                        loss += pooling_loss
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                self._log_metrics()

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file=f"checkpoint-{str(self.step).zfill(6)}.pt",
                        training_state=True,
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[self.evaluator.task_primary_metric[self.name]][
                                "metric"
                            ]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

                        # Evaluate current model on all 4 validation splits
                        if ((epoch_int % 100 == 0) and (epoch_int != 0)) or (
                            epoch_int == self.config["optim"]["max_epochs"] - 1
                        ):
                            self.eval_all_val_splits(
                                epoch_int == self.config["optim"]["max_epochs"] - 1
                            )

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

        # Time model
        if self.logger is not None:
            start_time = time.time()
            # batch = next(iter(self.train_loader))
            self._forward(batch, self.frame_averaging)
            self.logger.log({"Batch time": time.time() - start_time})

        # Check rotation invariance
        if self.test_rotation_invariance:
            energy_diff_z, energy_diff = self._test_rotation_invariance()
            self.logger.log({"2D_ri": energy_diff_z})
            self.logger.log({"3D_ri": energy_diff})

        self.train_dataset.close_db()
        if "val_dataset" in self.config:
            self.val_dataset.close_db()
        if "test_dataset" in self.config:
            self.test_dataset.close_db()

    def _forward(self, batch_list, fa=False):

        if fa == "full":
            y1, p1 = self.model(batch_list)
            original_pos = batch_list[0].pos
            batch_list[0].pos = batch_list[0].new_pos
            y2, p2 = self.model(batch_list)
            output = (y1 + y2) / 2
            batch_list[0].pos = original_pos
            try:
                pooling_loss = (p1 + p2) / 2
            except TypeError:
                pooling_loss = None
        else:
            output, pooling_loss = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {
            "energy": output,
        }, pooling_loss

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        loss = self.loss_fn["energy"](out["energy"], target_normed)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        return metrics

    def _log_metrics(self):
        log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
        log_dict.update(
            {
                "lr": self.scheduler.get_lr(),
                "epoch": self.epoch,
                "step": self.step,
            }
        )
        if (
            self.step % self.config["cmd"]["print_every"] == 0
            and distutils.is_master()
            and not self.is_hpo
        ):
            log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
            print(", ".join(log_str))
            self.metrics = {}

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split="train",
            )

    @torch.no_grad()
    def _test_rotation_invariance(self):
        # Check for rotation invariance
        self.model.eval()
        loader_iter = iter(self.val_loader)
        energy_diff = torch.zeros(1, device=self.device)
        energy_diff_z = torch.zeros(1, device=self.device)

        for i in range(10):
            batch = next(loader_iter)
            energy_diff_z += self.rotation_invariance_check(batch, "z")

        for i in range(10):
            batch = next(loader_iter)
            energy_diff += self.rotation_invariance_check(batch)

        batch_size = len(batch[0].natoms)
        energy_diff_z = energy_diff_z / (i * batch_size)
        energy_diff = energy_diff / (i * batch_size)

        return energy_diff_z, energy_diff
