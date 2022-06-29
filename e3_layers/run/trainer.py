import sys
import inspect
import logging
from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Union

if sys.version_info[1] >= 7:
    import contextlib
else:
    import contextlib2 as contextlib
from absl.flags import FLAGS
from tqdm import tqdm

import wandb
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from e3nn.o3 import Irreps
from ml_collections.config_dict import ConfigDict

from ..data import DataLoader, CondensedDataset
from ..utils import (
    build,
    pruneArgs,
    save_file,
    load_file,
    atomic_write,
    finish_all_writes,
    atomic_write_group,
)

from .loss import Loss, LossStat
from .metrics import Metrics
from .early_stopping import EarlyStopping

from torch.profiler import profile, schedule, ProfilerActivity


class Trainer:
    stop_keys = ["max_epochs", "early_stopping", "early_stopping_kwargs"]
    object_keys = ["lr_sched", "optim", "ema", "early_stopping_conds"]

    def __init__(
        self,
        model,
        data_config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_coeffs: Union[dict, str] = None,
        train_on_keys: Optional[List[str]] = None,
        metrics_components: Optional[Union[dict, str]] = None,
        metrics_key="validation_loss",
        early_stopping_conds: Optional[EarlyStopping] = None,
        max_epochs: int = 1000000,
        learning_rate: float = 1e-2,
        lr_scheduler_name: str = "none",
        optimizer_name: str = "Adam",
        max_gradient_norm: float = float("inf"),
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_use_num_updates=True,
        batch_size: int = 5,
        train_idcs: Optional[list] = None,
        val_idcs: Optional[list] = None,
        epoch_subdivision: int = 1,
        **kwargs,
    ):
        self._initialized = False
        logging.debug("* Initialize Trainer")

        # store all init arguments
        for key in self.init_keys:
            setattr(self, key, locals()[key])

        self.model = model
        self.data_config = data_config
        self.last_model_path = os.path.join(FLAGS.workdir, "last.pt")
        self.best_model_path = os.path.join(FLAGS.workdir, "best.pt")
        self.trainer_save_path = os.path.join(FLAGS.workdir, "trainer.pt")
        self.ema = None
        self.device = dist.get_rank()
        self.rank = dist.get_rank()
        self.torch_device = torch.device(self.device)

        logger = logging.getLogger()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        logger.handlers[0].setFormatter(formatter)
        if self.rank == 0:
            gfile_stream = open(os.path.join(FLAGS.workdir, "log.txt"), "w")
            handler = logging.StreamHandler(gfile_stream)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(FLAGS.verbose)
        else:
            logger = logging.getLogger(__name__)
            logger.setLevel(level=logging.WARNING)
        self.logger = logger

        self.loader_rng = (
            torch.Generator()
        )  # used for generating seeds for each dataloader worker process
        self.split_rng = torch.Generator()
        
        if FLAGS.seed is not None:
            torch.manual_seed(FLAGS.seed)
            np.random.seed(FLAGS.seed)
            self.split_rng.manual_seed(FLAGS.seed)
            self.loader_rng.manual_seed(FLAGS.seed + self.rank)

        # sort out all the other parameters
        # for samplers, optimizer and scheduler
        self.kwargs = deepcopy(kwargs)

        # initialize training states
        self.best_metrics = float("inf")
        self.best_epoch = 0
        self.iepoch = 0

        self.loss = Loss(self.loss_coeffs)
        self.loss_stat = LossStat(self.loss)

        # what do we train on?
        self.train_on_keys = self.loss.keys

        self.init_objects()

    def init_objects(self):

        self.model.to(self.torch_device)
        self.model = DDP(self.model)

        self.num_weights = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Number of weights: {self.num_weights}")

        self._initialized = True

        # initialize optimizer
        optim = getattr(torch.optim, self.optimizer_name)
        kwargs = pruneArgs(prefix="optimizer", **self.kwargs)
        self.optim = optim(
            params=self.model.parameters(), lr=self.learning_rate, **kwargs
        )

        self.max_gradient_norm = (
            float(self.max_gradient_norm)
            if self.max_gradient_norm is not None
            else float("inf")
        )

        # initialize scheduler
        assert self.lr_scheduler_name in [
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
            "none",
        ]
        self.lr_sched = None
        self.lr_scheduler_kwargs = {}
        if self.lr_scheduler_name != None:
            scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)
            kwargs = pruneArgs(prefix="lr_scheduler", **self.kwargs)
            self.lr_sched = scheduler(optimizer=self.optim, **kwargs)

        # initialize early stopping conditions
        kwargs = pruneArgs(prefix="early_stopping", **self.kwargs)
        n_args = 0
        for key, item in kwargs.items():
            if isinstance(item, dict) or isinstance(item, ConfigDict):
                new_dict = {}
                for k, v in item.items():
                    if (
                        k.lower().startswith("validation")
                        or k.lower().startswith("training")
                        or k.lower() in ["lr", "wall"]
                    ):
                        new_dict[k] = item[k]
                    else:
                        new_dict[f"{validation}_{k}"] = item[k]
                kwargs[key] = new_dict
                n_args += len(new_dict)
        self.early_stopping_conds = EarlyStopping(**kwargs) if n_args > 0 else None

        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.ema_decay,
                use_num_updates=self.ema_use_num_updates,
            )

    @property
    def init_keys(self):
        return [
            key
            for key in list(inspect.signature(Trainer.__init__).parameters.keys())
            if key not in (["self", "kwargs", "model"] + Trainer.object_keys)
        ]

    def init_metrics(self):
        if self.metrics_components is None:
            self.metrics_components = []
            for key, func in self.loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.lower().startswith("perspecies"),
                }
                self.metrics_components.append((key, "mae", params))
                self.metrics_components.append((key, "rmse", params))

        self.metrics = Metrics(
            components=self.metrics_components,
            **pruneArgs(prefix="metrics", **self.kwargs),
        )

        if not (
            self.metrics_key.lower().startswith("validation")
            or self.metrics_key.lower().startswith("training")
        ):
            raise RuntimeError(
                f"metrics_key should start with either {'validation'} or {'training'}"
            )

    def set_dataset(self, dataset, validation_dataset) -> None:
        data_config = self.data_config
        if self.train_idcs is None or self.val_idcs is None:
            if validation_dataset is None:
                # Sample both from `dataset`:
                total_n = len(dataset)
                if isinstance(data_config.n_train, float):
                    data_config.n_train = int(data_config.n_train * total_n)
                if isinstance(data_config.n_val, float):
                    data_config.n_val = int(data_config.n_val * total_n)
                if (data_config.n_train + data_config.n_val) > total_n:
                    raise ValueError(
                        "too little data for training and validation. please reduce n_train and n_val"
                    )

                if data_config.train_val_split == "random":
                    idcs = torch.randperm(total_n, generator=self.split_rng)
                elif data_config.train_val_split == "sequential":
                    idcs = torch.arange(total_n)
                else:
                    raise NotImplementedError(
                        f"splitting mode {data_config.train_val_split} not implemented"
                    )

                self.train_idcs = idcs[: data_config.n_train]
                self.val_idcs = idcs[
                    data_config.n_train : data_config.n_train + data_config.n_val
                ]
            else:
                if data_config.n_train > len(dataset):
                    raise ValueError("Not enough data in dataset for requested n_train")
                if data_config.n_val > len(validation_dataset):
                    raise ValueError("Not enough data in dataset for requested n_train")
                if data_config.train_val_split == "random":
                    self.train_idcs = torch.randperm(
                        len(dataset), generator=self.split_rng
                    )[: self.n_train]
                    self.val_idcs = torch.randperm(
                        len(validation_dataset), generator=self.split_rng
                    )[: self.n_val]
                elif data_config.train_val_split == "sequential":
                    self.train_idcs = torch.arange(self.n_train)
                    self.val_idcs = torch.arange(self.n_val)
                else:
                    raise NotImplementedError(
                        f"splitting mode {data_config.train_val_split} not implemented"
                    )

        if validation_dataset is None:
            validation_dataset = dataset

        # torch_geometric datasets inherantly support subsets using `index_select`
        self.dataset_train = dataset.index_select(self.train_idcs)
        self.dataset_val = validation_dataset.index_select(self.val_idcs)

        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        dl_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=FLAGS.dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                FLAGS.dataloader_num_workers > 0 and self.max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(self.torch_device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if FLAGS.dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=self.loader_rng,
        )
        self.dl_train = DataLoader(
            dataset=self.dataset_train,
            shuffle=data_config.shuffle,  # training should shuffle
            **dl_kwargs,
        )
        # validation, on the other hand, shouldn't shuffle
        # we still pass the generator just to be safe
        self.dl_val = DataLoader(dataset=self.dataset_val, **dl_kwargs)

    def train(self):
        """Training"""
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        if not self._initialized:
            self.init()

        self.init_log()
        self.wall = perf_counter()

        if self.rank == 0:
            with atomic_write_group():
                if self.iepoch == -1:
                    self.save()

        self.init_metrics()

        while not self.stop_cond:
            self.epoch_step()
            self.end_of_epoch_save()
        self.final_log()

        self.save()
        finish_all_writes()

    def equivarianceTest(self, out):
        mat = out["_rotation_matrix"]
        batch_size = mat.shape[0]
        for key, value in out.items():
            if key in self.dataset_train.irreps:
                transform = self.dataset_train.irreps[key]
                if isinstance(transform, Irreps):
                    D = transform.D_from_matrix(mat)
                    value = value.view(batch_size, -1, transform.dim)
                    old_std = value.std(0)
                    value = value @ D
                else:
                    old_std = value.std(0)
                    value = transform(mat.transpose(-1, -2), value)
                std = value.std(0)
                if old_std.max() < 1e-3:
                    self.logger.info(f"{key} too small to perform equivariance test")
                elif torch.allclose(
                    std, torch.zeros(std.shape, device=std.device), atol=1e-4
                ):
                    self.logger.info(f"equivariance test succeeded for {key}")
                else:
                    self.logger.warning(f"equivariance test failed for {key}")

    def batch_step(self, data, validation=False):
        # no need to have gradients from old steps taking up memory
        self.optim.zero_grad(set_to_none=True)

        if validation:
            self.model.eval()
        else:
            self.model.train()

        data = data.to(self.torch_device)
        out = self.model(data.clone())

        if FLAGS.equivariance_test:
            self.equivarianceTest(out)

        if not validation:
            # Actually do an optimization step, since we're training:
            loss, loss_contrib = self.loss(pred=out, ref=data)
            # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            # See https://stackoverflow.com/a/56069467
            # Has to happen after .backward() so there are grads to clip
            if self.max_gradient_norm < float("inf"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )

            self.optim.step()

            if self.use_ema:
                self.ema.update()

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
                self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)

        with torch.no_grad():
            if validation:
                scaled_out = out
                _data_unscaled = data
                loss, loss_contrib = self.loss(pred=scaled_out, ref=_data_unscaled)

            # save metrics stats
            self.batch_losses = self.loss_stat(loss, loss_contrib)
            self.batch_metrics = self.metrics(pred=out, ref=data)

    @property
    def stop_cond(self):
        """kill the training early"""
        if self.early_stopping_conds is not None and hasattr(self, "mae_dict"):
            early_stop, early_stop_args, debug_args = self.early_stopping_conds(
                self.mae_dict
            )
            if debug_args is not None:
                self.logger.debug(debug_args)
            if early_stop:
                self.stop_arg = early_stop_args
                return True

        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True

        return False

    def reset_metrics(self):
        self.loss_stat.reset()
        self.loss_stat.to(self.torch_device)
        self.metrics.reset()
        self.metrics.to(self.torch_device)

    def epoch_step(self):

        datasets = [self.dl_train, self.dl_val]
        categories = ["training", "validation"]
        iterables = [iter(self.dl_train), iter(self.dl_val)]

        for idivision in range(self.epoch_subdivision):
            self.metrics_dict = {}
            self.loss_dict = {}

            for category, dataset, iterable in zip(categories, datasets, iterables):
                split_size = len(dataset) // self.epoch_subdivision
                if category == "validation" and self.use_ema:
                    cm = self.ema.average_parameters()
                elif category == 'training' and FLAGS.profiling:
                    my_schedule=torch.profiler.schedule(
                            wait=1,
                            warmup=1,
                            active=3)
                    cm = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True, with_stack=True, profile_memory=True,
                             schedule=my_schedule)
                else:
                    cm = contextlib.nullcontext()

                pbar = range(
                    idivision * split_size,
                    min((idivision + 1) * split_size, len(dataset)),
                )
                if self.rank == 0:
                    pbar = tqdm(pbar)
                    s = "training" if category == "training" else "valid"
                    pbar.set_description(
                        f"{FLAGS.name} {s} epoch{self.iepoch}-{idivision}"
                    )

                with cm:
                    self.reset_metrics()
                    self.n_batches = len(dataset)
                    for self.ibatch in pbar:
                        batch = next(iterable)
                        self.batch_step(
                            data=batch,
                            validation=(category == "validation"),
                        )
                        if (self.ibatch + 1) % FLAGS.log_period == 0 or (
                            self.ibatch + 1
                        ) == self.n_batches:
                            if self.rank == 0:
                                self.end_of_batch_log(batch_type=category)
                        if (self.ibatch + 1) % (
                            len(dataset) // self.epoch_subdivision
                        ) == 0:
                          
                            break
                        if category == 'training' and FLAGS.profiling:
                            cm.step()

                    self.metrics_dict[category] = self.metrics.current_result()
                    self.loss_dict[category] = self.loss_stat.current_result()
                if category == 'training' and FLAGS.profiling:
                    with open(os.path.join(FLAGS.workdir, "profiling.txt"), 'w') as f:
                        f.write(cm.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
                        f.write(cm.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total', row_limit=10))
                        f.write(cm.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                    cm.export_chrome_trace(os.path.join(FLAGS.workdir, "profiling.json"))
                    # currently this causes segfault https://github.com/pytorch/pytorch/issues/69443

            self.end_of_epoch_log()

            if self.lr_scheduler_name == "ReduceLROnPlateau":
                self.lr_sched.step(metrics=self.mae_dict[self.metrics_key])
        self.iepoch += 1
        
        data_config = self.data_config
        if 'reload' in data_config and data_config.reload:
            dataset = CondensedDataset(**data_config)
            self.set_dataset(dataset, validation_dataset=None)

    @property
    def epoch_logger(self):
        return self.logger

    @property
    def init_epoch_logger(self):
        return self.logger

    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """
        log_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value
        for name, value in self.batch_losses.items():
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        # append details from metrics
        metrics, skip_keys = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            type_names=self.dataset_train.type_names,
        )
        for key, value in metrics.items():
            header += f", {key}"
            if key not in skip_keys:
                log_str += f" {value:12.3g}"
                log_header += f" {key:>12.12}"

        self.logger.info("")
        self.logger.info(f"{batch_type}")
        self.logger.info(log_header)
        self.logger.info(log_str)

    def init_log(self):
        if self.iepoch > 0:
            self.logger.info("! Restarting training ...")
        else:
            self.logger.info("! Starting training ...")

    def final_log(self):
        self.logger.info(f"! Stop training: {self.stop_arg}")
        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    def end_of_epoch_log(self):
        """
        log validation details at the end of each epoch
        """

        lr = self.optim.param_groups[0]["lr"]
        wall = perf_counter() - self.wall
        self.mae_dict = dict(
            LR=lr,
            epoch=self.iepoch,
            wall=wall,
        )

        header = "epoch, wall, LR"

        categories = ["training", "validation"] if self.iepoch > 0 else ["validation"]
        log_header = {}
        log_str = {}

        strings = ["Epoch", "wal", "LR"]
        mat_str = f"{self.iepoch:10d}, {wall:8.3f}, {lr:8.3g}"
        for cat in categories:
            log_header[cat] = "# "
            log_header[cat] += " ".join([f"{s:>8s}" for s in strings])
            log_str[cat] = f"{self.iepoch:10d} {wall:8.3f} {lr:8.3g}"

        for category in categories:

            met, skip_keys = self.metrics.flatten_metrics(
                metrics=self.metrics_dict[category],
                type_names=self.dataset_train.type_names,
            )

            # append details from loss
            for key, value in self.loss_dict[category].items():
                mat_str += f", {value:16.5g}"
                header += f", {category}_{key}"
                log_str[category] += f" {value:12.3g}"
                log_header[category] += f" {key:>12.12}"
                self.mae_dict[f"{category}_{key}"] = value

            # append details from metrics
            for key, value in met.items():
                mat_str += f", {value:12.3g}"
                header += f", {category}_{key}"
                if key not in skip_keys:
                    log_str[category] += f" {value:12.3g}"
                    log_header[category] += f" {key:>12.12}"
                self.mae_dict[f"{category}_{key}"] = value

        if self.rank > 0:
            return

        if self.iepoch == 0:
            self.init_epoch_logger.info(header)
            self.init_epoch_logger.info(mat_str)
        elif self.iepoch == 1:
            self.epoch_logger.info(header)

        if self.iepoch > 0:
            self.epoch_logger.info(mat_str)

        if self.iepoch > 0:
            self.logger.info("\n\n  Train      " + log_header["training"])
            self.logger.info("! Train      " + log_str["training"])
            self.logger.info("! Validation " + log_str["validation"])
        else:
            self.logger.info("\n\n  Initialization     " + log_header["validation"])
            self.logger.info("! Initial Validation " + log_str["validation"])

        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    @property
    def params(self):
        return self.as_dict(state_dict=False, training_progress=False, kwargs=False)

    def as_dict(
        self,
        state_dict: bool = False,
        training_progress: bool = False,
        kwargs: bool = True,
    ):
        """convert instance to a dictionary
        Args:

        state_dict (bool): if True, the state_dicts of the optimizer, lr scheduler, and EMA will be included
        """

        dictionary = {}

        for key in self.init_keys:
            dictionary[key] = getattr(self, key, None)

        if kwargs:
            dictionary.update(getattr(self, "kwargs", {}))

        if state_dict:
            dictionary["state_dict"] = {}
            for key in Trainer.object_keys:
                item = getattr(self, key, None)
                if item is not None:
                    dictionary["state_dict"][key] = item.state_dict()
            dictionary["state_dict"]["rng_state"] = torch.get_rng_state()
            dictionary["state_dict"]["split_rng_state"] = self.split_rng.get_state()
            dictionary["state_dict"]["loader_rng_state"] = self.loader_rng.get_state()
            if torch.cuda.is_available():
                dictionary["state_dict"]["cuda_rng_state"] = torch.cuda.get_rng_state(
                    device=self.torch_device
                )

        if training_progress:
            dictionary["progress"] = {}
            for key in ["iepoch", "best_epoch"]:
                dictionary["progress"][key] = self.__dict__.get(key, -1)
            dictionary["progress"]["best_metrics"] = self.__dict__.get(
                "best_metrics", float("inf")
            )
            dictionary["progress"]["stop_arg"] = self.__dict__.get("stop_arg", None)

            # TODO: these might not both be available, str defined, but no weights
            dictionary["progress"]["best_model_path"] = self.best_model_path
            dictionary["progress"]["last_model_path"] = self.last_model_path
            dictionary["progress"]["trainer_save_path"] = self.trainer_save_path
            if hasattr(self, "config_save_path"):
                dictionary["progress"]["config_save_path"] = self.config_save_path

        return dictionary

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """
        if self.rank > 0:
            return
        with atomic_write_group():
            current_metrics = self.mae_dict[self.metrics_key]
            if current_metrics < self.best_metrics:
                self.best_metrics = current_metrics
                self.best_epoch = self.iepoch

                self.save_ema_model(self.best_model_path, blocking=False)

                self.logger.info(
                    f"! Best model {self.best_epoch:8d} {self.best_metrics:8.3f}"
                )

            if FLAGS.save_period > 0 and (self.iepoch + 1) % FLAGS.save_period == 0:
                self.save(blocking=False)
                ckpt_path = self.last_model_path
                self.save_model(ckpt_path, blocking=False)

    def save_ema_model(self, path, blocking: bool = True):

        if self.use_ema:
            # If using EMA, store the EMA validation model
            # that gave us the good val metrics that made the model "best"
            # in the first place
            cm = self.ema.average_parameters()
        else:
            # otherwise, do nothing
            cm = contextlib.nullcontext()

        with cm:
            self.save_model(path, blocking=blocking)

    def save_model(self, path, blocking: bool = True):
        with atomic_write(path, blocking=blocking, binary=True) as write_to:
            if isinstance(self.model, DDP):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            torch.save(state_dict, write_to)

    def save(self, filename: Optional[str] = None, format=None, blocking: bool = True):
        """save the file as filename

        Args:

        filename (str): name of the file
        format (str): format of the file. yaml and json format will not save the weights.
        """

        if filename is None:
            filename = self.trainer_save_path

        logger = self.logger

        state_dict = (
            True
            if format == "torch"
            or filename.endswith(".pth")
            or filename.endswith(".pt")
            else False
        )

        filename = save_file(
            item=self.as_dict(state_dict=state_dict, training_progress=True),
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
            blocking=blocking,
        )
        logger.debug(f"Saved trainer to {filename}")

        self.save_model(self.last_model_path, blocking=blocking)
        logger.debug(f"Saved last model to {self.last_model_path}")

        return filename

    @classmethod
    def from_file(
        cls, filename: str, **kwargs
    ):
        """load a model from file

        Args:

        filename (str): name of the file
        """

        dictionary = load_file(
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename
        )
        return cls.from_dict(dictionary, **kwargs)

    @classmethod
    def from_dict(cls, dictionary, **kwargs):
        """load model from dictionary

        Args:

        dictionary (dict):
        """

        dictionary = deepcopy(dictionary)

        model_config = kwargs['model_config']
        model = None
        iepoch = -1
        if "model" in dictionary:
            model = dictionary.pop("model")
        elif "progress" in dictionary:
            progress = dictionary["progress"]

            # load the model from file
            iepoch = progress["iepoch"]
            if os.path.isfile(progress["last_model_path"]):
                load_path = Path(progress["last_model_path"])
                iepoch = progress["iepoch"]
            else:
                raise AttributeError("model weights & bias are not saved")

            model = Trainer.load_model_from_training_session(
                traindir=load_path.parent,
                model_name=load_path.name,
                model_config=model_config
            )
            logging.debug(f"Reload the model from {load_path}")

            dictionary.pop("progress")

        state_dict = dictionary.pop("state_dict", None)

        trainer = cls(model=model, **dictionary)

        if state_dict is not None and trainer.model is not None:
            logging.debug("Reload optimizer and scheduler states")
            for key in Trainer.object_keys:
                item = getattr(trainer, key, None)
                if item is not None:
                    item.load_state_dict(state_dict[key])
            trainer._initialized = True

            torch.set_rng_state(state_dict["rng_state"])
            trainer.split_rng.set_state(state_dict["split_rng_state"])
            trainer.loader_rng.set_state(state_dict["loader_rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])

        if "progress" in dictionary:
            trainer.best_metrics = progress["best_metrics"]
            trainer.best_epoch = progress["best_epoch"]
            stop_arg = progress.pop("stop_arg", None)
        else:
            trainer.best_metrics = float("inf")
            trainer.best_epoch = 0
            stop_arg = None
        trainer.iepoch = iepoch

        # final sanity check
        if trainer.stop_cond:
            raise RuntimeError(
                f"The previous run has properly stopped with {stop_arg}."
                "Please either increase the max_epoch or change early stop criteria"
            )

        return trainer

    @staticmethod
    def load_model_from_training_session(
        traindir,
        model_config,
        model_name="best_model.pth",
        device="cpu"
    ):
        traindir = str(traindir)
        model_name = str(model_name)

        model = build(model_config)
        if model is not None: 
            model.to(
                device=torch.device(device)
            )
            model_state_dict = torch.load(
                traindir + "/" + model_name, map_location=device
            )
            model.load_state_dict(model_state_dict)
        model = model

        return model


class TrainerWandB(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        print("logging with wandb")
        print(self.mae_dict)
        wandb.log(self.mae_dict)

    def init(self, config={}):
        super().init()

        if not self._initialized:
            return

        # upload some new fields to wandb
        wandb.config.update({"num_weights": self.num_weights})

        if self.kwargs.get("wandb_watch", False):
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            wandb.watch(self.model, **wandb_watch_kwargs)