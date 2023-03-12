import os
import logging
import shutil
import time
from logging.handlers import QueueHandler
from typing import Dict, Any, List
import datetime
import yaml

import torch
from torch import nn, Tensor
import torch.cuda
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp

import cv_lib.utils as cv_utils
from cv_lib.optimizers import get_optimizer
from cv_lib.schedulers import get_scheduler
import cv_lib.distributed.utils as dist_utils
from cv_lib.utils.cuda_utils import preserve_gpu_with_id

from schema_inference.data import build_train_dataloader
from models import get_model, ModelWrapper
from schema_inference.loss import get_loss_fn, Loss
import schema_inference.utils as utils
from schema_inference.eval import Evaluation


class Trainer:
    def __init__(
        self,
        train_cfg: Dict[str, Any],
        log_args: utils.LogArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        model: nn.Module,
        loss: Loss,
        loss_weights: Dict[str, float],
        evaluator: Evaluation,
        distributed: bool,
        device: torch.device,
        resume: str = "",
        use_amp: bool = False
    ):
        # set up logger
        self.logger = logging.getLogger("trainer_rank_{}".format(dist_utils.get_rank()))

        # only write in master process
        self.tb_writer = None
        if dist_utils.is_main_process():
            self.tb_writer, _ = cv_utils.get_tb_writer(log_args.logdir, log_args.filename)
        dist_utils.barrier()

        self.train_cfg = train_cfg
        self.start_epoch = 0
        self.epoch = 0
        self.total_epoch = self.train_cfg["train_epochs"]
        self.iter = 0
        self.step = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_step = len(self.train_loader)
        self.total_iter = self.total_step * self.total_epoch
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.loss = loss
        self.loss_weights = loss_weights
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device
        self.ckpt_path = log_args.ckpt_path
        self.amp = use_amp
        # best index
        self.best_acc = 0
        self.best_iter = 0

        # for pytorch amp
        self.scaler: amp.GradScaler = None
        if self.amp:
            self.logger.info("Using AMP train")
            self.scaler = amp.GradScaler()
        self.resume(resume)
        self.logger.info("Start training for %d epochs", self.train_cfg["train_epochs"] - self.start_epoch)

    def resume(self, resume_fp: str = ""):
        """
        Resume training from checkpoint
        """
        # not a valid file
        if not os.path.isfile(resume_fp):
            return
        ckpt = torch.load(resume_fp, map_location="cpu")

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            real_model = self.model.module
        else:
            real_model = self.model
        if isinstance(real_model, ModelWrapper):
            real_model = real_model.module
        real_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.best_iter = ckpt.get("best_iter", 0)
        self.best_acc = ckpt.get("best_acc", 0)
        # load grad scaler
        if self.scaler is not None and "grad_scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["grad_scaler"])
        self.iter = ckpt["iter"] + 1
        self.start_epoch = ckpt["epoch"] + 1
        self.logger.info("Loaded ckpt with epoch: %d, iter: %d", ckpt["epoch"], ckpt["iter"])

    def train_iter(self, x: Tensor, targets: List[Dict[str, Any]]):
        self.model.train()
        self.loss.train()
        # move to device
        x, targets = utils.move_data_to_device(x, targets, self.device)

        self.optimizer.zero_grad()
        with amp.autocast(enabled=self.amp):
            output = self.model(x)
            loss_dict: Dict[str, torch.Tensor] = self.loss(output, targets)
            weighted_loss: Dict[str, torch.Tensor] = dict()
            for k, loss in loss_dict.items():
                k_prefix = k.split(".")[0]
                if k_prefix in self.loss_weights:
                    weighted_loss[k] = loss * self.loss_weights[k_prefix]
            loss: torch.Tensor = sum(weighted_loss.values())

        if self.amp:
            self.scaler.scale(loss).backward()
            # grad clip
            if "clip_max_norm" in self.train_cfg:
                # Un-scales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # grad clip
            if "clip_max_norm" in self.train_cfg:
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg["clip_max_norm"]
                )
            self.optimizer.step()

        # save memory
        self.optimizer.zero_grad(set_to_none=True)

        weighted_loss: torch.Tensor = dist_utils.reduce_tensor(loss.detach())
        loss_dict = dist_utils.reduce_dict(loss_dict)
        # print
        if self.iter % self.train_cfg["print_interval"] == 0 and dist_utils.is_main_process():
            loss_dict = cv_utils.tensor_dict_items(loss_dict, ndigits=4)
            # reduce loss
            self.logger.info(
                "Epoch %3d|%3d, step %4d|%4d, iter %5d|%5d, lr:\n%s,\nloss: %.5f, loss dict: %s",
                self.epoch, self.total_epoch,
                self.step, self.total_step,
                self.iter, self.total_iter,
                cv_utils.to_json_str(self.lr_scheduler.get_last_lr()),
                weighted_loss.item(),
                cv_utils.to_json_str(loss_dict)
            )
            self.tb_writer.add_scalar("Train/Loss", weighted_loss, self.iter)
            error_dict = dict()
            for k in list(loss_dict.keys()):
                if "error" in k:
                    error_dict[k] = loss_dict.pop(k)
            self.tb_writer.add_scalars("Train/Loss_dict", loss_dict, self.iter)
            self.tb_writer.add_scalars("Train/Error_dict", error_dict, self.iter)
            self.tb_writer.add_scalar("Train/Lr", self.lr_scheduler.get_last_lr()[0], self.iter)
        dist_utils.barrier()
        self.iter += 1

    def validate_and_save(self, show_tb: bool = True):
        self.logger.info("Start evaluation")
        eval_dict = self.evaluator(self.model)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            real_model = self.model.module
        else:
            real_model = self.model
        if isinstance(real_model, ModelWrapper):
            real_model = real_model.module
        model_state_dict = real_model.state_dict()

        if dist_utils.is_main_process():
            self.logger.info("evaluation done")
            loss = eval_dict["loss"]
            loss_dict = eval_dict["loss_dict"]
            loss_dict = cv_utils.tensor_dict_items(loss_dict, ndigits=4)
            acc_dict: Dict[int, float] = cv_utils.tensor_dict_items(eval_dict["acc"], ndigits=4)
            acc_top_1 = acc_dict[1]
            # write logger
            info = "Validation loss: {:.5f}, acc@1: {:.4f}\nloss dict: {}"
            info = info.format(
                loss,
                acc_top_1,
                cv_utils.to_json_str(loss_dict)
            )
            self.logger.info(info)
            if show_tb:
                # write tb logger, compatible with mutual training
                self.tb_writer.add_scalar("Val/Loss", loss, self.iter)
                self.tb_writer.add_scalar("Val/Acc@1", acc_top_1, self.iter)
                self.tb_writer.add_scalar("Val/Err@1", 1 - acc_top_1, self.iter)
                self.tb_writer.add_scalars("Val/Loss_dict", loss_dict, self.iter)

            # save ckpt
            state_dict = {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": self.epoch,
                "iter": self.iter,
                "eval_dict": eval_dict,
                "loss_dict": loss_dict,
                "best_acc": self.best_acc,
                "best_iter": self.best_iter
            }
            if self.scaler is not None:
                state_dict["grad_scaler"] = self.scaler.state_dict()
            save_fp = os.path.join(self.ckpt_path, f"iter-{self.iter}.pth")
            self.logger.info("Saving state dict to %s...", save_fp)
            torch.save(state_dict, save_fp)
            if acc_top_1 > self.best_acc:
                # best index
                self.best_acc = acc_top_1
                self.best_iter = self.iter
                shutil.copy(save_fp, os.path.join(self.ckpt_path, "best.pth"))
        dist_utils.barrier()

    def __call__(self):
        start_time = time.time()
        self.validate_and_save(show_tb=False)
        # start one epoch
        for self.epoch in range(self.start_epoch, self.train_cfg["train_epochs"]):
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            for self.step, (x, target) in enumerate(self.train_loader):
                self.train_iter(x, target)
                # validation
                if self.iter % self.train_cfg["val_interval"] == 0:
                    self.validate_and_save()
            self.lr_scheduler.step()
        self.logger.info("Final validation")
        self.validate_and_save()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if dist_utils.is_main_process():
            self.logger.info("Training time %s", total_time_str)
            self.logger.info("Best acc: %f, iter: %d", self.best_acc, self.best_iter)


def backbone_worker(
    gpu_id: int,
    launch_args: utils.DistLaunchArgs,
    log_args: utils.LogArgs,
    global_cfg: Dict[str, Any],
    resume: str = ""
):
    """
    What created in this function is only used in this process and not shareable
    """
    # setup process root logger
    if launch_args.distributed:
        root_logger = logging.getLogger()
        handler = QueueHandler(log_args.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    # split configs
    data_cfg: Dict[str, Any] = cv_utils.get_cfg(global_cfg["dataset"])
    train_cfg: Dict[str, Any] = global_cfg["training"]
    val_cfg: Dict[str, Any] = global_cfg["validation"]
    model_cfg: Dict[str, Any] = global_cfg["model"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    # set debug number of workers
    if launch_args.debug:
        train_cfg["num_workers"] = 0
        val_cfg["num_workers"] = 0
        train_cfg["print_interval"] = 1
        train_cfg["val_interval"] = 10
    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            current_rank = launch_args.rank * launch_args.ngpus_per_node + gpu_id
        dist.init_process_group(
            backend=launch_args.backend,
            init_method=launch_args.master_url,
            world_size=launch_args.world_size,
            rank=current_rank
        )

    assert dist_utils.get_rank() == current_rank, "code bug"
    # set up process logger
    logger = logging.getLogger("worker_rank_{}".format(current_rank))

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make deterministic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        cv_utils.make_deterministic(seed)
    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    # preserve memory
    if launch_args.preserve_gpu:
        preserve_gpu_with_id(
            gpu_id,
            preserve_percent=launch_args.preserve_percent
        )

    # get dataloader
    logger.info("Building dataset...")
    train_loader, val_loader, n_classes, _ = build_train_dataloader(
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args,
    )
    # create model
    logger.info("Building model...")
    model = get_model(model_cfg, n_classes)
    logger.info(
        "Built model with %d parameters, %d trainable parameters",
        cv_utils.count_parameters(model, include_no_grad=True),
        cv_utils.count_parameters(model, include_no_grad=False)
    )
    if model_cfg.get("pre_train", None) is not None:
        lax_names = model_cfg.get("lax_names", list())
        utils.load_pretrain_model(
            pretrain_fp=model_cfg["pre_train"],
            model=model,
            lax_names=lax_names
        )
        logger.info("Loaded pretrain model: %s", model_cfg["pre_train"])
    model.to(device)
    model_without_ddp = model
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    optimizer = get_optimizer(model_without_ddp.parameters(), train_cfg["optimizer"])
    logger.info("Loaded optimizer:\n%s", optimizer)
    lr_scheduler = get_scheduler(optimizer, train_cfg["lr_schedule"])

    loss = get_loss_fn(loss_cfg)
    loss.to(device)

    evaluator = Evaluation(
        loss_fn=loss,
        val_loader=val_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device,
        top_k=(1,)
    )

    trainer = Trainer(
        train_cfg=train_cfg,
        log_args=log_args,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model=model,
        loss=loss,
        loss_weights=loss_cfg["weight_dict"],
        evaluator=evaluator,
        distributed=distributed,
        device=device,
        resume=resume,
        use_amp=launch_args.use_amp
    )
    # start training
    trainer()

