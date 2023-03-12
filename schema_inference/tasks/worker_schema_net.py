import os
import logging
import shutil
import time
from logging.handlers import QueueHandler
from typing import Dict, Any, List
import datetime
import yaml

import torch
import torch.nn as nn
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

import schema_inference.utils as utils
import schema_inference.graph as graph
from schema_inference.eval import Evaluation
from schema_inference.data import build_train_dataloader
from schema_inference.loss import get_loss_fn, Loss


class SchemaNetTrainer:
    def __init__(
        self,
        train_cfg: Dict[str, Any],
        log_args: utils.LogArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        predictor: graph.SchemaNetPredictor,
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
        self.predictor = predictor
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
        # get real wrapper
        if isinstance(predictor, nn.parallel.DistributedDataParallel):
            real_predictor = predictor.module
        else:
            real_predictor = predictor
        self.real_predictor: graph.SchemaNetPredictor = real_predictor

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

        self.real_predictor.load_state_dict(ckpt["predictor"])
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

    def train_iter(self, x: torch.Tensor, targets: List[Dict[str, Any]]):
        self.predictor.train()
        self.loss.train()
        # move to device
        x, targets = utils.move_data_to_device(x, targets, self.device)

        self.optimizer.zero_grad()
        self.real_predictor.schema_net.normalize()
        with amp.autocast(enabled=self.amp):
            output = self.predictor(x)
            loss_dict: Dict[str, torch.Tensor] = self.loss(output, targets)
            weighted_loss: Dict[str, torch.Tensor] = dict()
            for k, loss in loss_dict.items():
                k_prefix = k.split(".")[0]
                if k_prefix in self.loss_weights:
                    weighted_loss[k] = loss * self.loss_weights[k_prefix]
            loss: torch.Tensor = sum(weighted_loss.values())

        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
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
            self.tb_writer.add_scalars("Train/Loss_dict", loss_dict, self.iter)
            self.tb_writer.add_scalar("Train/Lr", self.lr_scheduler.get_last_lr()[0], self.iter)
            # weights
            graph = self.real_predictor.schema_net
            weights = {
                "v_geo": graph.vertex_attribute_weights.tensor[0, 0].item(),
                "v_attn": graph.vertex_attribute_weights.tensor[1, 0].item(),
                "e_geo": graph.edge_attribute_weights.tensor[0, 0].item(),
                "e_attn": graph.edge_attribute_weights.tensor[1, 0].item()
            }
            self.tb_writer.add_scalars("Weights", weights, self.iter)
        dist_utils.barrier()
        self.iter += 1

    def validate_and_save(self, show_tb: bool = True):
        self.logger.info("Start evaluation")
        self.real_predictor.schema_net.normalize()
        eval_dict = self.evaluator(self.predictor)

        if dist_utils.is_main_process():
            self.logger.info("evaluation done")
            loss = eval_dict["loss"]
            loss_dict = eval_dict["loss_dict"]
            loss_dict = cv_utils.tensor_dict_items(loss_dict, ndigits=4)
            acc_dict: Dict[int, float] = cv_utils.tensor_dict_items(eval_dict["acc"], ndigits=4)
            acc = acc_dict[1]
            # write logger
            info = "Validation loss: {:.5f}, acc: {:.4f}\nloss dict: {}"
            info = info.format(
                loss,
                acc,
                cv_utils.to_json_str(loss_dict),
                cv_utils.to_json_str(acc_dict),
            )
            self.logger.info(info)
            if show_tb:
                # write tb logger, compatible with mutual training
                self.tb_writer.add_scalar("Val/Loss", loss, self.iter)
                self.tb_writer.add_scalar("Val/Acc", acc, self.iter)
                self.tb_writer.add_scalars("Val/Loss_dict", loss_dict, self.iter)

            # save ckpt
            state_dict = {
                "predictor": self.real_predictor.state_dict(),
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
            if acc > self.best_acc:
                # best index
                self.best_acc = acc
                self.best_iter = self.iter
                shutil.copy(save_fp, os.path.join(self.ckpt_path, "best.pth"))
        dist_utils.barrier()

    def __call__(self):
        start_time = time.time()
        self.logger.info("Initial testing")
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


def schema_net_worker(
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
    schema_net_cfg: Dict[str, Any] = global_cfg["schema_net"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]

    # set debug number of workers
    if launch_args.debug:
        train_cfg["num_workers"] = 0
        train_cfg["batch_size"] = 1
        val_cfg["num_workers"] = 0
        val_cfg["batch_size"] = 1
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
    logger.info("Loading jit models...")
    backbone_jit: torch.jit.ScriptModule = torch.jit.load(schema_net_cfg["backbone_jit"], map_location=device)
    discretization_jit: torch.jit.ScriptModule = torch.jit.load(schema_net_cfg["discretization_jit"], map_location=device)
    logger.info("Loaded jit models")
    ingredient_wrapper = utils.IngredientModelWrapper(backbone_jit, discretization_jit)
    # create SchemaNet and matcher
    schema_net = graph.SchemaNet(
        num_vertices=ingredient_wrapper.num_ingredients,
        num_classes=n_classes,
        **schema_net_cfg.get("ir_atlas", dict())
    )
    # initial SchemaNet
    init_fp = schema_net_cfg.get("initial_state_fp", None)
    if init_fp is not None:
        state_fp = torch.load(init_fp, map_location="cpu")
        schema_net.load_state_dict(state_fp)
        logger.info("Loaded from initial SchemaNet")

    matcher = graph.Matcher(
        num_codes=ingredient_wrapper.num_ingredients,
        gnn_cfg=schema_net_cfg["gnn"],
        **schema_net_cfg["matcher"]
    )
    wrapper = graph.SchemaNetPredictor(
        ingredient_wrapper,
        schema_net,
        matcher
    )
    wrapper.to(device)
    logger.info(
        "Built SchemaNet with %d parameters, %d trainable parameters",
        cv_utils.count_parameters(wrapper, include_no_grad=True),
        cv_utils.count_parameters(wrapper, include_no_grad=False)
    )
    params = utils.customs_param_group(
        wrapper.named_parameters(),
        train_cfg["param_groups"],
        train_cfg["drop_remain"]
    )
    optimizer = get_optimizer(params, train_cfg["optimizer"])
    logger.info("Loaded optimizer:\n%s", optimizer)
    lr_scheduler = get_scheduler(optimizer, train_cfg["lr_schedule"])

    loss = get_loss_fn(
        loss_cfg,
        vertex_weights=wrapper.schema_net.vertex_weights,
        edge_weights=wrapper.schema_net.edge_weights
    )
    loss.to(device)

    if distributed:
        wrapper = nn.parallel.DistributedDataParallel(
            wrapper,
            device_ids=[gpu_id]
        )

    evaluator = Evaluation(
        loss_fn=loss,
        val_loader=val_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device
    )

    trainer = SchemaNetTrainer(
        train_cfg=train_cfg,
        log_args=log_args,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        predictor=wrapper,
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
