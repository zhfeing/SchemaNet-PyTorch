import collections
from typing import Any, Dict, Tuple, List
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv_lib.distributed.utils as dist_utils
import cv_lib.metrics as metrics

from schema_inference.loss import Loss
from schema_inference.utils import move_data_to_device
from discretization import VisualWordEncoder


class IncEvaluation:
    """
    Distributed classification evaluator
    """
    def __init__(
        self,
        loss_fn: Loss,
        base_val_loader: DataLoader,
        base_n_classes: List[int],
        inc_val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device,
        top_k: Tuple[int] = (1,)
    ):
        self.main_process = dist_utils.is_main_process()
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self.base_val_loader = base_val_loader
        self.base_n_classes = base_n_classes
        self.inc_val_loader = inc_val_loader
        self.device = device
        self.top_k = top_k

    def get_loss(self, **loss_kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict: Dict[str, torch.Tensor] = self.loss_fn(**loss_kwargs)
        weighted_losses: Dict[str, torch.Tensor] = dict()
        for k, loss in loss_dict.items():
            k_prefix = k.split(".")[0]
            if k_prefix in self.loss_weights:
                weighted_losses[k] = loss * self.loss_weights[k_prefix]
        loss = sum(weighted_losses.values())
        loss = loss.detach()
        return loss, loss_dict

    def __call__(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Return:
            dictionary:
            {
                loss:
                loss_dict:
                performance:
            }
        """
        model.eval()
        self.loss_fn.eval()

        loss_meter = metrics.AverageMeter()
        loss_dict_meter = metrics.DictAverageMeter()
        # 0: total, 1: inc_val, 2 ~ 1 + #base tasks: base_vals
        acc_meter = [metrics.DictAverageMeter() for _ in range(2 + len(self.base_n_classes))]
        # only show in main process
        # tqdm_shower = None
        # if self.main_process:
        #     tqdm_shower = tqdm.tqdm(total=len(self.base_val_loader), desc="Val Batch")

        with torch.no_grad():
            for i, val_loader in enumerate(self.base_val_loader):
                for samples, target in val_loader:
                    bs = samples.shape[0]
                    target["label"] = target["label"] + sum(self.base_n_classes[:i])
                    samples, target = move_data_to_device(samples, target, self.device)
                    output = model(samples, task=i)
                    # calculate loss
                    loss, loss_dict = self.get_loss(output=output, target=target)
                    loss_meter.update(loss, n=bs)
                    loss_dict_meter.update(loss_dict, n=bs)
                    # calculate acc
                    acc_top_k = metrics.accuracy(output["pred"], target["label"], self.top_k)
                    acc_top_k = {k: acc for k, acc in zip(self.top_k, acc_top_k)}
                    acc_meter[0].update(acc_top_k, n=bs)
                    acc_meter[i + 2].update(acc_top_k, n=bs)

            for samples, target in self.inc_val_loader:
                bs = samples.shape[0]
                target["label"] = target["label"] + sum(self.base_n_classes)
                samples, target = move_data_to_device(samples, target, self.device)
                output = model(samples, task=len(self.base_n_classes))
                # calculate loss
                loss, loss_dict = self.get_loss(output=output, target=target)
                loss_meter.update(loss, n=bs)
                loss_dict_meter.update(loss_dict, n=bs)
                # calculate acc
                acc_top_k = metrics.accuracy(output["pred"], target["label"], self.top_k)
                acc_top_k = {k: acc for k, acc in zip(self.top_k, acc_top_k)}
                acc_meter[0].update(acc_top_k, n=bs)
                acc_meter[1].update(acc_top_k, n=bs)
        dist_utils.barrier()

        # accumulate
        loss_meter.accumulate()
        loss_dict_meter.accumulate()
        for meter in acc_meter:
            meter.accumulate()
        loss_meter.sync()
        loss_dict_meter.sync()
        for meter in acc_meter:
            meter.sync()

        for i in range(len(acc_meter)):
            acc_meter[i] = acc_meter[i].value()

        ret = dict(
            loss=loss_meter.value(),
            loss_dict=loss_dict_meter.value(),
            acc=acc_meter
        )
        return ret


class CBEvaluation(IncEvaluation):
    """
    Distributed KG classification evaluator
    """
    def __init__(
        self,
        loss_fn: Loss,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device
    ):
        super().__init__(
            loss_fn=loss_fn,
            val_loader=val_loader,
            loss_weights=loss_weights,
            device=device
        )

    def __call__(
        self,
        model: nn.Module,
        mid_encoder: VisualWordEncoder,
        calculate_origin: bool = False
    ) -> Dict[str, Any]:
        """
        Return:
            dictionary:
            {
                loss:
                loss_dict:
                performance:
            }
        """
        model.eval()
        self.loss_fn.eval()

        loss_meter = metrics.AverageMeter()
        loss_dict_meter = metrics.DictAverageMeter()
        acc_meter = metrics.DictAverageMeter()
        # only show in main process
        tqdm_shower = None
        if self.main_process:
            tqdm_shower = tqdm.tqdm(total=len(self.val_loader), desc="Val Batch")

        with torch.no_grad():
            for samples, target in self.val_loader:
                bs = samples.shape[0]
                samples, target = move_data_to_device(samples, target, self.device)
                mid_encoder.discretization.deactivate()
                output_origin = model(samples)
                mid_encoder.discretization.activate()
                output = model(samples)
                # calculate loss
                loss, loss_dict = self.get_loss(
                    output=output,
                    output_origin=output_origin,
                    target=target
                )
                loss_meter.update(loss, n=bs)
                loss_dict_meter.update(loss_dict, n=bs)
                # calculate acc
                acc_dict = cb_acc(
                    pred=output["pred"],
                    origin_pred=output_origin["pred"],
                    gt=target["label"],
                    calculate_origin=calculate_origin
                )
                acc_meter.update(acc_dict, n=bs)
                # update tqdm
                if self.main_process:
                    tqdm_shower.update()
        if self.main_process:
            tqdm_shower.close()
        dist_utils.barrier()

        # accumulate
        loss_meter.accumulate()
        loss_dict_meter.accumulate()
        acc_meter.accumulate()
        loss_meter.sync()
        loss_dict_meter.sync()
        acc_meter.sync()

        ret = dict(
            loss=loss_meter.value(),
            loss_dict=loss_dict_meter.value(),
            acc=acc_meter.value()
        )
        return ret


def cb_acc(
    pred: torch.Tensor,
    origin_pred: torch.Tensor,
    gt: torch.LongTensor,
    calculate_origin: bool
) -> Dict[str, torch.Tensor]:
    bs = gt.shape[0]
    acc = collections.OrderedDict()

    origin_pred = origin_pred.argmax(1)
    pred = pred.argmax(1)

    # kg: correct
    acc["acc"] = torch.sum(pred == gt) / bs
    # kg match: True
    acc["acc_model"] = torch.sum(origin_pred == pred) / bs
    if calculate_origin:
        acc["acc_origin"] = torch.sum(origin_pred == gt) / bs
    return acc

