# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Optional

import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def _hard_targets(targets: torch.Tensor) -> torch.Tensor:
    # works for both hard (N,) and soft/one-hot (N, C) labels (e.g., mixup)
    return targets.argmax(dim=1) if targets.ndim == 2 else targets


def train_mix_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    base = SoftTargetCrossEntropy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "base_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "distillation_loss", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    metric_logger.add_meter(
        "acc1", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )
    metric_logger.add_meter(
        "acc5", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            outputs = model(samples)

            hard_t = _hard_targets(targets)
            acc1, acc5 = accuracy(outputs.logits, hard_t, topk=(1, 5))

            base_loss = base(outputs.logits, targets)
            with torch.no_grad():
                teacher_outputs = teacher_model(samples)
                distillation_loss = criterion(outputs.logits, teacher_outputs)
        loss = base_loss * (1 - 0.9) + distillation_loss * 0.9
        loss_value = loss.item()
        base_loss_value = base_loss.item()
        distillation_loss_value = distillation_loss.item()
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(base_loss=base_loss_value)
        metric_logger.update(distillation_loss=distillation_loss_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_L1(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "acc1", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )
    metric_logger.add_meter(
        "acc5", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            hard_t = _hard_targets(targets)
            acc1, acc5 = accuracy(outputs.logits, hard_t, topk=(1, 5))

            regularization_loss = 0
            n = 0
            for name, parameters in model.named_parameters():
                regular = (
                    "cov1.weight" in name
                    or "cov2.weight" in name
                    or "cov3.weight" in name
                    or "dense.weight" in name
                    or "query.weight" in name
                    or "key.weight" in name
                    or "value.weight" in name
                )
                if regular:
                    n = n + 1
                    regularization_loss += (
                        torch.sum(torch.abs(torch.abs(parameters) - 1.0))
                        / parameters.numel()
                    )
            if n != 0:
                regularization_loss = regularization_loss / n
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(samples)
                loss1 = criterion(outputs.logits, teacher_outputs)
                loss = 0.9 * loss1 + 0.1 * regularization_loss

            else:
                loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch2(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "acc1", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )
    metric_logger.add_meter(
        "acc5", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            hard_t = _hard_targets(targets)
            acc1, acc5 = accuracy(outputs.logits, hard_t, topk=(1, 5))
            loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    criterion2: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "acc1", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )
    metric_logger.add_meter(
        "acc5", utils.SmoothedValue(window_size=1, fmt="{value:.3f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            outputs = model(samples)
            hard_t = _hard_targets(targets)
            acc1, acc5 = accuracy(outputs.logits, hard_t, topk=(1, 5))
            if teacher_model is not None and mixup_fn is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(samples)
                loss1 = criterion(outputs.logits, teacher_outputs)
                loss2 = criterion2(outputs.logits, targets)
                loss = 0.9 * loss1 + 0.1 * loss2

            elif teacher_model is not None and mixup_fn is None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(samples)
                loss = criterion(outputs.logits, teacher_outputs)

            else:
                loss = criterion(outputs.logits, targets)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, split_name="val"):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{split_name}:'


    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.logits, target)

        acc1, acc5 = accuracy(output.logits, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print('* [{}] Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(split_name, top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
