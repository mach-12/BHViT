import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from timm.models import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset, build_transform, CervicalCancerDataset
from engine import train_one_epoch, evaluate, train_one_epoch_L1
from samplers import RASampler
from models import SyncBatchNormT, get_model
import utils
from losses import DistributionLoss
from plot import TrainingPlots, ViTAttentionPlots
import os

from collections import Counter


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--model-type", default="", type=str, help="Type of the model")
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--avg-res3", action="store_true", help="Avg pooling layer residual at FFN"
    )
    parser.add_argument(
        "--avg-res5", action="store_true", help="Avg pooling layer residual at FFN"
    )
    parser.add_argument(
        "--shift3", action="store_true", help="Avg pooling layer residual at FFN"
    )
    parser.add_argument(
        "--shift5", action="store_true", help="Avg pooling layer residual at FFN"
    )
    parser.add_argument(
        "--replace-ln-bn", action="store_true", help="replace LayerNorm with BatchNorm"
    )
    parser.add_argument(
        "--disable-layerscale", action="store_true", help="Disable LayerScale"
    )
    parser.add_argument(
        "--enable-cls-token",
        action="store_true",
        help="Use cls token instead of average pooling",
    )

    parser.add_argument(
        "--weight-bits", default=32, type=int, help="Bitwidth of weights"
    )
    parser.add_argument(
        "--input-bits", default=32, type=int, help="Bitwidth of activations"
    )
    parser.add_argument(
        "--some-fp",
        action="store_true",
        help="Patch embedding layers in the middle are in full-precision",
    )
    parser.add_argument(
        "--gsb", action="store_true", help="improvement of attention matrix"
    )
    parser.add_argument("--recu", action="store_true", help="improvement of weight")
    parser.add_argument(
        "--regularization_loss", action="store_true", help="improvement of weight"
    )
    parser.add_argument(
        "--teacher-model",
        default="",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--teacher-model-type", default="", type=str, help="Type of the model"
    )
    parser.add_argument(
        "--teacher-model-file", default="", type=str, help="Type of the model"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--drop-block",
        type=float,
        default=None,
        metavar="PCT",
        help="Drop block rate (default: None)",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-8,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="/data/huikong/dataset/imagenet-1k/ILSVRC2012",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "INAT", "INAT19", "CERVICAL"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output-dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--current-best-model",
        default="",
        help="load current best model for initialization or to evaluate current best model",
    )
    parser.add_argument(
        "--no-strict-load",
        action="store_true",
        help="less strict when loading the current best model",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # interpretability via attention rollouts
    parser.add_argument(
        "--attn-viz",
        action="store_true",
        help="Enable attention rollout visualizations after evaluation.",
    )
    parser.add_argument(
        "--attn-max-images",
        default=8,
        type=int,
        help="Maximum number of images to visualize for attention rollout.",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=8,
        help="How many validation images to visuaglize",
    )
    parser.add_argument(
        "--viz-every",
        type=int,
        default=0,
        help="If >0, also visualize every N epochs (in addition to final epoch)",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on validation/test sets without training",
    )

    return parser


def main(args):
    import numpy as np

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    targets = [s[1] for s in dataset_train.samples]
    counts = Counter(targets)  # e.g., {0: n0, 1: n1, 2: n2}
    num_classes = args.nb_classes

    # inverse frequency (or use effective number of samples)
    class_weights = torch.tensor(
        [len(targets) / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float,
        device=device,
    )

    # optional test split (only for cervical cancer if /test exists)
    data_loader_test = None
    dataset_test = None
    test_dir = Path(args.data_path) / "test"
    if args.data_set == "CERVICAL" and test_dir.exists():
        dataset_test = CervicalCancerDataset(
            str(test_dir), transform=build_transform(is_train=False, args=args)
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    if args.distributed and utils.get_world_size() > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True,
        )
    else:
        # Balanced sampling (fix)
        import numpy as np

        targets = np.array([s[1] for s in dataset_train.samples])
        class_count = np.bincount(targets, minlength=args.nb_classes).astype(float)
        inv_freq = 1.0 / np.maximum(class_count, 1.0)
        sample_weights = inv_freq[targets]
        sampler_train = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    # model = create_model(
    #     args.model,
    #     pretrained=False,
    #     num_classes=args.nb_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    # )
    model = get_model(
        args, args.model, args.model_type, args.weight_bits, args.input_bits
    )
    # print(model)
    # TODO: finetuning

    model.to(device)

    if args.replace_ln_bn:
        model = SyncBatchNormT.convert_sync_batchnorm(model)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Use weighted CE; turn OFF label smoothing when using class weights
    criterion, criterion2 = None, None

    if args.teacher_model:
        # Distillation: teacher logits + weighted CE for hard targets
        criterion = DistributionLoss()
        criterion2 = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif args.mixup > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # Default: weighted CE (bug fix here)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    output_dir = Path(args.output_dir)

    # create plotter (rank-0 only)
    plotter = None
    if args.output_dir and utils.is_main_process():
        class_names = None
        if hasattr(dataset_val, "classes"):
            class_names = dataset_val.classes
        elif hasattr(dataset_val, "class_to_idx"):
            class_names = list(dataset_val.class_to_idx.keys())

        plotter = TrainingPlots(output_dir, class_names=class_names, args=args)

    attn_viz = None
    if args.output_dir and utils.is_main_process():
        attn_viz = ViTAttentionPlots(
            output_dir, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    attn_max_images = None
    if args.attn_max_images:
        attn_max_images = args.attn_max_images

    max_accuracy = 0.0
    if args.current_best_model:
        best_model = torch.load(
            args.current_best_model, map_location="cpu", weights_only=False
        )
        if "model" in best_model:
            best_model = best_model["model"]
        state_dict = model_without_ddp.state_dict()
        new_best_model = {}
        for k, v in best_model.items():
            if k in state_dict:
                if v.size() == state_dict[k].size():
                    new_best_model[k] = v
        state_dict.update(new_best_model)

        #      model_without_ddp.load_state_dict(best_model, strict=(not args.no_strict_load))
        model_without_ddp.load_state_dict(state_dict, strict=False)
        if args.resume:
            val_stats = evaluate(data_loader_val, model, device, split_name="Val")
            print(
                f"Accuracy of the best network on the {len(dataset_val)} val images: {val_stats['acc1']:.1f}%"
            )
            max_accuracy = val_stats["acc1"]

            if plotter is not None:
                plotter.update_epoch(train_stats, val_stats, epoch, test_stats)

            # if attn_viz is not None and args.viz_attn and args.viz_every > 0 and (epoch + 1) % args.viz_every == 0:
            #     attn_viz.visualize_from_loader(model, data_loader_val, device, max_images=args.viz_samples)

    teacher_model = None
    # regnety_160,deit_small_patch16_224
    if args.teacher_model:
        teacher_model = create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=1000,
        )
        teacher_model.to(device)
        teacher_model.eval()
        teacher_model_without_ddp = teacher_model
        if args.distributed:
            teacher_model = torch.nn.parallel.DistributedDataParallel(
                teacher_model, device_ids=[args.gpu]
            )
            teacher_model_without_ddp = teacher_model.module

        # best_teacher_model = torch.load(args.teacher_model_file, map_location='cpu')
        # if 'model' in best_teacher_model:
        #     best_teacher_model = best_teacher_model['model']
        # teacher_model_without_ddp.load_state_dict(best_teacher_model)
        # test_stats = evaluate(data_loader_val, teacher_model, device)
        # print(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True, weights_only=False
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        lr_scheduler.step(args.start_epoch - 1)

        val_stats = evaluate(data_loader_val, model, device, split_name="Val")
        print(
            f"Accuracy of the checkpoint network on the {len(dataset_val)} val images: {val_stats['acc1']:.1f}%"
        )

    print("Start training" if not args.eval_only else "Start evaluation")

    if args.eval_only:
        # Evaluation-only mode
        print("Running evaluation only...")

        # Validation
        val_stats = evaluate(data_loader_val, model, device, split_name="Val")
        print(f"Accuracy on {len(dataset_val)} val images: {val_stats['acc1']:.1f}%")

        # Optional Test (if available)
        test_stats = None
        if data_loader_test is not None:
            test_stats = evaluate(data_loader_test, model, device, split_name="Test")
            print(
                f"Accuracy on {len(dataset_test)} test images: {test_stats['acc1']:.1f}%"
            )

        # Generate plots and reports
        if plotter is not None:
            # Validation artifacts
            plotter.save_confusion_and_report(
                model, data_loader_val, device, normalize="true", file_prefix="val"
            )

            # Test artifacts (only if you have a test loader)
            if data_loader_test is not None:
                plotter.save_confusion_and_report(
                    model,
                    data_loader_test,
                    device,
                    normalize="true",
                    file_prefix="test",
                )

            plotter.save_summary(max_accuracy=val_stats["acc1"], total_epochs=0)

        # Attention visualizations
        if attn_viz is not None and args.attn_viz:
            # Use test set if available, otherwise validation set
            viz_loader = (
                data_loader_test if data_loader_test is not None else data_loader_val
            )
            attn_viz.visualize_from_loader(
                model, viz_loader, device, max_images=attn_max_images
            )

        print(f"Evaluation completed. Final accuracy: {val_stats['acc1']:.2f}%")
    else:
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            if args.regularization_loss:
                train_stats = train_one_epoch_L1(
                    model,
                    teacher_model,
                    criterion,
                    criterion2,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    args.clip_grad,
                    model_ema,
                    mixup_fn,
                )
            else:
                train_stats = train_one_epoch(
                    model,
                    teacher_model,
                    criterion,
                    criterion2,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    args.clip_grad,
                    model_ema,
                    mixup_fn,
                )

            lr_scheduler.step(epoch)
            if args.output_dir:
                checkpoint_paths = [output_dir / "checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:
                    checkpoint_dict = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    }
                    if args.model_ema:
                        checkpoint_dict["model_ema"] = get_state_dict(model_ema)
                    utils.save_on_master(checkpoint_dict, checkpoint_path)

            # Validation
            val_stats = evaluate(data_loader_val, model, device, split_name="Val")
            print(
                f"Accuracy on {len(dataset_val)} val images: {val_stats['acc1']:.1f}%"
            )

            # Optional Test (if available)
            test_stats = None
            if data_loader_test is not None:
                test_stats = evaluate(
                    data_loader_test, model, device, split_name="Test"
                )
                print(
                    f"Accuracy on {len(dataset_test)} test images: {test_stats['acc1']:.1f}%"
                )

            # Track best on validation
            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                if args.output_dir:
                    best_paths = [output_dir / "best.pth"]
                    for best_path in best_paths:
                        best_dict = {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "scaler": loss_scaler.state_dict(),
                            "args": args,
                        }
                        if args.model_ema:
                            best_dict["model_ema"] = get_state_dict(model_ema)
                        utils.save_on_master(best_dict, best_path)

            print(f"Max accuracy: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                **(
                    {f"test_{k}": v for k, v in test_stats.items()}
                    if test_stats is not None
                    else {}
                ),
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # For logging every epoch
            if plotter is not None:
                plotter.update_epoch(train_stats, val_stats, epoch, test_stats)

            # if plotter is not None and (epoch + 1) % args.epochs == 0:
            #     plotter.save_confusion_and_report(model, data_loader_val, device, normalize="true")

        # final confusion matrix + classification report + summary
        if plotter is not None:
            # uses the current (final) model weights
            # Validation artifacts
            plotter.save_confusion_and_report(
                model, data_loader_val, device, normalize="true", file_prefix="val"
            )

            # Test artifacts (only if you have a test loader)
            if data_loader_test is not None:
                plotter.save_confusion_and_report(
                    model,
                    data_loader_test,
                    device,
                    normalize="true",
                    file_prefix="test",
                )

            plotter.save_summary(max_accuracy=max_accuracy, total_epochs=args.epochs)

            # Final interpretability set on validation images
            if attn_viz is not None and args.attn_viz:
                viz_loader = (
                    data_loader_test
                    if data_loader_test is not None
                    else data_loader_val
                )
                attn_viz.visualize_from_loader(
                    model, viz_loader, device, max_images=attn_max_images
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
