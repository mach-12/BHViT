#!/usr/bin/env python3
"""
Aggregate BHViT K-Fold runs to match the reference notebook:
- Per-fold predictions on fold-val, holdout_val, test
- Per-fold metrics/figures
- Cross-fold ensemble (mean prob) on holdout_val and test
- Final summary JSON

Usage example:
python aggregate_kfold_results.py \
  --data-root /data/cervical_kfolds \
  --fold-dirs /runs/bhvit/fold0 /runs/bhvit/fold1 /runs/bhvit/fold2 /runs/bhvit/fold3 /runs/bhvit/fold4 \
  --out-dir /runs/bhvit_kfold_summary
"""

import argparse, json, os, math
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# ---------- Imports from repo (must be available in PYTHONPATH) ----------
from models import get_model as bhvit_get_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        required=True,
        help="Output root from kfold_export_imagefolder.py",
    )
    ap.add_argument(
        "--fold-dirs",
        nargs="+",
        required=True,
        help="Paths to fold run dirs (with best.pth)",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Where to write the aggregated outputs"
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# -------------------- Utils --------------------
def eval_transform(
    image_size=224, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    size = int((256 / 224) * image_size)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_targets.append(yb.numpy())
    probs = (
        np.concatenate(all_probs) if all_probs else np.zeros((0, 3), dtype=np.float32)
    )
    preds = np.concatenate(all_preds) if all_preds else np.zeros((0,), dtype=np.int64)
    targs = (
        np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=np.int64)
    )
    return targs, preds, probs


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def plot_confusion(
    cm: np.ndarray, class_names: List[str], out_png: Path, normalize=True
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = cm.astype(np.float32)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion" + (" (norm)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_multiclass_roc(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out_png: Path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize

    C = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(C)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(C):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(
        np.concatenate(
            [f.values()[0] if isinstance(f, dict) else f for f in fpr.values()]
        )
    )
    plt.figure(figsize=(7, 6))
    for i in range(C):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc="lower right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def load_model_from_best(best_path: Path, num_classes=3, device="cuda"):
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    # Recover args if present; fall back to defaults
    args = ckpt.get("args", None)
    if args is not None and hasattr(args, "model"):
        model = bhvit_get_model(
            args,
            args.model,
            getattr(args, "model_type", ""),
            getattr(args, "weight_bits", 32),
            getattr(args, "input_bits", 32),
        )
    else:
        # Minimal fallback: DeiT-B/16 head for 3 classes
        from timm import create_model

        model = create_model(
            "deit_base_patch16_224", pretrained=False, num_classes=num_classes
        )
    state = ckpt["model"] if "model" in ckpt else ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Derive class names from any fold's 'train' set
    any_fold = Path(args.fold_dirs[0])
    fold0_data = (
        Path(args.data_root) / any_fold.name
    )  # try matching by name first (e.g., fold0)
    if not fold0_data.exists():
        # fallback: assume folder names are fold0..K under data-root
        fold0_data = Path(args.data_root) / "fold0"
    classes = sorted([p.name for p in (fold0_data / "train").iterdir() if p.is_dir()])

    tfm = eval_transform(224)
    # Containers for ensemble across folds
    all_holdout_val_probs, all_holdout_test_probs = [], []
    ids_holdout_val, y_holdout_val = None, None
    ids_holdout_test, y_holdout_test = None, None

    per_fold_summaries = []

    for fold_dir in args.fold_dirs:
        fold_dir = Path(fold_dir)
        fold_name = fold_dir.name  # expect foldK
        data_fold = Path(args.data_root) / fold_name
        if not data_fold.exists():
            raise SystemExit(f"[error] data root missing {data_fold}")

        # Datasets/loaders
        ds_foldval = ImageFolder(data_fold / "val", transform=tfm)
        ds_holdout_val = ImageFolder(data_fold / "holdout_val", transform=tfm)
        ds_test = ImageFolder(data_fold / "test", transform=tfm)

        dl_foldval = DataLoader(
            ds_foldval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dl_holdout_val = DataLoader(
            ds_holdout_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Load best model
        best_path = fold_dir / "best.pth"
        if not best_path.exists():
            raise SystemExit(f"[error] missing {best_path}")
        model = load_model_from_best(best_path, num_classes=len(classes), device=device)

        # Inference
        y_fv, p_fv, prob_fv = predict(model, dl_foldval, device)
        y_hv, p_hv, prob_hv = predict(model, dl_holdout_val, device)
        y_te, p_te, prob_te = predict(model, dl_test, device)

        # Save preds
        pred_dir = out_dir / "predictions" / fold_name
        save_json(
            pred_dir / f"{fold_name}_foldval_preds.json",
            {
                "y_true": y_fv.tolist(),
                "y_pred": p_fv.tolist(),
                "y_prob": prob_fv.tolist(),
            },
        )
        save_json(
            pred_dir / f"{fold_name}_holdout_val_preds.json",
            {
                "y_true": y_hv.tolist(),
                "y_pred": p_hv.tolist(),
                "y_prob": prob_hv.tolist(),
            },
        )
        save_json(
            pred_dir / f"{fold_name}_holdout_test_preds.json",
            {
                "y_true": y_te.tolist(),
                "y_pred": p_te.tolist(),
                "y_prob": prob_te.tolist(),
            },
        )

        # Metrics & figures per fold (holdout_val is the key one)
        fig_dir = out_dir / "figures" / fold_name
        met_dir = out_dir / "metrics" / fold_name

        acc_fv = float(accuracy_score(y_fv, p_fv)) if len(y_fv) else float("nan")
        acc_hv = float(accuracy_score(y_hv, p_hv)) if len(y_hv) else float("nan")
        acc_te = float(accuracy_score(y_te, p_te)) if len(y_te) else float("nan")

        save_json(
            met_dir / "metrics.json",
            {
                "fold": fold_name,
                "foldval_acc": acc_fv,
                "holdout_val_acc": acc_hv,
                "holdout_test_acc": acc_te,
                "holdout_val_report": classification_report(
                    y_hv, p_hv, output_dict=True, zero_division=0
                ),
                "holdout_test_report": classification_report(
                    y_te, p_te, output_dict=True, zero_division=0
                ),
            },
        )

        cm_hv = confusion_matrix(y_hv, p_hv, labels=list(range(len(classes))))
        cm_te = confusion_matrix(y_te, p_te, labels=list(range(len(classes))))
        plot_confusion(cm_hv, classes, fig_dir / "holdout_val_confusion.png")
        plot_confusion(cm_te, classes, fig_dir / "holdout_test_confusion.png")
        if len(y_hv) and prob_hv.shape[1] == len(classes):
            try:
                plot_multiclass_roc(
                    y_hv, prob_hv, classes, fig_dir / "holdout_val_roc.png"
                )
            except Exception:
                pass
        if len(y_te) and prob_te.shape[1] == len(classes):
            try:
                plot_multiclass_roc(
                    y_te, prob_te, classes, fig_dir / "holdout_test_roc.png"
                )
            except Exception:
                pass

        per_fold_summaries.append(
            {
                "fold": fold_name,
                "foldval_acc": acc_fv,
                "holdout_val_acc": acc_hv,
                "holdout_test_acc": acc_te,
            }
        )

        # for ensemble across folds (holdouts only)
        all_holdout_val_probs.append(prob_hv)
        all_holdout_test_probs.append(prob_te)
        # stash y once
        if y_holdout_val is None:
            y_holdout_val = y_hv
        if y_holdout_test is None:
            y_holdout_test = y_te

    # Ensemble across folds (mean probs)
    hv_stack = np.stack(all_holdout_val_probs, axis=0)  # (K, N_val, C)
    te_stack = np.stack(all_holdout_test_probs, axis=0)  # (K, N_test, C)
    hv_mean = hv_stack.mean(axis=0)
    te_mean = te_stack.mean(axis=0)
    hv_pred = hv_mean.argmax(axis=1)
    te_pred = te_mean.argmax(axis=1)

    ens_dir = out_dir / "predictions"
    save_json(
        ens_dir / "holdout_val_ensemble.json",
        {
            "y_true": y_holdout_val.tolist(),
            "y_pred": hv_pred.tolist(),
            "y_prob": hv_mean.tolist(),
        },
    )
    save_json(
        ens_dir / "holdout_test_ensemble.json",
        {
            "y_true": y_holdout_test.tolist(),
            "y_pred": te_pred.tolist(),
            "y_prob": te_mean.tolist(),
        },
    )

    # Ensemble metrics
    met_root = out_dir / "metrics"
    acc_hv_ens = float(accuracy_score(y_holdout_val, hv_pred))
    acc_te_ens = float(accuracy_score(y_holdout_test, te_pred))
    save_json(
        met_root / "ensemble_metrics.json",
        {
            "holdout_val_acc": acc_hv_ens,
            "holdout_test_acc": acc_te_ens,
            "per_fold": per_fold_summaries,
        },
    )

    # Ensemble plots
    fig_root = out_dir / "figures"
    plot_confusion(
        confusion_matrix(y_holdout_val, hv_pred, labels=list(range(len(classes)))),
        classes,
        fig_root / "holdout_val_confusion_ens.png",
    )
    plot_confusion(
        confusion_matrix(y_holdout_test, te_pred, labels=list(range(len(classes)))),
        classes,
        fig_root / "holdout_test_confusion_ens.png",
    )
    try:
        plot_multiclass_roc(
            y_holdout_val, hv_mean, classes, fig_root / "holdout_val_roc_ens.png"
        )
    except Exception:
        pass
    try:
        plot_multiclass_roc(
            y_holdout_test, te_mean, classes, fig_root / "holdout_test_roc_ens.png"
        )
    except Exception:
        pass

    # Final summary
    save_json(
        met_root / "summary.json",
        {
            "classes": classes,
            "folds": per_fold_summaries,
            "ensemble": {"holdout_val_acc": acc_hv_ens, "holdout_test_acc": acc_te_ens},
        },
    )

    print(f"[done] wrote aggregated results to {out_dir}")


if __name__ == "__main__":
    main()
