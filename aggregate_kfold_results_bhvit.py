# aggregate_kfold_results_bhvit.py
#!/usr/bin/env python3
"""
Aggregate 5-fold BHViT runs (trained with main_new.py) to reproduce the
'notebook-style' ensemble outputs using the SAME model construction,
the SAME eval transforms, and compatible artifacts.

Expected layout:
  DATA_ROOT/
    fold0/{train,val,holdout_val,test}/Type_1,...   # created by your exporter
    ...
    fold4/...

  RUNS (args.output_dir from training):
    /path/to/run/fold0/best.pth  (+ optional model_ema, train_args.json)
    ...
    /path/to/run/fold4/best.pth

Outputs (OUT_DIR):
  predictions/
    fold0/{val,holdout_val,holdout_test}_preds.json
    ...
    holdout_val_ensemble.json
    holdout_test_ensemble.json
  metrics/
    fold0/metrics.json
    ...
    ensemble_metrics.json
    summary.json
  figures/
    fold0/{holdout_val_confusion.png, holdout_val_roc.png, ...}
    ...
    {holdout_val_confusion_ens.png, holdout_val_roc_ens.png, ...}
"""

import argparse, json, os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder

# --- import from your repo ---
from models import get_model
from datasets import CervicalCancerDataset, build_transform
from plot import TrainingPlots


# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Aggregate K-Fold BHViT results")
    ap.add_argument(
        "--data-root",
        required=True,
        help="Root with fold0..fold4 (from kfold exporter)",
    )
    ap.add_argument(
        "--fold-runs",
        nargs="+",
        required=True,
        help="List of 5 run dirs (each contains best.pth). "
        "We match each to data-root/<basename> (e.g., 'fold0').",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Where to write aggregated predictions/metrics/figures",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--use-ema-if-available",
        action="store_true",
        help="If checkpoint has model_ema, load those weights.",
    )
    ap.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True for state_dict load (default False).",
    )
    return ap.parse_args()


# ---------------------- Utils ----------------------
@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, y_pred, y_prob"""
    model.eval()
    ys, ps, proba = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        proba.append(prob)
        ps.append(prob.argmax(axis=1))
        ys.append(yb.numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.zeros((0,), dtype=np.int64)
    y_prob = np.concatenate(proba) if proba else np.zeros((0, 0), dtype=np.float32)
    return y_true, y_pred, y_prob


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_args_from_ckpt(ckpt: dict) -> SimpleNamespace:
    """
    Prefer the argparse Namespace saved in checkpoint['args'].
    Fallback to a minimal SimpleNamespace with sensible defaults for eval.
    """
    if "args" in ckpt and ckpt["args"] is not None:
        # already a Namespace (pickled by torch.save)
        ns = ckpt["args"]
        # tiny safety net: make sure a few attributes exist
        defaults = dict(
            input_size=getattr(ns, "input_size", 224),
            data_set=getattr(ns, "data_set", "CERVICAL"),
            aa=getattr(ns, "aa", "noaug"),
            num_workers=getattr(ns, "num_workers", 8),
            pin_mem=getattr(ns, "pin_mem", True),
            batch_size=getattr(ns, "batch_size", 64),
            model=getattr(ns, "model", "deit_base_patch16_224"),
            model_type=getattr(ns, "model_type", ""),
            weight_bits=getattr(ns, "weight_bits", 32),
            input_bits=getattr(ns, "input_bits", 32),
            device=getattr(ns, "device", "cuda"),
            output_dir=getattr(ns, "output_dir", ""),
        )
        for k, v in defaults.items():
            if not hasattr(ns, k):
                setattr(ns, k, v)
        return ns
    # fallback — minimal config for eval
    return SimpleNamespace(
        input_size=224,
        data_set="CERVICAL",
        aa="noaug",
        num_workers=8,
        pin_mem=True,
        batch_size=64,
        model="deit_base_patch16_224",
        model_type="",
        weight_bits=32,
        input_bits=32,
        device="cuda",
        output_dir="",
    )


def load_model(
    best_path: Path, device: torch.device, use_ema: bool, strict: bool
) -> Tuple[nn.Module, SimpleNamespace]:
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    args = load_args_from_ckpt(ckpt)

    # build model EXACTLY like training
    model = get_model(
        args, args.model, args.model_type, args.weight_bits, args.input_bits
    )
    state = None
    if use_ema and ("model_ema" in ckpt):
        # ckpt["model_ema"] is a (possibly nested) state dict
        state = ckpt["model_ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt  # last-ditch — assume pure state_dict

    # non-strict load by default: allows minor head mismatches
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model, args


def build_eval_loader(
    split_dir: Path, args_ns: SimpleNamespace, batch_size: int, num_workers: int
) -> DataLoader:
    """
    Use YOUR eval transform so normalization & resize exactly match training code.
    (datasets.build_transform with is_train=False)
    """
    tfm = build_transform(is_train=False, args=args_ns)
    ds = CervicalCancerDataset(str(split_dir), transform=tfm)
    return (
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=getattr(args_ns, "pin_mem", True),
        ),
        ds,
    )


# ---------------------- Main ----------------------
def main():
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # For ensemble across folds
    holdout_val_probs_stack: List[np.ndarray] = []
    holdout_test_probs_stack: List[np.ndarray] = []
    y_holdout_val_ref, y_holdout_test_ref = None, None

    per_fold_summaries = []

    # derive class names from the first fold's holdout_val
    # (all folds share the same class structure)
    first_fold_name = Path(args.fold_runs[0]).name
    classes = sorted(
        [
            p.name
            for p in (data_root / first_fold_name / "holdout_val").iterdir()
            if p.is_dir()
        ]
    )

    for run_dir in args.fold_runs:
        run_dir = Path(run_dir)
        fold_name = run_dir.name  # must match data_root/<fold_name>
        fold_root = data_root / fold_name
        if not fold_root.exists():
            raise SystemExit(f"[error] data folder missing: {fold_root}")

        best_path = run_dir / "best.pth"
        if not best_path.exists():
            raise SystemExit(f"[error] checkpoint missing: {best_path}")

        print(f"\n=== Fold: {fold_name} ===")
        model, train_args = load_model(
            best_path,
            device,
            use_ema=args.use_ema_if_available,
            strict=args.strict_load,
        )

        # Build loaders for val/holdout_val/test with YOUR eval transform
        dl_val, ds_val = build_eval_loader(
            fold_root / "val", train_args, args.batch_size, args.num_workers
        )
        dl_hv, ds_hv = build_eval_loader(
            fold_root / "holdout_val", train_args, args.batch_size, args.num_workers
        )
        dl_te, ds_te = build_eval_loader(
            fold_root / "test", train_args, args.batch_size, args.num_workers
        )

        # 1) Predictions on all three splits
        y_v, p_v, prob_v = predict(model, dl_val, device)
        y_hv, p_hv, prob_hv = predict(model, dl_hv, device)
        y_te, p_te, prob_te = predict(model, dl_te, device)

        # save per-fold predictions
        pred_dir = out_dir / "predictions" / fold_name
        save_json(
            pred_dir / "val_preds.json",
            {"y_true": y_v.tolist(), "y_pred": p_v.tolist(), "y_prob": prob_v.tolist()},
        )
        save_json(
            pred_dir / "holdout_val_preds.json",
            {
                "y_true": y_hv.tolist(),
                "y_pred": p_hv.tolist(),
                "y_prob": prob_hv.tolist(),
            },
        )
        save_json(
            pred_dir / "holdout_test_preds.json",
            {
                "y_true": y_te.tolist(),
                "y_pred": p_te.tolist(),
                "y_prob": prob_te.tolist(),
            },
        )

        # 2) Per-fold metrics & figures (using your TrainingPlots to match style/files)
        fold_fig_dir = out_dir / "figures" / fold_name
        fold_met_dir = out_dir / "metrics" / fold_name
        tp = TrainingPlots(
            out_dir=fold_fig_dir.parent.parent
            / fold_name,  # put plots/ + metrics/ under OUT_DIR/<fold>
            class_names=getattr(ds_hv, "classes", None),
            args={},
        )  # args dump optional here

        # Confusion + report (val/holdout_val/test)
        tp.save_confusion_and_report(
            model, dl_val, device, normalize="true", file_prefix="val"
        )
        tp.save_confusion_and_report(
            model, dl_hv, device, normalize="true", file_prefix="holdout_val"
        )
        tp.save_confusion_and_report(
            model, dl_te, device, normalize="true", file_prefix="test"
        )

        # Summarize simple accuracies
        acc_val = float((p_v == y_v).mean()) if len(y_v) else float("nan")
        acc_hv = float((p_hv == y_hv).mean()) if len(y_hv) else float("nan")
        acc_te = float((p_te == y_te).mean()) if len(y_te) else float("nan")
        save_json(
            fold_met_dir / "metrics.json",
            {
                "fold": fold_name,
                "val_acc": acc_val,
                "holdout_val_acc": acc_hv,
                "holdout_test_acc": acc_te,
            },
        )

        per_fold_summaries.append(
            {
                "fold": fold_name,
                "val_acc": acc_val,
                "holdout_val_acc": acc_hv,
                "holdout_test_acc": acc_te,
            }
        )

        # for ensemble across folds (only holdouts)
        holdout_val_probs_stack.append(prob_hv)
        holdout_test_probs_stack.append(prob_te)
        if y_holdout_val_ref is None:
            y_holdout_val_ref = y_hv
        if y_holdout_test_ref is None:
            y_holdout_test_ref = y_te

    # ---------------- Ensemble across folds (mean probabilities) ----------------
    hv_mean = np.mean(np.stack(holdout_val_probs_stack, axis=0), axis=0)  # (N_val, C)
    te_mean = np.mean(np.stack(holdout_test_probs_stack, axis=0), axis=0)  # (N_test, C)
    hv_pred = hv_mean.argmax(axis=1)
    te_pred = te_mean.argmax(axis=1)

    # Save ensemble predictions
    ens_pred_dir = out_dir / "predictions"
    save_json(
        ens_pred_dir / "holdout_val_ensemble.json",
        {
            "y_true": y_holdout_val_ref.tolist(),
            "y_pred": hv_pred.tolist(),
            "y_prob": hv_mean.tolist(),
        },
    )
    save_json(
        ens_pred_dir / "holdout_test_ensemble.json",
        {
            "y_true": y_holdout_test_ref.tolist(),
            "y_pred": te_pred.tolist(),
            "y_prob": te_mean.tolist(),
        },
    )

    # Ensemble metrics
    acc_hv_ens = float((hv_pred == y_holdout_val_ref).mean())
    acc_te_ens = float((te_pred == y_holdout_test_ref).mean())
    save_json(
        out_dir / "metrics" / "ensemble_metrics.json",
        {
            "holdout_val_acc": acc_hv_ens,
            "holdout_test_acc": acc_te_ens,
            "per_fold": per_fold_summaries,
        },
    )

    # Ensemble figures using your plot class (so filenames/layout match)
    # We just create a lightweight DataLoader from numpy for plotting convenience;
    # instead, we’ll reuse TrainingPlots helpers by building fake loaders through an ImageFolder with indices.
    # Simpler: build loaders on holdout splits again and write figures using a tiny helper model wrapper that returns our cached probs.
    # But TrainingPlots already knows how to compute probs; so we’ll quickly dump confusion & ROC using numpy here.

    # Confusion + ROC for ensemble
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
    except Exception as e:
        print("[warn] sklearn not found; ensemble figures will be skipped.")
        confusion_matrix = None

    if confusion_matrix is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_confusion(
            cm: np.ndarray, class_names: List[str], out_png: Path, normalize=True
        ):
            out_png.parent.mkdir(parents=True, exist_ok=True)
            cm = cm.astype(np.float32)
            if normalize:
                cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            im = ax.imshow(cm, interpolation="nearest", aspect="auto")
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(len(class_names)),
                yticks=np.arange(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel="True",
                xlabel="Predicted",
                title="Confusion (normalized)" if normalize else "Confusion",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{cm[i,j]:.2f}" if normalize else int(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
            fig.tight_layout()
            plt.savefig(out_png)
            plt.close(fig)

        # class names from filesystem
        class_names = classes

        cm_hv = confusion_matrix(
            y_holdout_val_ref, hv_pred, labels=list(range(len(class_names)))
        )
        cm_te = confusion_matrix(
            y_holdout_test_ref, te_pred, labels=list(range(len(class_names)))
        )
        plot_confusion(
            cm_hv, class_names, out_dir / "figures" / "holdout_val_confusion_ens.png"
        )
        plot_confusion(
            cm_te, class_names, out_dir / "figures" / "holdout_test_confusion_ens.png"
        )

        # ROC (OvR) if there is at least one positive per class
        def plot_roc(y_true, y_prob, class_names, out_png: Path):
            Y = label_binarize(y_true, classes=list(range(len(class_names))))
            valid = []
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(class_names)):
                pos = Y[:, i].sum()
                neg = (1 - Y[:, i]).sum()
                if pos > 0 and neg > 0:
                    fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    valid.append(i)
            if not valid:
                return
            all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in valid:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= len(valid)
            fig = plt.figure(figsize=(7, 6), dpi=150)
            for i in valid:
                plt.plot(
                    fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})"
                )
            plt.plot([0, 1], [0, 1], linestyle=":")
            plt.plot(all_fpr, mean_tpr, linestyle="--", label="macro")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC (ensemble)")
            plt.legend(loc="lower right")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close(fig)

        plot_roc(
            y_holdout_val_ref,
            hv_mean,
            class_names,
            out_dir / "figures" / "holdout_val_roc_ens.png",
        )
        plot_roc(
            y_holdout_test_ref,
            te_mean,
            class_names,
            out_dir / "figures" / "holdout_test_roc_ens.png",
        )

    # Final summary
    save_json(
        out_dir / "metrics" / "summary.json",
        {
            "folds": per_fold_summaries,
            "ensemble": {"holdout_val_acc": acc_hv_ens, "holdout_test_acc": acc_te_ens},
        },
    )

    print("\n[done] wrote aggregated results to", str(out_dir.resolve()))


if __name__ == "__main__":
    main()
