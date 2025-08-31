# aggregate_kfold_results_bhvit.py
#!/usr/bin/env python3
import argparse, json, os, re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_model
from datasets import CervicalCancerDataset, build_transform
from plot import TrainingPlots

# ---------------------- CLI ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Aggregate 5-fold BHViT results")
    ap.add_argument("--data-root", required=True,
                    help="Root that contains fold0..fold4 (and optionally folds.json, kfold_manifest.json)")
    ap.add_argument("--fold-runs", nargs="+", required=True,
                    help="List of 5 run dirs each containing best.pth (e.g., ../outputs/bhvittiny_0 ... _4)")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write aggregated predictions/metrics/figures")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--use-ema-if-available", action="store_true",
                    help="If checkpoint has model_ema, prefer those weights.")
    ap.add_argument("--strict-load", action="store_true",
                    help="Use strict=True for state_dict load (default False).")
    return ap.parse_args()

# ---------------------- Helpers ----------------------
def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def load_args_from_ckpt(ckpt: dict) -> SimpleNamespace:
    if "args" in ckpt and ckpt["args"] is not None:
        ns = ckpt["args"]
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
    return SimpleNamespace(
        input_size=224, data_set="CERVICAL", aa="noaug",
        num_workers=8, pin_mem=True, batch_size=64,
        model="deit_base_patch16_224", model_type="",
        weight_bits=32, input_bits=32, device="cuda", output_dir=""
    )

def load_model(best_path: Path, device: torch.device, use_ema: bool, strict: bool):
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    args = load_args_from_ckpt(ckpt)

    model = get_model(args, args.model, args.model_type, args.weight_bits, args.input_bits)
    if use_ema and ("model_ema" in ckpt):
        state = ckpt["model_ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    return model, args

def build_eval_loader(split_dir: Path, args_ns: SimpleNamespace, batch_size: int, num_workers: int):
    tfm = build_transform(is_train=False, args=args_ns)
    ds = CervicalCancerDataset(str(split_dir), transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=getattr(args_ns, "pin_mem", True), drop_last=False)
    return dl, ds

@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
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

def infer_fold_name_from_run(run_dir: Path, fallback_index: int) -> str:
    # Grab trailing integer from path (e.g., bhvittiny_3 -> fold3)
    m = re.search(r"(\d+)$", run_dir.name)
    if m:
        return f"fold{int(m.group(1))}"
    return f"fold{fallback_index}"

def available_splits(fold_root: Path) -> List[str]:
    # We only evaluate splits that actually exist in each fold.
    return [s for s in ["val", "holdout_val", "test"] if (fold_root / s).exists()]

# ---------------------- Main ----------------------
def main():
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Map each run dir -> foldX
    run_dirs = [Path(p) for p in args.fold_runs]
    mappings = []
    for i, rd in enumerate(run_dirs):
        fold_name = infer_fold_name_from_run(rd, i)
        mappings.append((rd, fold_name))

    # Determine class names from the first existing split
    first_fold_dir = data_root / mappings[0][1]
    class_source = None
    for split in ["holdout_val", "val", "test"]:
        if (first_fold_dir / split).exists():
            class_source = first_fold_dir / split
            break
    if class_source is None:
        raise SystemExit(f"No splits found under {first_fold_dir} (looked for val/holdout_val/test).")
    classes = sorted([p.name for p in class_source.iterdir() if p.is_dir()])

    # Track which splits are present across ALL folds for ensembling
    common_splits = set(["val", "holdout_val", "test"])
    for _, fold_name in mappings:
        fold_root = data_root / fold_name
        common_splits &= set(available_splits(fold_root))
    common_splits = sorted(list(common_splits))

    # Storage for ensemble
    prob_stack = {s: [] for s in common_splits if s != "val"}  # ensemble only on holdout_val/test typically
    y_ref = {s: None for s in common_splits}

    per_fold_summaries = []

    for idx, (run_dir, fold_name) in enumerate(mappings):
        fold_root = data_root / fold_name
        best_path = run_dir / "best.pth"
        if not best_path.exists():
            raise SystemExit(f"[error] checkpoint missing: {best_path}")
        if not fold_root.exists():
            raise SystemExit(f"[error] fold directory missing: {fold_root}")

        print(f"\n=== Fold: {fold_name}  |  Run: {run_dir} ===")
        model, train_args = load_model(best_path, device, use_ema=args.use_ema_if_available, strict=args.strict_load)

        # Build TP once per fold so artifacts match your style
        tp = TrainingPlots(out_dir=out_dir / fold_name, class_names=classes, args={})

        fold_metrics = {"fold": fold_name}

        # Evaluate available splits
        for split in common_splits:
            dl, _ = build_eval_loader(fold_root / split, train_args, args.batch_size, args.num_workers)
            y_true, y_pred, y_prob = predict(model, dl, device)

            # Save predictions
            save_json(out_dir / "predictions" / fold_name / f"{split}_preds.json",
                      {"y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "y_prob": y_prob.tolist()})

            # Confusion/report/ROC via your plot module
            tp.save_confusion_and_report(model, dl, device, normalize="true", file_prefix=split)

            # Acc for quick glance
            fold_metrics[f"{split}_acc"] = float((y_pred == y_true).mean()) if len(y_true) else float("nan")

            # For ensemble on non-val splits
            if split != "val":
                prob_stack[split].append(y_prob)
                if y_ref[split] is None:
                    y_ref[split] = y_true

        save_json(out_dir / "metrics" / fold_name / "metrics.json", fold_metrics)
        per_fold_summaries.append(fold_metrics)

    # Ensemble across folds (mean probs) for each common non-val split
    ensemble_summary = {}
    for split in prob_stack.keys():
        if len(prob_stack[split]) == 0:
            continue
        probs_mean = np.mean(np.stack(prob_stack[split], axis=0), axis=0)
        preds_ens = probs_mean.argmax(axis=1)
        y_true = y_ref[split]
        acc = float((preds_ens == y_true).mean())
        ensemble_summary[f"{split}_acc"] = acc

        # Save preds
        save_json(out_dir / "predictions" / f"{split}_ensemble.json",
                  {"y_true": y_true.tolist(), "y_pred": preds_ens.tolist(), "y_prob": probs_mean.tolist()})

        # Confusion + simple ROC (macro) using numpy/sklearn directly
        try:
            from sklearn.metrics import confusion_matrix
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Confusion
            cm = confusion_matrix(y_true, preds_ens, labels=list(range(len(classes))))
            def plot_confusion(cm_arr, cname, out_png: Path, normalize=True):
                cmf = cm_arr.astype(np.float32)
                if normalize:
                    cmf = cmf / cmf.sum(axis=1, keepdims=True).clip(min=1e-9)
                fig, ax = plt.subplots(figsize=(6,5), dpi=150)
                im = ax.imshow(cmf, interpolation="nearest", aspect="auto")
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(len(cname)), yticks=np.arange(len(cname)),
                       xticklabels=cname, yticklabels=cname,
                       ylabel="True", xlabel="Predicted",
                       title=f"Confusion ({split} ensemble, normalized)")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                thresh = cmf.max() / 2.
                for i in range(cmf.shape[0]):
                    for j in range(cmf.shape[1]):
                        ax.text(j, i, f"{cmf[i,j]:.2f}",
                                ha="center", va="center",
                                color="white" if cmf[i,j] > thresh else "black")
                fig.tight_layout()
                out_png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png); plt.close(fig)

            plot_confusion(cm, classes, out_dir / "figures" / f"{split}_confusion_ens.png")

            # ROC (macro)
            Y = label_binarize(y_true, classes=list(range(len(classes))))
            valid = []
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(classes)):
                pos = Y[:, i].sum(); neg = (1 - Y[:, i]).sum()
                if pos > 0 and neg > 0:
                    fpr[i], tpr[i], _ = roc_curve(Y[:, i], probs_mean[:, i]); valid.append(i)
            if valid:
                all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in valid: mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(valid)
                fig = plt.figure(figsize=(7,6), dpi=150)
                for i in valid:
                    # per-class AUC
                    from sklearn.metrics import auc as _auc
                    auc_i = _auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC={auc_i:.3f})")
                plt.plot([0,1],[0,1], linestyle=":")
                plt.plot(all_fpr, mean_tpr, linestyle="--", label="macro")
                plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({split} ensemble)")
                plt.legend(loc="lower right")
                fig.tight_layout()
                fig.savefig(out_dir / "figures" / f"{split}_roc_ens.png"); plt.close(fig)
        except Exception as e:
            print(f"[warn] Skipping figures for ensemble {split}: {e}")

    save_json(out_dir / "metrics" / "summary.json",
              {"per_fold": per_fold_summaries, "ensemble": ensemble_summary})

    print("\n[done] wrote aggregated results to", str(out_dir.resolve()))

if __name__ == "__main__":
    main()
