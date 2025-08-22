# plot.py
"""
Utilities for logging, plotting and dumping training/validation metrics.

Creates:
  <output_dir>/plots/
    - loss_acc.png
    - confusion_matrix.png
    - roc_auc.png                  #  (or <prefix>_roc_auc.png when called with file_prefix)
  <output_dir>/metrics/
    - history.json
    - history.csv
    - confusion_matrix.npy
    - confusion_matrix.csv
    - confusion_matrix_normalized.csv
    - classification_report.json
    - classification_report.txt
    - predictions.csv  (idx, target, pred, correct, confidence)
    - roc_auc.json                 #  (per-class, micro, macro AUC; or with prefix)
    - (optional)                   # you can extend with CSV later if needed
Safe for DDP when called only on rank 0.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import torch

# Use a non-interactive backend
import matplotlib

from transformer.explanibility import BHViTAttribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ViTAttentionPlots:
    """
    Attention & rollout visualizations using BHViTAttribution.
    Saves results under <output_dir>/interpret/.
    """

    def __init__(self, out_dir: Path, mean=None, std=None):
        self.out_dir = Path(out_dir)
        self.viz_dir = self.out_dir / "interpret"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Defaults to ImageNet stats; override if your dataset differs
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # Attribution tool (lazy init per model)
        self.tool: BHViTAttribution | None = None

    def visualize_from_loader(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_images: int = 8,
    ) -> None:
        """
        Run attribution visualizations for a batch of images.
        """
        net = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        was_training = net.training
        net.eval()

        saved = 0
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, _ = batch[0], batch[1]
            elif isinstance(batch, dict) and "image" in batch:
                images = batch["image"]
            else:
                raise ValueError("Unsupported batch format for attention visualization")

            for b in range(images.size(0)):
                if saved >= max_images:
                    break
                img = images[b : b + 1].to(device, non_blocking=True)  # 1 x C x H x W
                self._visualize_single(net, img, device, index=saved)
                saved += 1
            if saved >= max_images:
                break

        if was_training:
            net.train()

    # ---------------- internals ----------------

    def _visualize_single(
        self,
        net: torch.nn.Module,
        img_1x: torch.Tensor,
        device: torch.device,
        index: int,
    ) -> None:
        """
        Run BHViTAttribution and save all rollout/head visualizations for one image.
        """
        # lazily init attribution tool
        if self.tool is None:
            self.tool = BHViTAttribution(net, image_size=224, window_size=7)

        pixel_values = img_1x.to(device)

        # --- non-grad rollouts under no_grad ---
        with torch.no_grad():
            rollouts: Dict[str, torch.Tensor] = {
                "mean": self.tool.last_stage_rollout(pixel_values, fusion="mean"),
                "gmean": self.tool.last_stage_rollout(pixel_values, fusion="gmean"),
                "max": self.tool.last_stage_rollout(pixel_values, fusion="max"),
                "sum": self.tool.last_stage_rollout(pixel_values, fusion="sum"),
            }
            head_maps = self.tool.last_stage_headmaps(pixel_values, layer_idx=-1)

        # --- grad rollout with autograd enabled ---
        pixel_values.requires_grad_(True)
        rollouts["grad"] = self.tool.last_stage_rollout(
            pixel_values, fusion="grad", use_grad_for_target=True
        )
        pixel_values.requires_grad_(False)

        # prepare base image
        base = self._denorm_to_numpy(img_1x.squeeze(0))

        # save rollout overlays
        for fusion, heat in rollouts.items():
            heat = heat[0].detach().cpu().numpy()
            self._save_overlay(
                base,
                self._resize_to_image(heat, base),
                self.viz_dir / f"sample_{index:03d}_rollout_{fusion}.png",
                title=f"Rollout ({fusion})",
            )

        # save per-head maps
        fig, axes = plt.subplots(
            1, min(4, head_maps.shape[1]), figsize=(12, 4), dpi=150
        )
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for i in range(min(4, head_maps.shape[1])):
            ax = axes[i]
            ax.imshow(head_maps[0, i].detach().cpu().numpy(), cmap="viridis")
            ax.set_title(f"Head {i}")
            ax.axis("off")
        fig.savefig(self.viz_dir / f"sample_{index:03d}_head_maps.png")
        plt.close(fig)

    # ---------------- utilities ----------------

    def _denorm_to_numpy(self, img_cxhxw: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(self.mean, device=img_cxhxw.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=img_cxhxw.device).view(3, 1, 1)
        x = img_cxhxw * std + mean
        x = x.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        return x

    def _resize_to_image(self, heat_hw: np.ndarray, base_hwc: np.ndarray) -> np.ndarray:
        h, w = base_hwc.shape[:2]
        t = torch.from_numpy(heat_hw).float().unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(h, w), mode="bilinear", align_corners=False
        )
        return t.squeeze(0).squeeze(0).cpu().numpy()

    def _save_overlay(
        self,
        base_img_hwc: np.ndarray,
        heat_hw: np.ndarray,
        fpath: Path,
        title: str = "",
    ) -> None:
        fig = plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(base_img_hwc)
        plt.imshow(heat_hw, alpha=0.6, cmap="jet")
        plt.axis("off")
        if title:
            plt.title(title)
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)


class TrainingPlots:
    def __init__(self, out_dir: Path, class_names: Optional[List[str]] = None) -> None:
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "plots"
        self.metrics_dir = self.out_dir / "metrics"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names  # may be None; will fallback to numeric labels
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],  # NEW
            "train_acc1": [],
            "val_acc1": [],
            "test_acc1": [],  # NEW
            "lr": [],
        }

    # ---------- Public API ----------

    def update_epoch(
        self,
        train_stats: Dict,
        val_stats: Dict,
        epoch: int,
        test_stats: Optional[Dict] = None,
    ) -> None:
        """
        Record stats and write rolling history (json/csv) + update loss/acc plot.

        Expected typical keys:
        train_stats: {"loss": float, "acc1": (optional), "lr": float}
        val_stats:   {"loss": float, "acc1": float}
        test_stats:  {"loss": float, "acc1": float}  (optional)
        Works even if some keys are missing.
        """
        self.history["epoch"].append(int(epoch))

        # Train loss / acc1
        self.history["train_loss"].append(float(train_stats.get("loss", np.nan)))
        tacc = train_stats.get("acc1", np.nan)
        if isinstance(tacc, (list, tuple)):
            tacc = tacc[0]
        self.history["train_acc1"].append(float(tacc) if tacc is not None else np.nan)

        # Val loss / acc1
        self.history["val_loss"].append(float(val_stats.get("loss", np.nan)))
        vacc = val_stats.get("acc1", np.nan)
        if isinstance(vacc, (list, tuple)):
            vacc = vacc[0]

        self.history["val_acc1"].append(float(vacc) if vacc is not None else np.nan)

        # Test loss / acc1 (optional)
        if test_stats is not None:
            self.history["test_loss"].append(float(test_stats.get("loss", np.nan)))
            teacc = test_stats.get("acc1", np.nan)
            if isinstance(teacc, (list, tuple)):
                teacc = teacc[0]
            self.history["test_acc1"].append(
                float(teacc) if teacc is not None else np.nan
            )
        else:
            # keep CSV column alignment even when test isn’t evaluated this epoch
            self.history["test_loss"].append(np.nan)
            self.history["test_acc1"].append(np.nan)

        # LR
        self.history["lr"].append(float(train_stats.get("lr", np.nan)))

        self._dump_history()
        self._plot_loss_acc()

    @torch.inference_mode()
    def save_confusion_and_report(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        normalize: str = "true",
        max_batches: Optional[int] = None,
        file_prefix: str = "",  # NEW
    ) -> None:
        """
        Runs a validation/test pass and saves:
        - {prefix_}confusion_matrix.png/.npy/.csv/.normalized.csv
        - {prefix_}classification_report.json/.txt
        - {prefix_}predictions.csv
        """
        stem = f"{file_prefix}_" if file_prefix else ""  # NEW

        y_true, y_pred, y_conf = self._collect_predictions(
            model, data_loader, device, max_batches
        )

        n_classes = (
            int(max(int(y_true.max()), int(y_pred.max())) + 1) if len(y_true) else 0
        )
        labels = list(range(n_classes))
        if self.class_names is None or len(self.class_names) != n_classes:
            self.class_names = [str(i) for i in labels]

        # Compute confusion matrix & report
        try:
            from sklearn.metrics import confusion_matrix, classification_report
        except Exception as e:
            raise ImportError(
                "scikit-learn is required for confusion matrix & classification report "
                "(pip install scikit-learn)"
            ) from e

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self._save_cm_arrays(cm, labels, file_prefix=file_prefix)  # NEW

        # Plot confusion matrix
        self._plot_confusion_matrix(
            cm,
            labels,
            self.class_names,
            normalize=normalize,
            file_prefix=file_prefix,  # NEW
        )

        self._save_roc_auc(
            model,
            data_loader,
            device,
            labels,
            self.class_names,
            file_prefix=file_prefix,
        )

        # Classification report
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        report_txt = classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )

        (self.metrics_dir / f"{stem}classification_report.json").write_text(
            json.dumps(report_dict, indent=2)
        )
        (self.metrics_dir / f"{stem}classification_report.txt").write_text(report_txt)

        # Per-sample predictions
        self._dump_predictions_csv(
            y_true, y_pred, y_conf, file_prefix=file_prefix
        )  # NEW

    def save_summary(self, max_accuracy: float, total_epochs: int) -> None:
        summary = {
            "max_accuracy_top1": float(max_accuracy),
            "total_epochs": int(total_epochs),
            "last_epoch": (
                int(self.history["epoch"][-1]) if self.history["epoch"] else None
            ),
        }
        (self.metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- Internals ----------
    def _dump_history(self) -> None:
        (self.metrics_dir / "history.json").write_text(
            json.dumps(self.history, indent=2)
        )
        keys = [
            "epoch",
            "train_loss",
            "val_loss",
            "test_loss",
            "train_acc1",
            "val_acc1",
            "test_acc1",
            "lr",
        ]
        with (self.metrics_dir / "history.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            rows = zip(*(self.history[k] for k in keys))
            for r in rows:
                writer.writerow(r)

    def _plot_loss_acc(self) -> None:
        """Plot separate loss and accuracy graphs."""
        epochs = self.history["epoch"]
        if not epochs:
            return

        # Create separate plots for loss and accuracy
        self._plot_loss()
        self._plot_accuracy()

    def _plot_loss(self) -> None:
        """Plot loss vs epoch for train, val, test."""
        epochs = self.history["epoch"]
        if not epochs:
            return

        fig = plt.figure(figsize=(10, 6), dpi=150)

        # Plot loss curves
        if not all(math.isnan(x) for x in self.history["train_loss"]):
            plt.plot(
                epochs,
                self.history["train_loss"],
                label="Train Loss",
                color="blue",
                linewidth=2,
            )
        if not all(math.isnan(x) for x in self.history["val_loss"]):
            plt.plot(
                epochs,
                self.history["val_loss"],
                label="Val Loss",
                color="red",
                linewidth=2,
            )
        if not all(math.isnan(x) for x in self.history["test_loss"]):
            plt.plot(
                epochs,
                self.history["test_loss"],
                label="Test Loss",
                color="green",
                linewidth=2,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.img_dir / "loss_vs_epoch.png")
        plt.close(fig)

    def _plot_accuracy(self) -> None:
        """Plot accuracy vs epoch for train, val, test."""
        epochs = self.history["epoch"]
        if not epochs:
            return

        fig = plt.figure(figsize=(10, 6), dpi=150)

        # Plot accuracy curves
        if not all(math.isnan(x) for x in self.history["train_acc1"]):
            plt.plot(
                epochs,
                self.history["train_acc1"],
                label="Train Acc@1",
                color="blue",
                linewidth=2,
            )
        if not all(math.isnan(x) for x in self.history["val_acc1"]):
            plt.plot(
                epochs,
                self.history["val_acc1"],
                label="Val Acc@1",
                color="red",
                linewidth=2,
            )
        if not all(math.isnan(x) for x in self.history["test_acc1"]):
            plt.plot(
                epochs,
                self.history["test_acc1"],
                label="Test Acc@1",
                color="green",
                linewidth=2,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.img_dir / "accuracy_vs_epoch.png")
        plt.close(fig)

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[int],
        class_names: List[str],
        normalize: str = "true",
        file_prefix: str = "",  # NEW
    ) -> None:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
        except Exception as e:
            raise ImportError(
                "scikit-learn required for plotting confusion matrix"
            ) from e

        cm_plot = cm.copy().astype(np.float64)
        norm_mode = (normalize or "").lower()
        if norm_mode in {"true", "pred", "all"}:
            with np.errstate(invalid="ignore", divide="ignore"):
                if norm_mode == "true":
                    cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
                elif norm_mode == "pred":
                    cm_plot = cm_plot / cm_plot.sum(axis=0, keepdims=True)
                else:
                    cm_plot = cm_plot / cm_plot.sum()

        size = min(20, max(6, len(labels) * 0.5))
        fig, ax = plt.subplots(figsize=(size, size), dpi=150)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_plot, display_labels=class_names
        )
        disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True)
        plt.title(
            "Confusion Matrix"
            + (
                f" (normalized={norm_mode})"
                if norm_mode in {"true", "pred", "all"}
                else ""
            )
        )
        plt.xticks(rotation=90)
        fig.tight_layout()

        stem = f"{file_prefix}_" if file_prefix else ""  # NEW
        fig.savefig(self.img_dir / f"{stem}confusion_matrix.png")  # NEW
        plt.close(fig)

    def _save_cm_arrays(
        self, cm: np.ndarray, labels: List[int], file_prefix: str = ""
    ) -> None:
        stem = f"{file_prefix}_" if file_prefix else ""  # NEW

        # Raw cm
        np.save(self.metrics_dir / f"{stem}confusion_matrix.npy", cm)

        # CSV raw
        with (self.metrics_dir / f"{stem}confusion_matrix.csv").open(
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, row in enumerate(cm):
                writer.writerow([labels[i]] + list(row.tolist()))

        # CSV normalized by true labels (rows)
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(np.float64) / np.maximum(row_sum, 1)
        with (self.metrics_dir / f"{stem}confusion_matrix_normalized.csv").open(
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, row in enumerate(cm_norm):
                writer.writerow([labels[i]] + [f"{v:.6f}" for v in row.tolist()])

    # ---------- NEW: ROC–AUC utilities ----------
    @torch.inference_mode()
    def _collect_probas(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            y_true: (N,) int labels
            y_score: (N, C) predicted probabilities for each class
        """
        model_was_training = model.training
        model.eval()

        all_t: List[np.ndarray] = []
        all_s: List[np.ndarray] = []

        for i, batch in enumerate(data_loader):
            # Support (images, targets) or dict-style batches
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            elif isinstance(batch, dict) and "image" in batch and "target" in batch:
                images, targets = batch["image"], batch["target"]
            else:
                raise ValueError("Unsupported batch format for validation dataloader")

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            logits = self._as_tensor(outputs)
            probs = torch.softmax(logits, dim=1)

            all_t.append(targets.detach().cpu().numpy())
            all_s.append(probs.detach().cpu().numpy())

            if max_batches is not None and (i + 1) >= max_batches:
                break

        if model_was_training:
            model.train()

        y_true = np.concatenate(all_t) if all_t else np.array([], dtype=np.int64)
        y_score = np.concatenate(all_s) if all_s else np.zeros((0, 0), dtype=np.float32)
        return y_true, y_score

        def _save_roc_auc(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            device: torch.device,
            labels: List[int],
            class_names: List[str],
            file_prefix: str = "",
        ) -> None:
            """
            Compute one-vs-rest ROC curves for each class and micro/macro averages.
            Saves:
                plots/<prefix_>roc_auc.png
                metrics/<prefix_>roc_auc.json
            """
            try:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc, roc_auc_score
            except Exception as e:
                raise ImportError(
                    "scikit-learn is required for ROC–AUC (pip install scikit-learn)"
                ) from e

            y_true, y_score = self._collect_probas(model, data_loader, device)
            if y_true.size == 0 or y_score.size == 0:
                return  # nothing to do

            n_classes = y_score.shape[1]
            # Safety: ensure names align
            if len(class_names) != n_classes:
                class_names = [str(i) for i in range(n_classes)]

            # Binarize ground-truth for OvR
            Y = label_binarize(y_true, classes=labels)  # (N, C)
            if Y.shape[1] != n_classes:
                # If labels were inferred smaller, expand to match y_score
                # (shouldn't happen with labels derived above, but guard anyway)
                full = np.zeros((Y.shape[0], n_classes), dtype=Y.dtype)
                full[:, : Y.shape[1]] = Y
                Y = full

            fpr: Dict[Any, np.ndarray] = {}
            tpr: Dict[Any, np.ndarray] = {}
            roc_auc: Dict[str, Optional[float]] = {}
            valid_cls: List[int] = []

            # Per-class curves
            for i in range(n_classes):
                pos = Y[:, i].sum()
                neg = (1 - Y[:, i]).sum()
                if pos > 0 and neg > 0:
                    fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_score[:, i])
                    roc_auc[class_names[i]] = float(auc(fpr[i], tpr[i]))
                    valid_cls.append(i)
                else:
                    # Not enough samples to compute ROC for this class
                    roc_auc[class_names[i]] = None

            # Micro-average
            if Y.sum() > 0 and (Y.size - Y.sum()) > 0:
                fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_score.ravel())
                roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))
            else:
                roc_auc["micro"] = None

            # Macro-average (mean TPR at merged FPR)
            if valid_cls:
                all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_cls]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in valid_cls:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(valid_cls)
                roc_auc["macro"] = float(auc(all_fpr, mean_tpr))
            else:
                roc_auc["macro"] = None

            # --- Plot ---
            size = 8
            fig = plt.figure(figsize=(size, 6), dpi=150)
            # Per-class lines
            for i in range(n_classes):
                if i in fpr:
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        lw=2,
                        label=f"{class_names[i]} (AUC={roc_auc[class_names[i]]:.3f})",
                    )
            # Micro/macro if available
            if "micro" in fpr:
                plt.plot(
                    fpr["micro"],
                    tpr["micro"],
                    lw=2,
                    linestyle="--",
                    label=f"micro (AUC={roc_auc['micro']:.3f})",
                )
            if valid_cls:
                plt.plot(
                    all_fpr,
                    mean_tpr,
                    lw=2,
                    linestyle="--",
                    label=f"macro (AUC={roc_auc['macro']:.3f})",
                )

            # Chance line
            plt.plot([0, 1], [0, 1], lw=1, linestyle=":", label="chance")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (one-vs-rest)")
            plt.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            stem = f"{file_prefix}_" if file_prefix else ""
            fig.savefig(self.img_dir / f"{stem}roc_auc.png")
            plt.close(fig)

            # --- Save metrics JSON ---
            # Convert None to JSON null
            (self.metrics_dir / f"{stem}roc_auc.json").write_text(
                json.dumps(roc_auc, indent=2)
            )

    def _dump_predictions_csv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_conf: np.ndarray,
        file_prefix: str = "",
    ) -> None:
        stem = f"{file_prefix}_" if file_prefix else ""  # NEW
        with (self.metrics_dir / f"{stem}predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "target", "pred", "correct", "confidence"])
            for i, (t, p, c) in enumerate(zip(y_true, y_pred, y_conf)):
                writer.writerow([i, int(t), int(p), int(t == p), float(c)])

    def _as_tensor(self, outputs):
        """Return a logits Tensor from a variety of model output types."""
        # 1) Already a Tensor
        if torch.is_tensor(outputs):
            return outputs

        # 2) HF / dataclass with .logits
        if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
            return outputs.logits

        # 3) Dict-like with 'logits'
        if (
            isinstance(outputs, dict)
            and "logits" in outputs
            and torch.is_tensor(outputs["logits"])
        ):
            return outputs["logits"]

        # 4) Tuple/list: pick first tensor-ish thing (or .logits on items)
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            for item in outputs:
                if torch.is_tensor(item):
                    return item
                if hasattr(item, "logits") and torch.is_tensor(item.logits):
                    return item.logits
            # fallback to first element if it’s a tensor
            if torch.is_tensor(outputs[0]):
                return outputs[0]

        raise TypeError(
            f"Could not extract a Tensor of logits from model output of type {type(outputs)}"
        )

    @torch.inference_mode()
    def _collect_predictions(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_was_training = model.training
        model.eval()

        all_t = []
        all_p = []
        all_c = []
        for i, batch in enumerate(data_loader):
            # Support (images, targets) or dict-style batches
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            elif isinstance(batch, dict) and "image" in batch and "target" in batch:
                images, targets = batch["image"], batch["target"]
            else:
                raise ValueError("Unsupported batch format for validation dataloader")

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)

            outputs = self._as_tensor(outputs)

            probs = torch.softmax(outputs, dim=1)
            conf, preds = probs.max(dim=1)

            all_t.append(targets.detach().cpu().numpy())
            all_p.append(preds.detach().cpu().numpy())
            all_c.append(conf.detach().cpu().numpy())

            if max_batches is not None and (i + 1) >= max_batches:
                break

        if model_was_training:
            model.train()

        y_true = np.concatenate(all_t) if all_t else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
        y_conf = np.concatenate(all_c) if all_c else np.array([], dtype=np.float32)
        return y_true, y_pred, y_conf
