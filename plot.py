# plot.py
"""
Utilities for logging, plotting and dumping training/validation metrics.

Creates:
  <output_dir>/plots/
    - loss_acc.png
    - confusion_matrix.png
  <output_dir>/metrics/
    - history.json
    - history.csv
    - confusion_matrix.npy
    - confusion_matrix.csv
    - confusion_matrix_normalized.csv
    - classification_report.json
    - classification_report.txt
    - predictions.csv  (idx, target, pred, correct, confidence)

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ViTAttentionPlots:
    """
    HF ViT/DeiT/BHViT attention & rollout visualizations.
    Uses model outputs.attentions (set via output_attentions=True).
    Saves under <output_dir>/interpret/.
    """

    def __init__(self, out_dir: Path, mean=None, std=None):
        self.out_dir = Path(out_dir)
        self.viz_dir = self.out_dir / "interpret"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        # Defaults to ImageNet stats; override if your dataset differs
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    @torch.inference_mode()
    def visualize_from_loader(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_images: int = 8,
        save_rollout: bool = True,
    ) -> None:
        net = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        was_training = net.training
        net.eval()

        # Ensure attentions are produced
        if hasattr(net, "config"):
            try:
                net.config.output_attentions = True
                net.config.return_dict = True
            except Exception:
                pass

        saved = 0
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, _ = batch[0], batch[1]
            elif isinstance(batch, dict) and "image" in batch and "target" in batch:
                images, _ = batch["image"], batch["target"]
            else:
                raise ValueError("Unsupported batch format for attention visualization")

            for b in range(images.size(0)):
                if saved >= max_images:
                    break
                img = images[b : b + 1].to(device, non_blocking=True)  # 1 x C x H x W
                ok = self._visualize_single_hf(
                    net, img, device, index=saved, save_rollout=save_rollout
                )
                if not ok:
                    # Fallback note if no attentions available
                    (self.viz_dir / f"sample_{saved:03d}_SKIPPED.txt").write_text(
                        "No attentions returned by the model. "
                        "Ensure BHViT forward supports output_attentions/return_dict."
                    )
                saved += 1
            if saved >= max_images:
                break

        if was_training:
            net.train()

    # ---------------- internals ----------------

    def _best_grid_from_M(self, M: int) -> tuple[int, int]:
        """Return (gh, gw) with gh*gw=M and |gh-gw| minimal."""
        r = int(math.sqrt(M))
        for h in range(r, 0, -1):
            if M % h == 0:
                return h, M // h
        return 1, M  # fallback for prime M

    def _infer_grid_hw(self, num_tokens: int) -> tuple[int, int]:
        """
        Infer (H, W) grid from a flat length `num_tokens`.
        Prefers square; falls back to a near-rectangle if needed.
        """
        side = int(round(num_tokens**0.5))
        if side * side == num_tokens:
            return side, side
        # fallback for non-perfect squares
        h = int(math.floor(num_tokens**0.5))
        w = int(math.ceil(num_tokens / max(h, 1)))
        return h, w

    def _infer_tokens_start(self, N: int) -> tuple[int, int, int]:
        """
        Infer special token count t in [1..32]. Choose t whose M=N-t factorizes
        into the most square-ish grid. Returns (tokens_start, gh_guess, gw_guess).
        """
        best = None
        for t in range(1, min(32, N)):
            M = N - t
            if M <= 0:
                continue
            gh, gw = self._best_grid_from_M(M)
            score = abs(gh - gw)  # smaller is better
            if best is None or score < best[0]:
                best = (score, t, gh, gw)
        # Fallback to a single CLS if something is odd
        if best is None:
            return 1, *self._best_grid_from_M(N - 1)
        _, t, gh, gw = best
        return t, gh, gw

    def _cfg_grid_tokens(self, net) -> int | None:
        """Return expected grid token count from config if available, else None."""
        try:
            cfg = getattr(net, "config", None)
            if cfg is None:
                return None
            img = int(getattr(cfg, "image_size", 224))
            patch = int(getattr(cfg, "patch_size", 16))
            if patch > 0:
                g = (img // patch) * (img // patch)
                return g if g > 0 else None
        except Exception:
            pass
        return None

    def _infer_tokens_and_grid(self, net, N: int) -> tuple[int, int, int]:
        """
        Robustly infer (tokens_start, gh, gw):
        - tries tokens_start in [1..32] (small special token counts),
        - prefers M=N-t that factorizes to near-square grid,
        - if cfg grid is known, also prefers M close to that.
        """
        cfg_M = self._cfg_grid_tokens(net)  # e.g., 196 for 224/16
        best = None
        for t in range(1, min(32, N)):  # search small special-token counts
            M = N - t
            gh, gw = self._best_grid_from_M(M)
            squareness = abs(gh - gw)  # smaller is better
            cfg_dist = abs(M - cfg_M) if cfg_M is not None else 0
            score = (10 * squareness) + cfg_dist  # bias toward square, then toward cfg
            if best is None or score < best[0]:
                best = (score, t, gh, gw, M)
        _, tokens_start, gh, gw, _ = best
        return tokens_start, gh, gw

    def _visualize_single_hf(
        self,
        net: torch.nn.Module,
        img_1x: torch.Tensor,
        device: torch.device,
        index: int,
        save_rollout: bool = True,
    ) -> bool:
        # Try calling with HF keywords first; fall back to positional
        outputs = None
        try:
            outputs = net(pixel_values=img_1x, output_attentions=True, return_dict=True)
        except TypeError:
            try:
                outputs = net(img_1x, output_attentions=True, return_dict=True)
            except TypeError:
                outputs = net(img_1x)

        # Extract attentions (list/tuple of L tensors: [B, heads, N, N])
        attentions = None
        if outputs is not None:
            if isinstance(outputs, dict):
                attentions = outputs.get("attentions", None)
            else:
                attentions = getattr(outputs, "attentions", None)

        if not attentions:
            return False

        # Head-average -> list of [1, N, N]
        attn_mean_list = [a.mean(dim=1) for a in attentions]  # L x [B, N, N]
        A0 = attn_mean_list[0]  # [1, N, N]
        N = int(A0.shape[-1])

        # robust tokens/grid
        tokens_start, gh_guess, gw_guess = self._infer_tokens_start(N)
        M_expected = N - tokens_start

        per_layer_cls = []
        for A in attn_mean_list:
            cls_to_patches = A[:, 0, tokens_start:]  # [1, M]
            M = int(cls_to_patches.shape[-1])
            gh_layer, gw_layer = self._best_grid_from_M(M)
            num_tokens = cls_to_patches.numel()  # e.g., 196
            gh_layer, gw_layer = self._infer_grid_hw(num_tokens)  # -> (14, 14) for 196
            grid = cls_to_patches.reshape(gh_layer, gw_layer)  # [14,14]

            # if later code expects [1,1,H,W] for interpolation:
            heat = grid.unsqueeze(0).unsqueeze(0)  # [1,1,14,14]
            per_layer_cls.append(heat)

        # Rollout
        rollout_heat = None
        if save_rollout:
            rollout_heat = self._attention_rollout(attn_mean_list, tokens_start, gh, gw)

        # Save overlays
        base = self._denorm_to_numpy(img_1x.squeeze(0))
        last_layer = per_layer_cls[-1].squeeze(0).cpu().numpy()
        self._save_overlay(
            base,
            self._resize_to_image(last_layer, base),
            self.viz_dir / f"sample_{index:03d}_overlay_last.png",
            title="Last layer CLS attention",
        )

        if rollout_heat is not None:
            self._save_overlay(
                base,
                self._resize_to_image(rollout_heat, base),
                self.viz_dir / f"sample_{index:03d}_overlay_rollout.png",
                title="Attention Rollout",
            )

        self._save_per_layer_grid(
            base,
            per_layer_cls,
            fpath=self.viz_dir / f"sample_{index:03d}_layers_grid.png",
        )

        # Dumps
        np.save(
            self.viz_dir / f"sample_{index:03d}_per_layer_cls.npy",
            np.stack([h.squeeze(0).cpu().numpy() for h in per_layer_cls], axis=0),
        )
        if rollout_heat is not None:
            np.save(self.viz_dir / f"sample_{index:03d}_rollout.npy", rollout_heat)
        return True

    def _infer_tokens_and_grid(
        self, net: torch.nn.Module, N: int
    ) -> Tuple[int, int, int]:
        """
        tokens_start: number of special tokens (usually 1; 2 for distilled).
        gh, gw: patch grid size.
        Strategy: prefer config.image_size/patch_size; else infer from N.
        """
        tokens_start = 1
        gh = gw = 14  # sensible default for 224/16
        try:
            cfg = getattr(net, "config", None)
            if cfg is not None:
                img = int(getattr(cfg, "image_size", 224))
                patch = int(getattr(cfg, "patch_size", 16))
                gh = img // patch
                gw = img // patch
                # infer special tokens by difference to N
                grid_tokens = gh * gw
                tokens_start = max(1, N - grid_tokens)
        except Exception:
            pass
        # If model is distilled (CLS + DIST), tokens_start should be 2
        for name in (
            "distillation",
            "use_distillation",
            "has_dist_token",
            "enable_cls_",
        ):
            if hasattr(net, "config") and bool(getattr(net.config, name, False)):
                tokens_start = max(tokens_start, 2)
        return tokens_start, gh, gw

    def _attention_rollout(
        self,
        attn_mean_list: List[torch.Tensor],
        tokens_start: int,
        gh_hint: int,
        gw_hint: int,
    ) -> np.ndarray:
        A_accum = None
        for A_l in attn_mean_list:
            A_l = A_l.squeeze(0)  # N x N
            N = A_l.shape[-1]
            A_hat = A_l + torch.eye(N, device=A_l.device)
            A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
            A_accum = A_hat if A_accum is None else (A_accum @ A_hat)
        cls_to_patches = A_accum[0, tokens_start:]  # (M,)
        M = int(cls_to_patches.numel())
        gh, gw = self._best_grid_from_M(M)  # factorize from actual M
        rollout = cls_to_patches.reshape(gh, gw).detach().cpu().numpy()
        return rollout

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

    def _save_per_layer_grid(
        self, base_img_hwc: np.ndarray, per_layer_cls: List[torch.Tensor], fpath: Path
    ) -> None:
        L = len(per_layer_cls)
        cols = 4
        rows = int(math.ceil(L / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = np.atleast_2d(axes)
        for i, heat in enumerate(per_layer_cls):
            ax = axes[i // cols, i % cols]
            ax.imshow(base_img_hwc)
            h = heat.squeeze(0).cpu().numpy()
            h = self._resize_to_image(h, base_img_hwc)
            ax.imshow(h, alpha=0.6, cmap="jet")
            ax.set_title(f"Layer {i+1}")
            ax.axis("off")
        for j in range(L, rows * cols):
            ax = axes[j // cols, j % cols]
            ax.axis("off")
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
            "train_acc1": [],
            "val_acc1": [],
            "lr": [],
        }

    # ---------- Public API ----------

    def update_epoch(self, train_stats: Dict, val_stats: Dict, epoch: int) -> None:
        """
        Record stats and write rolling history (json/csv) + update loss/acc plot.

        Expected typical keys:
          train_stats: {"loss": float, "acc1": (optional), "lr": float}
          val_stats:   {"loss": float, "acc1": float, "acc5": (optional)}
        Works even if some keys are missing.
        """
        self.history["epoch"].append(int(epoch))

        # Train loss
        self.history["train_loss"].append(float(train_stats.get("loss", np.nan)))
        # Val loss
        self.history["val_loss"].append(float(val_stats.get("loss", np.nan)))

        # Train acc1 (may be missing)
        tacc = train_stats.get("acc1", np.nan)
        if isinstance(tacc, (list, tuple)):
            tacc = tacc[0]
        self.history["train_acc1"].append(float(tacc) if tacc is not None else np.nan)

        # Val acc1
        vacc = val_stats.get("acc1", np.nan)
        if isinstance(vacc, (list, tuple)):
            vacc = vacc[0]
        self.history["val_acc1"].append(float(vacc) if vacc is not None else np.nan)

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
    ) -> None:
        """
        Runs a validation pass to collect predictions/targets and saves:
          - confusion_matrix.png, .npy, .csv, normalized .csv
          - classification_report.json, .txt
          - predictions.csv (per-sample)
        """
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
        except Exception as e:  # sklearn not installed
            raise ImportError(
                "scikit-learn is required for confusion matrix & classification report "
                "(pip install scikit-learn)"
            ) from e

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self._save_cm_arrays(cm, labels)

        # Plot confusion matrix
        self._plot_confusion_matrix(cm, labels, self.class_names, normalize=normalize)

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

        (self.metrics_dir / "classification_report.json").write_text(
            json.dumps(report_dict, indent=2)
        )
        (self.metrics_dir / "classification_report.txt").write_text(report_txt)

        # Save per-sample predictions
        self._dump_predictions_csv(y_true, y_pred, y_conf)

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
        # JSON
        (self.metrics_dir / "history.json").write_text(
            json.dumps(self.history, indent=2)
        )
        # CSV
        keys = ["epoch", "train_loss", "val_loss", "train_acc1", "val_acc1", "lr"]
        with (self.metrics_dir / "history.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            rows = zip(*(self.history[k] for k in keys))
            for r in rows:
                writer.writerow(r)

    def _plot_loss_acc(self) -> None:
        epochs = self.history["epoch"]
        if not epochs:
            return

        # Build figure
        fig = plt.figure(figsize=(10, 6), dpi=150)

        # Loss
        if not all(math.isnan(x) for x in self.history["train_loss"]):
            plt.plot(epochs, self.history["train_loss"], label="train loss")
        if not all(math.isnan(x) for x in self.history["val_loss"]):
            plt.plot(epochs, self.history["val_loss"], label="val loss")

        # Accuracy (top-1)
        # Plot on same axes but with dashed lines to differentiate
        if not all(math.isnan(x) for x in self.history["train_acc1"]):
            plt.plot(
                epochs,
                self.history["train_acc1"],
                linestyle="--",
                label="train acc@1 (%)",
            )
        if not all(math.isnan(x) for x in self.history["val_acc1"]):
            plt.plot(
                epochs, self.history["val_acc1"], linestyle="--", label="val acc@1 (%)"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Loss & Accuracy vs Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.img_dir / "loss_acc.png")
        plt.close(fig)

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[int],
        class_names: List[str],
        normalize: str = "true",
    ) -> None:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
        except Exception as e:
            raise ImportError(
                "scikit-learn required for plotting confusion matrix"
            ) from e

        # Optional normalization
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
        disp.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True)
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
        fig.savefig(self.img_dir / "confusion_matrix.png")
        plt.close(fig)

    def _save_cm_arrays(self, cm: np.ndarray, labels: List[int]) -> None:
        # Raw cm
        np.save(self.metrics_dir / "confusion_matrix.npy", cm)

        # CSV raw
        with (self.metrics_dir / "confusion_matrix.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, row in enumerate(cm):
                writer.writerow([labels[i]] + list(row.tolist()))

        # CSV normalized by true labels (rows)
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(np.float64) / np.maximum(row_sum, 1)
        with (self.metrics_dir / "confusion_matrix_normalized.csv").open(
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for i, row in enumerate(cm_norm):
                writer.writerow([labels[i]] + [f"{v:.6f}" for v in row.tolist()])

    def _dump_predictions_csv(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray
    ) -> None:
        with (self.metrics_dir / "predictions.csv").open("w", newline="") as f:
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
