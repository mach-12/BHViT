# ---------- bhvit_attribution.py (or append to BHViT.py) ----------
import math
from typing import List, Literal, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

HeadFusion = Literal["mean", "max", "sum", "gmean", "grad"]


@torch.no_grad()
def _safe_eye(n: int, device):
    return torch.eye(n, device=device).unsqueeze(0)  # (1, n, n)


def _fuse_heads(
    attn: torch.Tensor,
    fusion: HeadFusion = "mean",
    grad: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    attn: (B*Wn, H, L, L)
    grad: optional gradient for 'grad' fusion, same shape as attn
    returns: (B*Wn, L, L)
    """
    if fusion == "mean":
        return attn.mean(dim=1)
    if fusion == "max":
        return attn.max(dim=1).values
    if fusion == "sum":
        return attn.sum(dim=1)
    if fusion == "gmean":
        a = attn.clamp(min=eps).log().mean(dim=1).exp()
        return a
    if fusion == "grad":
        if grad is None:
            raise ValueError("grad fusion selected but no gradients provided.")
        # weight per head by its |grad| magnitude (class-specific)
        # -> (B*Wn, H, 1, 1)
        w = grad.abs().mean(dim=(-1, -2), keepdim=True)
        w = w / (w.sum(dim=1, keepdim=True) + eps)
        return (attn * w).sum(dim=1)
    raise ValueError(f"Unknown fusion: {fusion}")


def _row_stochastic(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return A / (A.sum(dim=-1, keepdim=True) + eps)


def _rollout(mats: List[torch.Tensor], add_residual: bool = True) -> torch.Tensor:
    """
    mats: list of (B, L, L), ordered from earlier -> later layer (within the same spatial stage)
    returns joint attention (B, L, L)
    """
    B, L, _ = mats[0].shape
    joint = _safe_eye(L, mats[0].device).expand(B, L, L).clone()
    for A in mats:
        if add_residual:
            A = _row_stochastic(A + _safe_eye(A.size(-1), A.device).expand_as(A))
        else:
            A = _row_stochastic(A)
        joint = torch.bmm(A, joint)
    return joint  # (B, L, L)


def _reshape_last_stage(attn: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    attn: (B*Wn, H, L, L)  expected Wn == 1 for last stage
    returns: (B, H, L, L)
    """
    B_times_Wn, H, L, _ = attn.shape
    Wn = L - 49  # because window_size=7 => local tokens = 7*7=49
    if Wn != 1:
        raise ValueError(f"Expected last-stage (Wn=1), got Wn={Wn} from L={L}")
    B = batch_size
    assert B_times_Wn == B * Wn
    return attn.view(B, H, L, L)


def _extract_local_block(A: torch.Tensor) -> torch.Tensor:
    """
    A: (..., L, L) where L = 49 + Wn and Wn == 1 here
    returns local-to-local block (..., 49, 49)
    """
    L_total = A.size(-1)
    L_local = 49
    return A[..., :L_local, :L_local]


class BHViTAttribution:
    """
    Tools for BHViT attention maps and rollout (last stage).
    Works with your BHViTForImageClassification as-is.
    """

    def __init__(self, model, image_size: int = 224, window_size: int = 7):
        self.model = model
        self.image_size = image_size
        self.window_size = window_size  # assumed fixed as in BHViTSelfAttention

    def _forward_with_attn(
        self,
        pixel_values: torch.Tensor,
        require_grads_for_gradfusion: bool = False,
        target_index: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], torch.Tensor]:
        """
        Returns:
          attentions: list of tensors of shape (B*Wn, H, L, L) for each layer in layerB
          attn_grads: same shape list if grad fusion requested; else None
          logits: (B, num_labels)
        """
        self.model.eval()
        # Forward pass with attentions
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits
        attns: List[torch.Tensor] = list(outputs.attentions)  # length == len(layerB)

        if not require_grads_for_gradfusion:
            return attns, None, logits

        # capture grads w.r.t. attention probs
        attn_vars = [a.clone().detach().requires_grad_(True) for a in attns]
        # Re-run a lightweight head-only pass using detached activations is tricky;
        # instead, we backprop through the original graph by retaining grads on the original tensors:
        for a in outputs.attentions:
            a.retain_grad()

        # pick target indices
        if target_index is None:
            # default: argmax per sample
            target_index = logits.argmax(dim=1)
        # Backprop class score
        self.model.zero_grad(set_to_none=True)
        sel = logits[torch.arange(logits.size(0), device=logits.device), target_index]
        sel.sum().backward(retain_graph=False)

        attn_grads = [a.grad.detach().clone() for a in outputs.attentions]
        return attns, attn_grads, logits

    @torch.no_grad()
    def last_stage_headmaps(
        self, pixel_values: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Returns per-head maps from a single last-stage layer.
        Shape: (B, H, 7, 7) where H=num_heads of that layer
        """
        attns, _, _ = self._forward_with_attn(
            pixel_values, require_grads_for_gradfusion=False
        )
        A = attns[layer_idx]  # (B*Wn, H, L, L)
        A = _reshape_last_stage(A, pixel_values.size(0))  # (B, H, L, L), Wn=1 enforced
        A_ll = _extract_local_block(A)  # (B, H, 49, 49)
        # For a heatmap per head, we aggregate target importance by averaging queries:
        # (common choice with mean-pooled logits)
        head_maps = A_ll.mean(dim=2)  # (B, H, 49)
        return head_maps.view(head_maps.size(0), head_maps.size(1), 7, 7)

    def last_stage_rollout(
        self,
        pixel_values: torch.Tensor,
        fusion: HeadFusion = "mean",
        add_residual: bool = True,
        seed: Literal["uniform", "ones"] = "uniform",
        layers: Optional[List[int]] = None,
        use_grad_for_target: bool = False,
        target_index: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Attention rollout over the last stage (7x7), fusing heads per layer.

        Returns:
          heatmaps: (B, H, W) upsampled to image_size x image_size
        """
        need_grads = (fusion == "grad") or use_grad_for_target
        attns, attn_grads, _ = self._forward_with_attn(
            pixel_values,
            require_grads_for_gradfusion=need_grads,
            target_index=target_index,
        )
        B = pixel_values.size(0)

        # Keep only last-stage layers (Wn == 1)
        last_stage = []
        last_stage_grads = [] if attn_grads is not None else None
        for i, A in enumerate(attns):
            L = A.size(-1)
            Wn = L - 49
            if Wn == 1:  # last stage
                last_stage.append(A)
                if last_stage_grads is not None:
                    last_stage_grads.append(attn_grads[i])

        if len(last_stage) == 0:
            raise RuntimeError(
                "No last-stage attentions found (expected layers after final patch_embed3)."
            )

        # Optionally subset specific layers (indices relative to last_stage list)
        if layers is not None:
            last_stage = [last_stage[i] for i in layers]
            if last_stage_grads is not None:
                last_stage_grads = [last_stage_grads[i] for i in layers]

        # Fuse heads -> (B, 49, 49) for each layer, in order
        fused: List[torch.Tensor] = []
        for li, A in enumerate(last_stage):
            A_bhll = _reshape_last_stage(A, B)  # (B, H, L, L)
            G = (
                None
                if last_stage_grads is None
                else _reshape_last_stage(last_stage_grads[li], B)
            )
            A_ll = _extract_local_block(A_bhll)  # (B, H, 49, 49)
            G_ll = None if G is None else _extract_local_block(G)
            fused_A = _fuse_heads(A_ll, fusion=fusion, grad=G_ll)  # (B, 49, 49)
            fused.append(fused_A)

        # Rollout across the selected last-stage layers
        joint = _rollout(fused, add_residual=add_residual)  # (B, 49, 49)

        # Seed distribution (mean-pooled classifier typically uses uniform)
        if seed == "uniform":
            s0 = torch.full((B, 49, 1), 1.0 / 49.0, device=joint.device)
        else:
            s0 = torch.ones(B, 49, 1, device=joint.device)

        scores = torch.bmm(joint.transpose(1, 2), s0).squeeze(-1)  # (B, 49)
        maps7 = scores.view(B, 1, 7, 7)
        if normalize:
            # min-max per sample for nicer visualization
            maps7 = (maps7 - maps7.amin(dim=(2, 3), keepdim=True)) / (
                maps7.amax(dim=(2, 3), keepdim=True) + 1e-6
            )
        maps = F.interpolate(
            maps7,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # (B, H, W)

        return maps  # torch.Tensor on same device as inputs
