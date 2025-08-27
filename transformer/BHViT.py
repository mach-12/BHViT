"""PyTorch ViT model."""

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)

from transformers.utils import logging
from transformers import ViTConfig
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .utils_quant import (
    QuantizeLinear,
    QuantizeConv2d,
    QuantizeConv2d2,
    BinaryQuantizer,
    BiTBinaryQuantizer,
    GSB_Attention,
    BinaryActivation_Attention,
)


logger = logging.get_logger(__name__)


# simple per-image stain normalization
class StainInstanceNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(
            num_channels, eps=eps, affine=affine, track_running_stats=False
        )

    def forward(self, x):
        return self.inorm(x)


# 2D sine-cosine pos-emb builder (H*W must be even-dim compatible)
def _get_2d_sincos_pos_embed(C: int, H: int, W: int, device):
    assert C % 2 == 0
    y = torch.linspace(-1, 1, steps=H, device=device)
    x = torch.linspace(-1, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H,W]
    freqs = torch.linspace(1.0, C // 4, steps=C // 4, device=device).view(1, 1, -1)
    yy = yy.unsqueeze(-1) * freqs  # [H,W,C/4]
    xx = xx.unsqueeze(-1) * freqs  # [H,W,C/4]
    pe = torch.cat(
        [torch.sin(yy), torch.cos(yy), torch.sin(xx), torch.cos(xx)], dim=-1
    )  # [H,W,C]
    return pe.permute(2, 0, 1).contiguous()  # [C,H,W]


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = (
            self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2)
            + self.move2
        )
        return out


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_ones=True):
        super().__init__()
        if init_ones:
            self.alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.move = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = x * self.alpha + self.move
        return out


class BHViTEmbeddings(nn.Module):
    """Construct Position and Patch Embeddings"""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embeddings = BHViTPatchEmbeddings(config)
        self.num_patches = config.image_size // 4
        C0 = config.hidden_size[0]

        # optional: stain norm before patch conv
        self.stain_norm = (
            StainInstanceNorm(
                getattr(config, "num_channels", 3),
                eps=config.layer_norm_eps,
                affine=True,
            )
            if getattr(config, "stain_instance_norm", False)
            else nn.Identity()
        )

        # positional embeddings: learned (default) or sine-cosine (toggle)
        if getattr(config, "use_sinusoidal_pe", False):
            # buffer is materialized on first forward when device is known
            self.register_buffer("position_embeddings", None, persistent=False)
            self.use_sincos = True
        else:
            self.use_sincos = False
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, C0, self.num_patches, self.num_patches)
            )
            trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stain_norm(x)
        x = self.patch_embeddings(x)  # [B, C0, H/4, W/4] with H/4 = W/4 = num_patches

        if self.use_sincos:
            if self.position_embeddings is None or self.position_embeddings.size(
                -1
            ) != x.size(-1):
                pe = _get_2d_sincos_pos_embed(x.size(1), x.size(2), x.size(3), x.device)
                self.position_embeddings = pe.unsqueeze(0)  # [1, C, Hp, Wp]
        x = x + self.position_embeddings
        return x


class BHViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, in_chans=3, out_chans=64):
        super().__init__()
        self.config = config
        self.coord_conv = bool(getattr(config, "coord_conv", False))
        cin = getattr(config, "num_channels", in_chans) + (2 if self.coord_conv else 0)
        self.proj = nn.Conv2d(cin, out_chans, kernel_size=4, stride=4)
        self.norm = config.norm_layer(64, eps=config.layer_norm_eps)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add (x,y) if enabled; normalized to [-1,1]
        if self.coord_conv:
            B, _, H, W = x.shape
            yy = (
                torch.linspace(-1, 1, steps=H, device=x.device)
                .view(1, 1, H, 1)
                .expand(B, 1, H, W)
            )
            xx = (
                torch.linspace(-1, 1, steps=W, device=x.device)
                .view(1, 1, 1, W)
                .expand(B, 1, H, W)
            )
            x = torch.cat([x, yy, xx], dim=1)
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1).flatten(1, 2))
        x = self.act(x)
        return x


#######################################
class Token_for_Attention(nn.Module):
    def __init__(self, dim, config, window_size=7):
        super(Token_for_Attention, self).__init__()
        self.window_size = window_size
        self.merge_avg = nn.AvgPool2d(
            kernel_size=window_size, stride=window_size, padding=3
        )
        self.merge_max = nn.MaxPool2d(
            kernel_size=window_size, stride=window_size, padding=3
        )
        self.a1 = nn.Parameter(0.5 * torch.ones([1, 1, dim]), requires_grad=True)
        self.norm = nn.LayerNorm(dim)

    def Merge_token(self, x):
        merge_token_avage = self.merge_avg(x).permute(0, 2, 3, 1).flatten(1, 2)
        merge_token_max = self.merge_max(x).permute(0, 2, 3, 1).flatten(1, 2)
        merge_token = (
            self.a1.expand_as(merge_token_max) * merge_token_max
            + (1.0 - self.a1).expand_as(merge_token_avage) * merge_token_avage
        )
        return merge_token

    def forward(self, x):
        windows = windows_split(x, self.window_size)
        rep_ratio = x.shape[2] // self.window_size
        merge_token = self.Merge_token(x)  # B,N,C->B,N1,C
        merge_token_new = torch.repeat_interleave(
            merge_token, rep_ratio**2, dim=0
        )  # B,N1,C->B1,N1,C
        token_all = torch.cat((windows, merge_token_new), dim=1)  # B1,N1+N,C
        token_all = self.norm(token_all)
        return token_all, windows.shape[1]


def windows_split(x, window_size):
    (
        B,
        C,
        H,
        W,
    ) = x.shape
    x = x.permute(0, 2, 3, 1).view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    )
    return windows


#########################################
class BHViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()

        ws = int(getattr(config, "attn_window", 7))
        self.token_FA = Token_for_Attention(
            dim=config.hidden_size[config.stages[layer_num]],
            config=config,
            window_size=ws,
        )
        self.windows_size = ws

        self.num_attention_heads = config.num_attention_heads[config.stages[layer_num]]
        self.attention_head_size = int(
            config.hidden_size[config.stages[layer_num]]
            / config.num_attention_heads[config.stages[layer_num]]
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.moveq = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.movek = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.movev = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )

        self.query = QuantizeLinear(
            config.hidden_size[config.stages[layer_num]],
            self.all_head_size,
            config=config,
        )
        self.key = QuantizeLinear(
            config.hidden_size[config.stages[layer_num]],
            self.all_head_size,
            config=config,
        )
        self.value = QuantizeLinear(
            config.hidden_size[config.stages[layer_num]],
            self.all_head_size,
            config=config,
        )

        self.normq = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.normk = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.normv = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )

        self.rpreluq = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluk = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.rpreluv = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.moveq2 = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.movek2 = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.movev2 = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )

        self.act_quantizer = None
        self.att_prob_quantizer = None
        self.att_prob_clip = None

        if config.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
            if config.gsb:
                self.att_prob_quantizer = GSB_Attention
                self.att_prob_clip = nn.Parameter(torch.tensor(0.005))
                self.att_prob_clip2 = nn.Parameter(torch.tensor(1.0))
                self.att_prob_clip3 = nn.Parameter(torch.tensor(1.0))
            else:
                self.att_prob_quantizer = BinaryActivation_Attention(
                    self.num_attention_heads, 3
                )
                self.att_prob_clip2 = None
        self.norm_context = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )

        self.rprelu_context = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.parm = nn.Parameter(
            0.5 * torch.ones([1, config.hidden_size[config.stages[layer_num]], 1, 1]),
            requires_grad=True,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def window_reverse(self, windows, window_size, H, W, B):
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(
            B, windows.shape[2], H, W
        )  # windows.shape=【B*H // window_size*W // window_size，window_size*window_size，C】
        return x

    def window_reverse_high(self, windows, window_size, H, W, B):
        x = windows.view(
            B,
            H // window_size * W // window_size,
            H // window_size,
            W // window_size,
            -1,
        )
        x = torch.mean(x, dim=1).permute(
            0, 3, 1, 2
        )  # windows.shape=【B*H // window_size*W // window_size，H // window_size*W // window_size，C】
        x = torch.nn.functional.interpolate(x, size=H, mode="nearest")
        return x

    def token_split(self, x, split_dim, H, W, B):
        x1, x2 = torch.split(x, split_dim, dim=1)
        x1 = self.window_reverse(x1, self.windows_size, H, W, B)
        x2 = self.window_reverse_high(x2, self.windows_size, H, W, B)
        return x1 * self.parm.expand_as(x1) + x2 * (1.0 - self.parm).expand_as(x2)

    def forward(
        self, hidden_states, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        B, C, H, W = hidden_states.shape
        hidden_states, split_dim = self.token_FA(hidden_states)
        mixed_query_layer = (
            self.normq(self.query(hidden_states + self.moveq)) + hidden_states
        )
        mixed_key_layer = (
            self.normk(self.key(hidden_states + self.movek)) + hidden_states
        )
        mixed_value_layer = (
            self.normv(self.value(hidden_states + self.movev)) + hidden_states
        )
        mixed_query_layer = self.rpreluq(mixed_query_layer)
        mixed_key_layer = self.rpreluk(mixed_key_layer)
        mixed_value_layer = self.rpreluv(mixed_value_layer)
        query_layer = mixed_query_layer + self.moveq2
        key_layer = mixed_key_layer + self.movek2
        value_layer = mixed_value_layer + self.movev2

        if self.act_quantizer is not None:
            query_layer = self.act_quantizer.apply(query_layer)
            key_layer = self.act_quantizer.apply(key_layer)
            value_layer = self.act_quantizer.apply(value_layer)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if self.att_prob_quantizer is not None:
            if self.att_prob_clip2 is not None:
                attention_probs = self.att_prob_quantizer.apply(
                    attention_probs,
                    self.att_prob_clip,
                    self.att_prob_clip2,
                    self.att_prob_clip3,
                    1,
                )
            else:
                attention_probs = self.att_prob_quantizer(attention_probs * 3)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # BHNC1 -> BNHC1
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,
        )  #  BNHC1 -> BN+ C
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = (
            self.norm_context(context_layer)
            + mixed_query_layer
            + mixed_key_layer
            + mixed_value_layer
        )
        context_layer = self.rprelu_context(context_layer)
        context_layer = self.token_split(context_layer, split_dim, H, W, B)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class BHViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.dense = QuantizeLinear(
            config.hidden_size[config.stages[layer_num]],
            config.hidden_size[config.stages[layer_num]],
            config=config,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.move = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.norm = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])

        self.layerscale = (
            LayerScale(config.hidden_size[config.stages[layer_num]])
            if not config.disable_layerscale
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        out = self.norm(self.dense(hidden_states + self.move)) + hidden_states
        out = self.rprelu(out)
        out = self.dropout(out)

        out = self.layerscale(out)

        return out


class BHViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.attention = BHViTSelfAttention(config, layer_num)
        self.output = BHViTSelfOutput(config, layer_num)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(
            self_outputs[0].permute(0, 2, 3, 1).flatten(1, 2)
        )

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig, layer_num) -> None:
        super().__init__()
        self.dense = QuantizeLinear(
            config.hidden_size[config.stages[layer_num]],
            config.intermediate_size[config.stages[layer_num]],
            config=config,
        )

        self.move = nn.Parameter(
            torch.zeros(config.hidden_size[config.stages[layer_num]])
        )
        self.norm = config.norm_layer(
            config.intermediate_size[config.stages[layer_num]],
            eps=config.layer_norm_eps,
        )
        self.rprelu = RPReLU(config.intermediate_size[config.stages[layer_num]])
        self.expansion_ratio = (
            config.intermediate_size[config.stages[layer_num]]
            // config.hidden_size[config.stages[layer_num]]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        out = self.norm(self.dense(hidden_states + self.move)) + torch.concat(
            [hidden_states for _ in range(self.expansion_ratio)], dim=-1
        )
        out = self.rprelu(out)
        # out = self.intermediate_act_fn(out)

        return out


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig, layer_num, drop_path=0.0) -> None:
        super().__init__()
        self.dense = QuantizeLinear(
            config.intermediate_size[config.stages[layer_num]],
            config.hidden_size[config.stages[layer_num]],
            config=config,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.move = nn.Parameter(
            torch.zeros(config.intermediate_size[config.stages[layer_num]])
        )
        self.norm = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.rprelu = RPReLU(config.hidden_size[config.stages[layer_num]])
        self.pooling = nn.AvgPool1d(
            config.intermediate_size[config.stages[layer_num]]
            // config.hidden_size[config.stages[layer_num]]
        )
        self.layerscale = (
            LayerScale(config.hidden_size[config.stages[layer_num]])
            if not config.disable_layerscale
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.dense(hidden_states + self.move)) + self.pooling(
            hidden_states
        )
        out = self.rprelu(out)
        out = self.dropout(out)

        out = self.layerscale(out)

        out = self.drop_path(out)

        return out


class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class Shift(nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, dim):
        x1 = torch.roll(x, 1, dims=dim)  # [:,:,1:,:]
        x2 = torch.roll(x, -1, dims=dim)  # [:,:,:-1,:]
        x = x + x1 + x2
        return x / 3


class Shift2(nn.Module):
    def __init__(self):
        super(Shift2, self).__init__()

    def forward(self, x, dim):
        x1 = torch.roll(x, 1, dims=dim)  # [:,:,1:,:]
        x2 = torch.roll(x, -1, dims=dim)  # [:,:,:-1,:]
        x3 = torch.roll(x, 2, dims=dim)  # [:,:,1:,:]
        x4 = torch.roll(x, -2, dims=dim)  # [:,:,:-1,:]
        x = x + x1 + x2 + x3 + x4
        return x / 5


class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x


# class token_mixer2(nn.Module):
#     def __init__(self, in_chn,config, pool=2,kernel_size=3, stride=1, padding='same'):
#         super(token_mixer2, self).__init__()
#         self.move = LearnableBiasnn(in_chn)
#         self.cov1 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding,1,4,bias=True,config=config )
#         self.pool1 = nn.MaxPool2d(pool, stride=pool)
#         self.cov2 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding,1,4,bias=True,config=config)
#         self.pool2 = nn.AvgPool2d(pool, stride=pool)
#         self.cov3 = QuantizeConv2d(in_chn, in_chn, kernel_size, stride, padding,1,4,bias=True,config=config )
#         self.norm = config.norm_layer(in_chn, eps=config.layer_norm_eps)
#         self.act1 = RPReLU(in_chn)
#         self.act2 = RPReLU(in_chn)
#         self.act3 = RPReLU(in_chn)
#     def forward(self, x):
#         B,C,H,W = x.shape
#         x = self.move(x)
#         x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1,2)
#         x1 = self.act1(x1)
#         x2 = self.pool1(x)
#         x2 = self.cov2(x)
#         x2 = torch.nn.functional.interpolate(x2, size=H, mode='nearest').permute(0, 2, 3, 1).flatten(1,2)
#         x2 = self.act2(x2)

#         x3 = self.pool2(x)
#         x3 = self.cov3(x)
#         x3 = torch.nn.functional.interpolate(x3, size=H, mode='nearest').permute(0, 2, 3, 1).flatten(1,2)
#         x3 = self.act3(x3)


#         x = self.norm(x1+x2+x3)
#         return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()
class token_mixer(nn.Module):
    def __init__(
        self,
        in_chn,
        config,
        dilation1=1,
        dilation2=3,
        dilation3=5,
        kernel_size=3,
        stride=1,
        padding="same",
    ):
        super(token_mixer, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        self.cov1 = QuantizeConv2d(
            in_chn,
            in_chn,
            kernel_size,
            stride,
            padding,
            dilation1,
            4,
            bias=True,
            config=config,
        )
        self.cov2 = QuantizeConv2d(
            in_chn,
            in_chn,
            kernel_size,
            stride,
            padding,
            dilation2,
            4,
            bias=True,
            config=config,
        )
        self.cov3 = QuantizeConv2d(
            in_chn,
            in_chn,
            kernel_size,
            stride,
            padding,
            dilation3,
            4,
            bias=True,
            config=config,
        )
        self.norm = config.norm_layer(in_chn, eps=config.layer_norm_eps)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1, 2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = self.act2(x2)
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1, 2)
        x3 = self.act3(x3)
        x = self.norm(x1 + x2 + x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()


class GCLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, layer_num, drop_path=0.0) -> None:
        super().__init__()
        self.GC = token_mixer(config.hidden_size[config.stages[layer_num]], config)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.config = config
        self.intermediate = ViTIntermediate(config, layer_num)
        self.output = ViTOutput(config, layer_num, drop_path=drop_path)
        self.layernorm_before = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.layernorm_after = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )

        self.shift3 = config.shift3
        self.shift5 = config.shift5

        if self.shift5:
            print("Using shift 5 Residual")
            # shift_window = 5
            self.shift_w5 = Shift2()
            self.layerscale_w5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_h5 = Shift2()
            self.layerscale_h5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_ch5 = Shift_channel_mix(2)
            self.layerscale_ch5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
        if self.shift3:
            print("Using shift 3 Residual")
            # shift_window = 3
            self.shift_w3 = Shift()
            self.layerscale_w3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_h3 = Shift()
            self.layerscale_h3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_ch3 = Shift_channel_mix(1)
            self.layerscale_ch3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )

    def forward(self, hidden_states: torch.Tensor):

        B, C, H, W = hidden_states.shape
        hidden_states_norm = self.layernorm_before(
            hidden_states.permute(0, 2, 3, 1).flatten(1, 2)
        )
        self_GC_outputs = self.GC(
            hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()
        )
        # first residual connection
        hidden_states = self_GC_outputs + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.layernorm_after(
            hidden_states.permute(0, 2, 3, 1).flatten(1, 2)
        )
        layer_output = self.intermediate(hidden_states_norm)

        # second residual connection is done here
        layer_output = (
            (
                self.output(layer_output)
                .permute(0, 2, 1)
                .view(-1, C, H, W)
                .contiguous()
                + hidden_states
            )
            .view(-1, C, H * W)
            .permute(0, 2, 1)
            .contiguous()
        )
        if self.shift3:
            layer_output += self.layerscale_h3(
                self.shift_h3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    2,
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_w3(
                self.shift_w3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    3,
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_ch3(
                self.shift_ch3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
        if self.shift5:
            layer_output += self.layerscale_h5(
                self.shift_h5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    2,
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_w5(
                self.shift_w5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    3,
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_ch5(
                self.shift_ch5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()
                )
                .view(-1, C, H * W)
                .permute(0, 2, 1)
                .contiguous()
            )
        outputs = layer_output.permute(0, 2, 1).view(-1, C, H, W).contiguous()

        return outputs


class BHViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, layer_num, drop_path=0.0) -> None:
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BHViTAttention(config, layer_num)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.intermediate = ViTIntermediate(config, layer_num)
        self.output = ViTOutput(config, layer_num, drop_path=drop_path)
        self.layernorm_before = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )
        self.layernorm_after = config.norm_layer(
            config.hidden_size[config.stages[layer_num]], eps=config.layer_norm_eps
        )

        self.shift3 = config.shift3
        self.shift5 = config.shift5

        if self.shift5:
            print("Using shift 5 Residual")
            # shift_window = 5
            self.shift_w5 = Shift2()
            self.layerscale_w5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_h5 = Shift2()
            self.layerscale_h5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_ch5 = Shift_channel_mix(2)
            self.layerscale_ch5 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
        if self.shift3:
            print("Using shift 3 Residual")
            # shift_window = 3
            self.shift_w3 = Shift()
            self.layerscale_w3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_h3 = Shift()
            self.layerscale_h3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )
            self.shift_ch3 = Shift_channel_mix(1)
            self.layerscale_ch3 = LayerScale(
                config.hidden_size[config.stages[layer_num]], init_ones=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).flatten(1, 2)

        hidden_states_norm = self.layernorm_before(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states_norm.permute(0, 2, 1)
            .view(-1, C, H, W)
            .contiguous(),  # in ViT, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(hidden_states_norm)

        # second residual connection is done here
        layer_output = self.output(layer_output) + hidden_states
        B, N, C = hidden_states_norm.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        if self.shift3:
            layer_output += self.layerscale_h3(
                self.shift_h3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    2,
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_w3(
                self.shift_w3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    3,
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_ch3(
                self.shift_ch3(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
        if self.shift5:
            layer_output += self.layerscale_h5(
                self.shift_h5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    2,
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_w5(
                self.shift_w5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous(),
                    3,
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
            layer_output += self.layerscale_ch5(
                self.shift_ch5(
                    hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()
                )
                .view(-1, C, N)
                .permute(0, 2, 1)
                .contiguous()
            )
        layer_output = layer_output.permute(0, 2, 1).view(-1, C, H, W).contiguous()
        outputs = (layer_output,) + outputs

        return outputs


class BinaryPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0], img_size[1]
        self.num_patches = self.H * self.W

        self.norm0 = config.norm_layer(in_dim)

        self.move = nn.Parameter(torch.zeros(1, in_dim, 1, 1))
        self.proj = QuantizeConv2d(
            in_dim, out_dim, self.patch_size, self.patch_size, bias=False, config=config
        )
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.norm = config.norm_layer(out_dim)
        self.rprelu = RPReLU(out_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, img_size[0] // 2 * img_size[0] // 2, out_dim)
        )
        trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, hidden_states):
        if len(hidden_states.shape) == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = self.norm0(hidden_states.permute(0, 2, 3, 1).flatten(1, 2))
        else:
            B, N, C = hidden_states.shape
            hidden_states = self.norm0(hidden_states)

        residual = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        residual = self.pool(residual).reshape(B, C, -1).permute(0, 2, 1).contiguous()

        hidden_states = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        hidden_states = self.proj(hidden_states + self.move.expand_as(hidden_states))
        B2, C2, H2, W2 = hidden_states.shape
        hidden_states = hidden_states.reshape(B2, C2, -1).permute(0, 2, 1).contiguous()

        residual = torch.concat([residual for _ in range(C2 // C)], dim=-1)

        hidden_states = self.norm(hidden_states) + residual
        hidden_states = self.rprelu(hidden_states)

        return (
            (hidden_states + self.position_embeddings)
            .permute(0, 2, 1)
            .reshape(B, 2 * C, self.H // 2, self.W // 2)
        )


################################################################################################
# class BinaryPatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)

#         self.img_size = img_size
#         self.patch_size = patch_size
#         # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
#         #     f"img_size {img_size} should be divided by patch_size {patch_size}."
#         self.H, self.W = img_size[0], img_size[1]
#         self.num_patches = self.H * self.W

#         self.norm0 = config.norm_layer(in_dim)

#         self.move = nn.Parameter(torch.zeros(1, in_dim, 1, 1))
#         self.proj = QuantizeConv2d2(in_dim, out_dim, self.patch_size, self.patch_size, bias=False, config=config)
#         self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
#         self.norm = config.norm_layer(out_dim)
#         self.rprelu = RPReLU(out_dim)

#         self.position_embeddings = nn.Parameter(torch.zeros(1, img_size[0]//2 * img_size[0]//2, out_dim))
#         trunc_normal_(self.position_embeddings, std=.02)


#     def forward(self, hidden_states):
#         if len(hidden_states.shape) ==4 :
#             B,C,H,W= hidden_states.shape
#             hidden_states = self.norm0(hidden_states.permute(0, 2, 3, 1).flatten(1,2))
#         else:
#             B,N,C= hidden_states.shape
#             hidden_states = self.norm0(hidden_states)

#         residual = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
#         residual = self.pool(residual).reshape(B, C, -1).permute(0, 2, 1).contiguous()

#         hidden_states = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
#         hidden_states = self.proj(hidden_states + self.move.expand_as(hidden_states))
#         B2, C2, H2, W2 = hidden_states.shape
#         hidden_states = hidden_states.reshape(B2, C2, -1).permute(0, 2, 1).contiguous()

#         residual = torch.concat([residual for _ in range(C2 // C)], dim=-1)

#         hidden_states = self.norm(hidden_states) + residual
#         hidden_states = self.rprelu(hidden_states)

#         return (hidden_states + self.position_embeddings).permute(0, 2, 1).reshape(B, 2*C, self.H//2, self.W//2)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=2, in_dim=3, out_dim=64, config=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0], img_size[1]
        self.num_patches = self.H * self.W

        self.norm0 = config.norm_layer(in_dim)
        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.norm = config.norm_layer(out_dim)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, img_size[0] // 2 * img_size[0] // 2, out_dim)
        )
        trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, hidden_states):
        if len(hidden_states.shape) == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = self.norm0(hidden_states.permute(0, 2, 3, 1).flatten(1, 2))
        else:
            B, N, C = hidden_states.shape
            hidden_states = self.norm0(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        hidden_states = self.proj(hidden_states).flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)

        return (
            (hidden_states + self.position_embeddings)
            .permute(0, 2, 1)
            .reshape(B, 2 * C, self.H // 2, self.W // 2)
        )


class BHViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                config.drop_path,
                config.num_hidden_layersA + config.num_hidden_layersB,
            )
        ]
        self.layerA = nn.ModuleList(
            [
                GCLayer(config, i, drop_path=dpr[i])
                for i in range(config.num_hidden_layersA)
            ]
        )
        self.layerB = nn.ModuleList(
            [
                BHViTLayer(
                    config,
                    i + config.num_hidden_layersA,
                    drop_path=dpr[i + config.num_hidden_layersA],
                )
                for i in range(config.num_hidden_layersB)
            ]
        )
        self.gradient_checkpointing = False

        if config.some_fp:
            self.patch_embed1 = PatchEmbed(
                56,
                in_dim=config.hidden_size[0],
                out_dim=config.hidden_size[1],
                config=config,
            )
            self.patch_embed2 = PatchEmbed(
                28,
                in_dim=config.hidden_size[1],
                out_dim=config.hidden_size[2],
                config=config,
            )
            self.patch_embed3 = PatchEmbed(
                14,
                in_dim=config.hidden_size[2],
                out_dim=config.hidden_size[3],
                config=config,
            )
        else:
            self.patch_embed1 = BinaryPatchEmbed(
                56,
                in_dim=config.hidden_size[0],
                out_dim=config.hidden_size[1],
                config=config,
            )
            self.patch_embed2 = BinaryPatchEmbed(
                28,
                in_dim=config.hidden_size[1],
                out_dim=config.hidden_size[2],
                config=config,
            )
            self.patch_embed3 = BinaryPatchEmbed(
                14,
                in_dim=config.hidden_size[2],
                out_dim=config.hidden_size[3],
                config=config,
            )

        self.depths = config.depths

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        ##### stage 12
        for i, layer_module in enumerate(self.layerA):

            layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs

            if i == self.depths[0] - 1:
                hidden_states = self.patch_embed1(hidden_states)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            elif i == self.depths[0] + self.depths[1] - 1:
                hidden_states = self.patch_embed2(hidden_states)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        ##### stage 34
        for i, layer_module in enumerate(self.layerB):

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if i == self.depths[2] - 1:
                hidden_states = self.patch_embed3(hidden_states)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BHViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.embeddings = BHViTEmbeddings(config)
        self.encoder = BHViTEncoder(config)

        self.layernorm = config.norm_layer(
            config.hidden_size[3], eps=config.layer_norm_eps
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(
            sequence_output.permute(0, 2, 3, 1).flatten(1, 2)
        )

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def generating_stage_per_depth(depths):
    i = 0
    stage_per_depth = []
    current_stage_depth = depths[i]
    while True:
        current_stage_depth -= 1
        stage_per_depth.append(i)
        if current_stage_depth == 0:
            i += 1
            if i == len(depths):
                break
            current_stage_depth = depths[i]
    return stage_per_depth


class TokenPooling(nn.Module):
    def __init__(self, in_dim: int, mode: str = "mean", topk_ratio: float = 1.0):
        super().__init__()
        self.mode = mode
        self.topk_ratio = topk_ratio
        if mode == "attn":
            self.att = nn.Linear(in_dim, 1)

    def forward(self, x):  # x: [B, N, C]
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "attn":
            a = self.att(x)  # [B, N, 1]
            w = F.softmax(a, dim=1)
            return (w * x).sum(dim=1)  # [B, C]
        # top-k by token energy
        scores = x.norm(dim=-1)  # [B, N]
        k = max(1, int(x.size(1) * self.topk_ratio))
        idx = scores.topk(k, dim=1).indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        topk_x = x.gather(1, idx)  # [B, k, C]
        return topk_x.mean(dim=1)


class BHViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.num_labels = config.num_labels
        config.num_hidden_layers = sum(config.depths)
        config.stages = generating_stage_per_depth(config.depths)

        self.vit = BHViTModel(config)
        self.config = config

        # Classifier head;token pooling (default "mean" to keep old behavior)
        pool_mode = getattr(config, "token_pool", "mean")
        topk_ratio = float(getattr(config, "topk_ratio", 1.0))
        self.pool = TokenPooling(
            config.hidden_size[3], mode=pool_mode, topk_ratio=topk_ratio
        )

        self.classifier = (
            nn.Linear(config.hidden_size[3], config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(module: nn.Module, name: str = ""):
        """ViT weight initialization, original timm impl (for reproducibility)"""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(torch.mean(sequence_output, dim=1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"position_embeddings", "cls_token", "dist_token"}
