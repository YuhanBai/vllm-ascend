# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The LongCat-2.5 image encoder, shared RVQ, and language projection."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm

VISION_NORM_EPS = 1e-6
ROTARY_THETA = 10_000.0
RVQ_DISTANCE_CHUNK = 256


class LongCat25VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LongCat25VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attention = MMEncoderAttention(num_heads=num_heads, head_size=self.head_size, scale=self.head_size**-0.5)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        qkv = self.qkv(x).reshape(1, len(x), 3, self.num_heads, self.head_size)
        q, k, v = qkv.unbind(dim=2)

        def rotate(tensor):
            left, right = tensor.float().chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
            return (tensor * cos + rotated * sin).to(tensor.dtype)

        output = self.attention(query=rotate(q), key=rotate(k), value=v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.proj(output.reshape_as(x))


class LongCat25VisionBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.norm1 = RMSNorm(hidden, eps=VISION_NORM_EPS)
        self.norm2 = RMSNorm(hidden, eps=VISION_NORM_EPS)
        self.attn = LongCat25VisionAttention(hidden, config["num_heads"])
        self.mlp = LongCat25VisionMLP(hidden, config["intermediate_size"], bias=True)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm2(x))


class LongCat25PatchEmbedding(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        kernel_size = (config["temporal_patch_size"], config["patch_size"], config["patch_size"])
        self.proj = nn.Conv3d(config["in_channels"], config["hidden_size"], kernel_size, bias=False)

    def forward(self, pixels):
        weight = self.proj.weight
        return F.linear(pixels.to(device=weight.device, dtype=weight.dtype), weight.flatten(1))


def _vision_layout(height: int, width: int, merge: int, window: int):
    """Return merge-group permutation, rotary coordinates, and window boundaries."""
    patch_positions = torch.arange(height * width, device="cpu").reshape(height, width)
    grouped = patch_positions.reshape(height // merge, merge, width // merge, merge)
    grouped = grouped.permute(0, 2, 1, 3).reshape(-1, merge**2)
    rows = torch.arange(height // merge, device="cpu")[:, None]
    columns = torch.arange(width // merge, device="cpu")[None, :]
    window_ids = (rows // window) * math.ceil(width / merge / window) + columns // window
    order = torch.argsort(window_ids.flatten(), stable=True)
    _, counts = torch.unique_consecutive(window_ids.flatten()[order], return_counts=True)
    positions = grouped[order].flatten()
    return order, positions // width, positions % width, (counts * merge**2).tolist()


class LongCat25VisionEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.merge = config["spatial_merge_size"]
        self.window = config["window_size"] // (self.merge * config["patch_size"])
        self.full_attention_layers = frozenset(config["fullatt_block_indexes"])
        self.head_size = config["hidden_size"] // config["num_heads"]
        self.patch_embed = LongCat25PatchEmbedding(config)
        self.blocks = nn.ModuleList(LongCat25VisionBlock(config) for _ in range(config["depth"]))

    def forward(self, pixels: torch.Tensor, grid: list[int]):
        temporal, height, width = grid
        if min(grid) <= 0 or height % self.merge or width % self.merge:
            raise ValueError(f"Invalid LongCat visual patch grid: {grid}.")
        order, rows, columns, lengths = _vision_layout(height, width, self.merge, self.window)
        if temporal > 1:
            groups_per_slice = len(order)
            order = (order[None, :] + torch.arange(temporal, device="cpu")[:, None] * groups_per_slice).flatten()
            rows = rows.repeat(temporal)
            columns = columns.repeat(temporal)
            lengths = lengths * temporal
        hidden = self.patch_embed(pixels)
        device = hidden.device
        order = order.to(device=device, non_blocking=True)
        hidden = hidden.reshape(len(order), self.merge**2, -1)[order].flatten(0, 1)

        axis_size = self.head_size // 2
        inv_freq = ROTARY_THETA ** (-torch.arange(0, axis_size, 2, dtype=torch.float32, device="cpu") / axis_size)
        angles = torch.cat((rows[:, None] * inv_freq, columns[:, None] * inv_freq), dim=-1)
        angles = torch.cat((angles, angles), dim=-1)[None, :, None, :]
        cos, sin = (x.to(device=device, non_blocking=True) for x in (angles.cos(), angles.sin()))
        cu_full = torch.arange(temporal + 1, dtype=torch.int32, device="cpu") * (height * width)
        cu_window = F.pad(torch.tensor(lengths, dtype=torch.int32, device="cpu").cumsum(0, dtype=torch.int32), (1, 0))
        max_full = torch.tensor(height * width, dtype=torch.int32, device="cpu")
        max_window = torch.tensor(max(lengths), dtype=torch.int32, device="cpu")
        for index, block in enumerate(self.blocks):
            cu, max_seq = (cu_full, max_full) if index in self.full_attention_layers else (cu_window, max_window)
            hidden = block(hidden, cos, sin, cu, max_seq)
        return hidden, order


class LongCat25PatchMerger(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"] * config["merge_size"] ** 2
        self.ln_q = RMSNorm(config["hidden_size"], eps=VISION_NORM_EPS)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config["out_hidden_size"]),
        )

    def forward(self, hidden, order):
        merged = self.mlp(self.ln_q(hidden).reshape(-1, self.hidden_size))
        return merged[torch.argsort(order)]


class LongCat25VisualQuantizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if not config["shared_codebook"] or config["quantizer_type"] != "rq":
            raise ValueError("LongCat-2.5 image inference requires the shared residual codebook.")
        self.depth = config["depth"]
        self.codebook = nn.Parameter(torch.empty(config["codebook_size"], config["codebook_dim"], dtype=torch.float32))
        if config["quant_conv"]:
            channels = config["in_channels"]
            self.quant_conv = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, config["codebook_dim"]),
            )
        else:
            self.quant_conv = nn.Identity()

    def forward(self, hidden):
        residual = self.quant_conv(hidden).float()
        codebook = self.codebook
        norms = codebook.square().sum(-1)
        indices = []
        for _ in range(self.depth):
            nearest = [
                (chunk.square().sum(-1, keepdim=True) + norms - (2 * chunk) @ codebook.T).argmin(dim=-1)
                for chunk in residual.split(RVQ_DISTANCE_CHUNK)
            ]
            level = torch.cat(nearest)
            residual = residual - codebook[level]
            indices.append(level)
        return indices


class LongCat25Vision(nn.Module):
    def __init__(self, vision: dict, quantizer: dict):
        super().__init__()
        self.discrete_visualModel = LongCat25VisionEncoder(vision)
        self.discrete_visualBridgeModel = LongCat25PatchMerger(vision)
        self.visual_quantizer = LongCat25VisualQuantizer(quantizer)

    def forward(self, pixels, grid):
        hidden, order = self.discrete_visualModel(pixels, grid)
        return self.visual_quantizer(self.discrete_visualBridgeModel(hidden, order))


class LongCat25VisionProjection(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = LongCat25VisionMLP(hidden_size, intermediate_size, bias=False)

    def forward(self, embeddings):
        return embeddings + self.mlp(self.post_attention_layernorm(embeddings))


class LongCat25VisionBridge(nn.Module):
    def __init__(self, vision: dict, quantizer: dict, hidden_size: int, eps: float):
        super().__init__()
        self.embedding_layers = nn.ModuleList(nn.Embedding(size, hidden_size) for size in quantizer["codebook_sizes"])
        self.transformer_block = LongCat25VisionProjection(hidden_size, vision["bridge_intermediate_size"], eps)

    def forward(self, indices):
        levels = iter(zip(self.embedding_layers, indices, strict=True))
        embedding, code = next(levels)
        hidden = embedding(code)
        for embedding, code in levels:
            hidden = hidden + embedding(code)
        return self.transformer_block(hidden)
