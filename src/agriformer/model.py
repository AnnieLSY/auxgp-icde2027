"""AgriFormer architecture used by the unified reproduction entry point."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HolisticMixerBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, values):
        return self.norm(values + self.mlp(values))


class HolisticFeatureMixer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, embed_dim=128, hidden_factor=1.5, dropout=0.15):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mixer = HolisticMixerBlock(embed_dim, dropout)
        hidden = int(embed_dim * hidden_factor)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim)
        )

    def forward(self, values):
        embedded = self.dropout(self.norm(self.embedding(values)))
        return self.output(self.mixer(embedded))


class LocalConvolutionExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, channels=64, hidden_factor=1.5, dropout=0.02):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        hidden = int(channels * hidden_factor)
        self.output = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dim)
        )

    def forward(self, values):
        pooled = self.pool(self.stack(values.unsqueeze(1))).squeeze(-1)
        return self.output(pooled)


class LinearAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        batch, length, width = values.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = self.to_q(values).view(shape).permute(0, 2, 1, 3)
        key = self.to_k(values).view(shape).permute(0, 2, 1, 3)
        value = self.to_v(values).view(shape).permute(0, 2, 1, 3)
        query = (F.elu(query) + 1.0) * self.scale
        key = F.elu(key) + 1.0
        key_value = torch.einsum("bhjd,bhje->bhde", key, value)
        output = torch.einsum("bhld,bhde->bhle", query, key_value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch, length, width)
        return self.dropout(self.to_out(output))


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, feedforward_factor: int):
        super().__init__()
        self.attention = LinearAttention(width, heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * feedforward_factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * feedforward_factor, width), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        values = values + self.dropout(self.attention(self.norm1(values)))
        return values + self.dropout(self.feedforward(self.norm2(values)))


class GlobalRelationAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width=128, heads=8, layers=2, dropout=0.05, feedforward_factor=2):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, input_dim, width))
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            AttentionBlock(width, heads, dropout, feedforward_factor) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, values):
        sequence = self.dropout(self.norm(self.embedding(values.unsqueeze(-1)) + self.position))
        for block in self.blocks:
            sequence = block(sequence)
        return self.output(self.final_norm(sequence.mean(dim=1)))


def _indices(grouping, name):
    return sorted({int(index) for index in grouping.get(name, [])})


class AgriFormer(nn.Module):
    """Three-branch model with learned fusion and trait-grouped output heads."""

    def __init__(self, context_dim: int, effect_dim: int, output_dim: int, task_groups: dict[str, list[int]]):
        super().__init__()
        self.context = HolisticFeatureMixer(context_dim, output_dim)
        self.convolution = LocalConvolutionExtractor(effect_dim, output_dim)
        self.attention = GlobalRelationAttention(effect_dim, output_dim)
        self.fusion_logits = nn.Parameter(torch.tensor([0.293, 0.410, 0.297]).log())
        self.shared = nn.Sequential(nn.Linear(output_dim, 128), nn.ReLU(), nn.Dropout(0.5))
        self.output_dim = output_dim
        self.high_indices = _indices(task_groups, "high")
        self.moderate_indices = _indices(task_groups, "moderate")
        self.low_indices = _indices(task_groups, "low")
        self.independent_indices = _indices(task_groups, "independent")
        assigned = self.high_indices + self.moderate_indices + self.low_indices + self.independent_indices
        if sorted(assigned) != list(range(output_dim)):
            raise ValueError("task groups must assign each output exactly once")
        if self.high_indices:
            self.high_shared = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            self.high_output = nn.Linear(64, len(self.high_indices))
        if self.moderate_indices:
            self.moderate_shared = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            self.moderate_output = nn.ModuleDict({str(i): nn.Linear(64, 1) for i in self.moderate_indices})
        if self.low_indices:
            self.low_output = nn.ModuleDict({str(i): nn.Linear(128, 1) for i in self.low_indices})
        if self.independent_indices:
            self.independent_output = nn.ModuleDict({str(i): nn.Linear(128, 1) for i in self.independent_indices})

    def forward(self, context, effects):
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = (
            weights[0] * self.context(context)
            + weights[1] * self.convolution(effects)
            + weights[2] * self.attention(effects)
        )
        shared = self.shared(fused)
        prediction = torch.empty((len(context), self.output_dim), device=context.device, dtype=context.dtype)
        if self.high_indices:
            prediction[:, self.high_indices] = self.high_output(self.high_shared(shared))
        if self.moderate_indices:
            moderate = self.moderate_shared(shared)
            for index in self.moderate_indices:
                prediction[:, index] = self.moderate_output[str(index)](moderate).squeeze(-1)
        for index in self.low_indices:
            prediction[:, index] = self.low_output[str(index)](shared).squeeze(-1)
        for index in self.independent_indices:
            prediction[:, index] = self.independent_output[str(index)](shared).squeeze(-1)
        return prediction
