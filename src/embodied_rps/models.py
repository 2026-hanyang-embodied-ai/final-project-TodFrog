"""PyTorch classifiers for skeleton RPS sequence classification."""

from __future__ import annotations

import math

import torch
from torch import nn

from embodied_rps.training_types import ModelRunConfig


class RpsClassifier(nn.Module):
    """Base class for typed RPS classifiers."""

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return pre-logit embeddings for auxiliary objectives."""

        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of observed sequences."""

        raise NotImplementedError


class MlpClassifier(RpsClassifier):
    """Static baseline using the last observed frame."""

    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = inputs[:, -1, :]
        for layer in list(self.network.children())[:-1]:
            encoded = layer(encoded)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("MLP embedding must be a tensor")
        return encoded

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.network(inputs[:, -1, :])
        if not isinstance(output, torch.Tensor):
            raise TypeError("MLP output must be a tensor")
        return output


class GruClassifier(RpsClassifier):
    """GRU sequence baseline."""

    def __init__(self, input_dim: int, hidden_size: int, layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        last_hidden = hidden[-1]
        if not isinstance(last_hidden, torch.Tensor):
            raise TypeError("GRU embedding must be a tensor")
        return last_hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        last_hidden = self.encode(inputs)
        output = self.head(last_hidden)
        if not isinstance(output, torch.Tensor):
            raise TypeError("GRU output must be a tensor")
        return output


class TcnClassifier(RpsClassifier):
    """Small causal-ish 1D temporal convolution classifier."""

    def __init__(self, input_dim: int, channels: int, kernel_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size - 1
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=kernel_size, padding=padding, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding * 2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, num_classes)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        encoded = self.network(channels_first).squeeze(-1)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("TCN embedding must be a tensor")
        return encoded

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(inputs)
        output = self.head(encoded)
        if not isinstance(output, torch.Tensor):
            raise TypeError("TCN output must be a tensor")
        return output


class TinyTransformerClassifier(RpsClassifier):
    """Tiny Transformer encoder baseline for short skeleton sequences."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        dim: int,
        heads: int,
        layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, num_classes)
        self._reset_position()

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)
        encoded = self.encoder(projected + self.position[:, : inputs.shape[1], :])
        pooled = encoded.mean(dim=1)
        if not isinstance(pooled, torch.Tensor):
            raise TypeError("Transformer embedding must be a tensor")
        return pooled

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(inputs)
        output = self.head(pooled)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Transformer output must be a tensor")
        return output

    def _reset_position(self) -> None:
        with torch.no_grad():
            sequence_length = int(self.position.shape[1])
            dim = int(self.position.shape[2])
            positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
            self.position[:, :, 0::2] = torch.sin(positions * div_term)
            self.position[:, :, 1::2] = torch.cos(positions * div_term[: self.position[:, :, 1::2].shape[2]])


class StGcnBlock(nn.Module):
    """Small spatial-temporal graph convolution block over hand joints."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.temporal = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("bctv,vw->bctw", inputs, adjacency)
        output = self.temporal(propagated)
        if not isinstance(output, torch.Tensor):
            raise TypeError("ST-GCN temporal output must be a tensor")
        activated = self.activation(output)
        dropped = self.dropout(activated)
        if not isinstance(dropped, torch.Tensor):
            raise TypeError("ST-GCN block output must be a tensor")
        return dropped


class StGcnClassifier(RpsClassifier):
    """Small ST-GCN baseline for the current five-node curl-joint hand graph."""

    def __init__(self, input_dim: int, channels: int, layers: int, kernel_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("ST-GCN expects position and velocity features for each joint")
        joint_count = input_dim // 2
        self.joint_count = joint_count
        blocks: list[StGcnBlock] = []
        in_channels = 2
        for _ in range(layers):
            blocks.append(StGcnBlock(in_channels, channels, kernel_size, dropout))
            in_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(channels, num_classes)
        self.register_buffer("adjacency", _normalized_hand_adjacency(joint_count), persistent=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0])
        sequence_length = int(inputs.shape[1])
        positions = inputs[:, :, : self.joint_count]
        velocities = inputs[:, :, self.joint_count : self.joint_count * 2]
        graph_input = torch.stack((positions, velocities), dim=1).reshape(
            batch_size,
            2,
            sequence_length,
            self.joint_count,
        )
        adjacency = self.adjacency
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("ST-GCN adjacency must be a tensor")
        encoded = graph_input
        for block in self.blocks:
            encoded = block(encoded, adjacency)
        pooled = encoded.mean(dim=(2, 3))
        if not isinstance(pooled, torch.Tensor):
            raise TypeError("ST-GCN embedding must be a tensor")
        return pooled

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(inputs)
        output = self.head(pooled)
        if not isinstance(output, torch.Tensor):
            raise TypeError("ST-GCN output must be a tensor")
        return output


def _normalized_hand_adjacency(joint_count: int) -> torch.Tensor:
    adjacency = torch.eye(joint_count, dtype=torch.float32)
    for index in range(joint_count - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_matrix = torch.diag(degree_inv_sqrt)
    normalized = degree_matrix @ adjacency @ degree_matrix
    if not isinstance(normalized, torch.Tensor):
        raise TypeError("normalized adjacency must be a tensor")
    return normalized


def build_classifier(
    config: ModelRunConfig,
    *,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
) -> RpsClassifier:
    """Build one configured classifier."""

    if config.model == "mlp":
        return MlpClassifier(input_dim, config.hidden_size, num_classes, config.dropout)
    if config.model == "gru":
        return GruClassifier(input_dim, config.hidden_size, config.layers, num_classes, config.dropout)
    if config.model == "tcn":
        return TcnClassifier(input_dim, config.hidden_size, config.kernel_size, num_classes, config.dropout)
    if config.model == "transformer":
        return TinyTransformerClassifier(
            input_dim,
            sequence_length,
            config.hidden_size,
            config.heads,
            config.layers,
            num_classes,
            config.dropout,
        )
    if config.model == "stgcn":
        return StGcnClassifier(
            input_dim,
            config.hidden_size,
            config.layers,
            config.kernel_size,
            num_classes,
            config.dropout,
        )
    raise ValueError(f"Unsupported model: {config.model}")


def parameter_count(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
