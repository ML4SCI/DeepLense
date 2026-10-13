import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(
        self, blocks: list[tuple[int, int, nn.Module | None, nn.Module | None]]
    ):
        super().__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class BYOLProjectionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super().__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BYOLPredictionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super().__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )
