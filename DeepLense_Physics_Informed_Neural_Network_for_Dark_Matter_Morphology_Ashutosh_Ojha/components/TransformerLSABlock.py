import torch
from torch import nn


class TransformerLSABlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        num_patches: int,
        num_hidden_neurons: int,
        num_hidden_layers: int,
        activation_function: nn.Module,
        device: torch.device,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device
        self.mlsa = MultiLocallySelfAttention(
            embedding_dim, num_head, num_patches, device, dropout
        )
        self.first_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForwardBlock(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            hidden_dim=num_hidden_neurons,
            num_hidden_layers=num_hidden_layers,
            activation_function=activation_function,
            task_type="regression",
            dropout=dropout,
        )
        self.second_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.temperature = nn.Parameter(torch.ones(1))

        self.batch_size = 64
        self.query_T = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_T = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key.shape[0]
        key = key.to(self.device)

        if query is None:
            query = self.query_T(key)
            query = (query / self.temperature).to(self.device)
        if value is None:
            value = self.value_T(key)

        value = value + self.mlsa(key, query, value)
        value = self.first_norm(value)
        value = value + self.feedforward(value)
        value = self.second_norm(value)
        value = self.dropout(value)

        return value
