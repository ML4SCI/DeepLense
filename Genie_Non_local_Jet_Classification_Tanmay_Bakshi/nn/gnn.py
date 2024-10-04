from typing import Callable
import torch
from torch import Tensor
from torch.nn import Dropout, Module, BatchNorm1d
from torch_geometric.nn import GraphSAGE, GCN, GAT


class GNN(Module):
    """
    A flexible GNN model.
    This implementation supports plain_last option.
    """

    def __init__(self, *,
                 conv: str,
                 output_dim: int,
                 hidden_dim: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 batch_norm: bool = False,
                 jk: str = None,
                 plain_last: bool = True,
                 **conv_kwargs
                 ):

        super().__init__()

        model_kwargs = dict(
            in_channels=-1,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_layers,
            act=activation_fn,
            dropout=dropout,
            norm='batchnorm' if batch_norm else None,
            jk=jk,
        )

        if conv == 'sage':
            self.model = GraphSAGE(
                project=False,
                **model_kwargs,
                **conv_kwargs,
            )
        elif conv == 'gcn':
            self.model = GCN(
                **model_kwargs,
                **conv_kwargs,
            )
        elif conv == 'gat':
            self.model = GAT(
                **model_kwargs,
                **conv_kwargs,
            )
        else:
            raise NotImplementedError(f'Unknown conv type: {conv}')

        self.dropout_fn = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.plain_last = plain_last
        if not plain_last and batch_norm:
            self.bn = BatchNorm1d(output_dim)

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = self.model(x, adj_t)
        if not self.plain_last:
            x = self.bn(x) if self.batch_norm else x
            x = self.dropout_fn(x)
            x = self.activation_fn(x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()
        if hasattr(self, 'bn'):
            self.bn.reset_parameters()