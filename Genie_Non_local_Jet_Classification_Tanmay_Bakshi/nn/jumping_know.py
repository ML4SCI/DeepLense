import torch
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch_geometric.nn import JumpingKnowledge as JK, Linear


class SelfAttention(MultiheadAttention):
    def forward(self, xs: Tensor) -> Tensor:
        """forward propagation

        Args:
            xs (Tensor): input with shape (batch_size, hidden_dim, num_phases)

        Returns:
            Tensor: output tensor with size (num_nodes, hidden_dim)
        """
        x = xs.transpose(2, int(self.batch_first))
        out: Tensor = super().forward(x, x, x, need_weights=True)[0]
        return out.mean(dim=int(self.batch_first))

    def reset_parameters(self):
        super()._reset_parameters()


class WeightedSum(Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.Q = Linear(in_channels=hidden_dim, out_channels=num_heads, bias=False)

        if num_heads > 1:
            self.fc = Linear(in_channels=num_heads, out_channels=1, bias=False)

    def forward(self, xs: Tensor) -> Tensor:
        """forward propagation

        Args:
            xs (Tensor): input with shape (batch_size, hidden_dim, num_phases)

        Returns:
            Tensor: output tensor with size (num_nodes, hidden_dim)
        """
        H = xs.transpose(1, 2)  # (node, hop, dim)
        W = self.Q(H).softmax(dim=1)  # (node, hop, head)
        Z = H.transpose(1, 2).matmul(W)

        if self.num_heads > 1:
            Z = self.fc(Z)

        return Z.squeeze(-1)

    def reset_parameters(self):
        self.Q.reset_parameters()
        if self.num_heads > 1:
            self.fc.reset_parameters()


class JumpingKnowledge(Module):
    supported_modes = ['cat', 'max', 'lstm', 'sum', 'mean', 'attn', 'wsum']
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        if mode == 'attn':
            self.hidden_dim = kwargs['hidden_dim']
            self.num_heads = kwargs['num_heads']
            self.attn = SelfAttention(self.hidden_dim, num_heads=self.num_heads, batch_first=True)
        elif mode == 'wsum':
            self.hidden_dim = kwargs['hidden_dim']
            self.num_heads = kwargs['num_heads']
            self.wsum = WeightedSum(self.hidden_dim, num_heads=self.num_heads)
        elif mode == 'lstm':
            self.lstm = JK(mode='lstm', **kwargs)

    def forward(self, xs: Tensor) -> Tensor:
        """forward propagation

        Args:
            xs (Tensor): input with shape (batch_size, hidden_dim, num_phases)

        Returns:
            Tensor: aggregated output with shape (batch_size, hidden_dim)
        """
        if self.mode == 'cat':
            return xs.transpose(1,2).reshape(xs.size(0), -1)
        elif self.mode == 'sum':
            return xs.sum(dim=-1)
        elif self.mode == 'mean':
            return xs.mean(dim=-1)
        elif self.mode == 'max':
            return xs.max(dim=-1)[0]
        elif self.mode == 'attn':
            return self.attn(xs)
        elif self.mode == 'wsum':
            return self.wsum(xs)
        elif self.mode == 'lstm':
            return self.lstm(xs.unbind(dim=-1))
        else:
            raise NotImplementedError(f'Unsupported JK mode: {self.mode}')

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()