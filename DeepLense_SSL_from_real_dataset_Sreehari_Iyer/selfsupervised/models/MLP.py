import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
            self, 
            backbone: nn.Module, 
            embed_dim: int=384,
            output_dim: int = 1,
            freeze_backbone: bool = False,
        ):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, output_dim)
        self.backbone = backbone.cuda()
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad = False

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x)
            return self.fc(x)
        x = self.backbone(x)
        return self.fc(x)