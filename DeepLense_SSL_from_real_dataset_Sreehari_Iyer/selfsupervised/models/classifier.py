import torch.nn as nn
from models.backbone import Backbone

class Classifier(nn.Module):
    def __init__(self, mode = "linear_probe", backbone = None, backbone_return_all_tokens: bool = False):
        super().__init__()
        if mode == "linear_probe":
            self.freeze_backbone = True
        elif mode == "finetune":
            self.freeze_backbone = False
        else:
            raise NotImplementedError(f"unknown mode {mode}")
        self.backbone = backbone
        self.fc = nn.Linear(self.backbone.embed_dim, 2)
        self.fc.apply(self._init_weights)
        if self.freeze_backbone:
            for params in self.backbone.parameters():
                params.requires_grad = False
        else:
            for params in self.backbone.parameters():
                params.requires_grad = True
        for params in self.fc.parameters():
            params.requires_grad = True
        self.backbone_return_all_tokens = backbone_return_all_tokens
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                if self.backbone_return_all_tokens:
                    out = self.backbone(x)[:, 0]
                else:
                    out = self.backbone(x)
            return self.fc(out)
        else:
            if self.backbone_return_all_tokens:
                out = self.backbone(x)[:, 0]
            else:
                out = self.backbone(x)
            return self.fc(out)