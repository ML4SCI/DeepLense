from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
import torch.nn as nn
import torch

class SimSiamTransformer(nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(input_dim, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

