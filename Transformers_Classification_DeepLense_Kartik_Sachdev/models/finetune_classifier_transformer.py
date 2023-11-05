import torch.nn as nn

class FinetuneClassifierTransformer(nn.Module):
    def __init__(self, backbone, head):
        super(FinetuneClassifierTransformer, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        z = self.backbone[0](x).flatten(start_dim=1)
        z = self.backbone[1](z)
        z = self.head(z)
        return z