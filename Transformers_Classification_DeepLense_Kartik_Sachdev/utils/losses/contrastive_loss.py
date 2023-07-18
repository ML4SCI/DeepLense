import torch
import torch.nn as nn


# Define the contrastive loss function
class ContrastiveLossCrossEntropy(nn.Module):
    def __init__(self, temperature, device):
        super(ContrastiveLossCrossEntropy, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)

        positives = similarity_matrix[~mask].view(batch_size, -1)
        negatives = similarity_matrix[mask].view(batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.arange(batch_size).to(self.device)

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

# Define the contrastive loss function
class ContrastiveLossEuclidean(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# Define the contrastive loss function
class ContrastiveLossEmbedding(nn.Module):
    def __init__(self, temperature, device):
        super(ContrastiveLossEmbedding, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        
        similarities = similarity_matrix[~mask].view(batch_size, -1)
        
        logits = similarities / self.temperature
        
        loss = -torch.log(torch.exp(logits).sum(1) / torch.exp(similarity_matrix / self.temperature).sum(1)).mean()
        return loss