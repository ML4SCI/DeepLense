import torch
import torch.nn as nn
import numpy as np
import warnings
from torch.nn.functional import cosine_similarity

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
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
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

        loss = -torch.log(
            torch.exp(logits).sum(1)
            / torch.exp(similarity_matrix / self.temperature).sum(1)
        ).mean()
        return loss


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = (
            torch.from_numpy(np.array([0] * N))
            .reshape(-1)
            .to(positive_samples.device)
            .long()
        )  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def contrastive_loss(proj_feat1, proj_feat2, temperature=0.5, neg_pairs="all"):
    """
    contrastive_loss(proj_feat1, proj_feat2)

    Returns contrastive loss, given sets of projected features, with positive
    pairs matched along the batch dimension.

    Required args:
    - proj_feat1 (2D torch Tensor): first set of projected features
        (batch_size x feat_size)
    - proj_feat2 (2D torch Tensor): second set of projected features
        (batch_size x feat_size)

    Optional args:
    - temperature (float): relaxation temperature. (default: 0.5)
    - neg_pairs (str or num): If "all", all available negative pairs are used
        for the loss calculation. Otherwise, only a certain number or
        proportion of the negative pairs available in the batch, as specified
        by the parameter, are randomly sampled and included in the
        calculation, e.g. 5 for 5 examples or 0.05 for 5% of negative pairs.
        (default: "all")

    Returns:
    - loss (float): mean contrastive loss
    """

    device = proj_feat1.device

    if len(proj_feat1) != len(proj_feat2):
        raise ValueError(
            f"Batch dimension of proj_feat1 ({len(proj_feat1)}) "
            f"and proj_feat2 ({len(proj_feat2)}) should be same"
        )

    batch_size = len(proj_feat1)  # N
    z1 = nn.functional.normalize(proj_feat1, dim=1)
    z2 = nn.functional.normalize(proj_feat2, dim=1)

    proj_features = torch.cat([z1, z2], dim=0)  # 2N x projected feature dimension
    similarity_mat = nn.functional.cosine_similarity(
        proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
    )  # dim: 2N x 2N

    # initialize arrays to identify sets of positive and negative examples
    pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1)
    neg_sample_indicators = torch.ones(2 * batch_size) - torch.eye(2 * batch_size)

    if neg_pairs != "all":
        # here, positive pairs are NOT included in the negative pairs
        min_val = 1
        max_val = torch.sum(neg_sample_indicators[0]).item() - 1
        if neg_pairs < 0:
            raise ValueError(
                f"Cannot use a negative amount of negative pairs " f"({neg_pairs})."
            )
        elif neg_pairs < 1:
            num_retain = int(neg_pairs * len(neg_sample_indicators))
        else:
            num_retain = int(neg_pairs)

        if num_retain < min_val:
            warnings.warn(
                "Increasing the number of negative pairs to use per "
                f"image in the contrastive loss from {num_retain} to the "
                f"minimum value of {min_val}."
            )
            num_retain = min_val
        elif num_retain > max_val:  # retain all
            num_retain = max_val

        # randomly identify the values to retain for each column
        exclusion_indicators = (
            torch.absolute(1 - neg_sample_indicators) + pos_sample_indicators
        )
        random_values = (
            torch.rand_like(neg_sample_indicators) + exclusion_indicators * 100
        )
        retain_bool = (
            torch.argsort(torch.argsort(random_values, axis=1), axis=1) < num_retain
        )

        neg_sample_indicators *= retain_bool
        if not (torch.sum(neg_sample_indicators, dim=1) == num_retain).all():
            raise NotImplementedError(
                "Implementation error. Not all images "
                f"have been assigned {num_retain} random negative pair(s)."
            )

    numerator = torch.sum(
        torch.exp(similarity_mat / temperature) * pos_sample_indicators.to(device),
        dim=1,
    )

    denominator = torch.sum(
        torch.exp(similarity_mat / temperature) * neg_sample_indicators.to(device),
        dim=1,
    )

    if (denominator < 1e-8).any():  # clamp, just in case
        denominator = torch.clamp(denominator, 1e-8)

    loss = torch.mean(-torch.log(numerator / denominator))

    return loss


def test_simclr_loss():
    
    
    arr_1 = torch.randn((20, 1, 224, 224))
    arr_3 = torch.randn((19, 1, 224, 224))
    
    assert arr_1.shape[0] != 20, "size correct"
    assert arr_3.shape[0] != 20, f"{arr_3} size not correct"


class NegativeCosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity used in the SimSiam[0] paper.

    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return -cosine_similarity(x0, x1, self.dim, self.eps).mean()
    

if __name__ == "__main__":
    cos = cosine_similarity
    test_simclr_loss()
