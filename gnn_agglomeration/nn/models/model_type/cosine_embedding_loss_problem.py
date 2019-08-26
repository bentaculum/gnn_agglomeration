import torch
import torch.nn.functional as F

from .model_type import ModelType


class CosineEmbeddingLossProblem(ModelType):
    def __init__(self, config):
        super().__init__(config)

        self.loss_name = 'Cosine Embedding loss'

        # TODO this is only true if there is no fc layer on top of each node after the GNN
        if self.config.att_heads_concat:
            self.out_channels = self.config.hidden_units[-1] * \
                self.config.attention_heads[-1]
        else:
            self.out_channels = self.config.hidden_units[-1]

    def out_nonlinearity(self, x):
        return x

    def loss_one_by_one(self, inputs, targets):
        """

        Args:
            inputs (tuple of torch.Tensor):
            targets (torch.Tensor):

        Returns:

        """
        # go from 0 = merge, 1 = split ground truth values to
        # 1 = merge, -1 = split targets
        # TODO careful with bitwise operations in torch, feature is being updated in recent versions
        targets = (~(targets.byte())).float()
        targets = targets * 2 - 1

        return F.cosine_embedding_loss(
            input1=inputs[0],
            input2=inputs[1],
            target=targets,
            reduction='none',
            margin=self.config.cosine_loss_margin)

    def out_to_predictions(self, out):
        cosine_similarity = F.cosine_similarity(out[0], out[1], dim=1)
        pred = (~(cosine_similarity > self.config.cosine_threshold)).float()
        return pred

    def out_to_one_dim(self, out):
        # Transform back to space from 0 to 1
        cosine_sim = F.cosine_similarity(out[0], out[1])
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)

        # go from 1 = merge, -1 = split cosine similarity to
        # 0 = merge, 1 = split values for RAG db
        return -((cosine_sim - 1) * 0.5)

    def predictions_to_list(self, predictions):
        return predictions.tolist()

    def metric(self, predictions, targets, mask):
        weighted_equal = predictions.eq(targets.float()).float() * mask.float()
        correct = weighted_equal.sum().item()
        acc = correct / (mask.sum().item() + torch.finfo(torch.float).tiny)
        return acc
