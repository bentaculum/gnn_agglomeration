import torch.nn.functional as F

from .model_type import ModelType


class CosineEmbeddingLossProblem(ModelType):
    def __init__(self, config):
        super().__init__(config)

        self.loss_name = 'Cosine Embedding loss'
        self.out_channels = self.config.out_dimensionality

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
        # TODO careful with bitwise operations in torch, feature is still beta
        targets = (~(targets.byte())).float()
        targets = targets * 2 - 1

        # TODO parametrize margin
        return F.cosine_embedding_loss(
            input1=inputs[0],
            input2=inputs[1],
            target=targets,
            reduction='none',
            margin=0.5)

    def out_to_predictions(self, out):
        # TODO parametrize embedding threshold
        cosine_similarity = F.cosine_similarity(out[0], out[1], dim=1)
        pred = (~(cosine_similarity > 0)).float()
        return pred

    def out_to_one_dim(self, out):
        return F.cosine_similarity(out[0], out[1])

    def predictions_to_list(self, predictions):
        return predictions.tolist()

    def metric(self, predictions, targets):
        correct = predictions.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return acc
