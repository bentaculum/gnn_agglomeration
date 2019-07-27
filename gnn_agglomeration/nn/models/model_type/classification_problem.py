import torch.nn.functional as F

from .model_type import ModelType


class ClassificationProblem(ModelType):
    def __init__(self, config):
        super().__init__(config)

        self.loss_name = 'NLL_loss'
        self.out_channels = self.config.classes

    def out_nonlinearity(self, x):
        return F.log_softmax(x, dim=1)

    def loss_one_by_one(self, inputs, targets):
        return F.nll_loss(inputs, targets, reduction='none')

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def out_to_one_dim(self, out):
        if self.out_channels > 2:
            raise NotImplementedError(
                "projection of outputs to continuous 1d output space not defined yet")

        return out[:, 1]

    def predictions_to_list(self, predictions):
        return predictions.tolist()

    def metric(self, predictions, targets):
        correct = predictions.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return acc
