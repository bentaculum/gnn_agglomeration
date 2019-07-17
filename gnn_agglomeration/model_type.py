import torch
from torch.nn import functional as F
from abc import ABC, abstractmethod
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelType(torch.nn.Module, ABC):
    loss_name: str
    out_channels: int

    def __init__(self, config):
        super(ModelType, self).__init__()

        self.config = config

    @abstractmethod
    def out_nonlinearity(self, x):
        pass

    @abstractmethod
    def loss_one_by_one(self, inputs, targets):
        pass

    def loss(self, inputs, targets, mask):
        l = self.loss_one_by_one(inputs, targets)
        # in pyg, the graphs in a mini-batch get merged on dimension 0
        # TODO remove assert statement
        assert l.size(0) == mask.size(0)

        return torch.sum(l * mask) / torch.sum(mask)

        # old, mean without masking
        # return torch.mean(l * mask, dim=0, keepdim=False)

    @abstractmethod
    def out_to_predictions(self, out):
        pass

    @abstractmethod
    def predictions_to_list(self, predictions):
        pass

    @abstractmethod
    def metric(self, predictions, targets):
        pass

    def plot_targets_vs_predictions(self, targets, predictions):
        # TODO improve: clip predictions to 0 as a quick fix
        predictions = [x if x > 0 else 0 for x in predictions]
        np.set_printoptions(suppress=True)
        size = max(max(targets), max(predictions))
        cm = np.zeros([size + 1, size + 1], dtype=int)
        for i in range(len(targets)):
            cm[predictions[i], targets[i]] += 1
        cm = np.flip(cm, axis=0)
        ax = sns.heatmap(
            cm,
            xticklabels=list(
                range(
                    size + 1)),
            yticklabels=list(
                reversed(
                    range(
                        size + 1))),
            annot=True,
            annot_kws={"size": 6},
            fmt='g')
        fig = ax.get_figure()
        plt.title('Confusion matrix')
        plt.xlabel('Targets')
        plt.ylabel('Predictions')
        fig.savefig(
            os.path.join(
                self.config.run_abs_path,
                self.config.confusion_matrix_path))
