import torch
import os
import tensorboardX
import torch.nn.functional as F

from model_type import ModelType

import chartify
import pandas as pd


class ClassificationProblem(ModelType):
    def __init__(self, config):
        super(ClassificationProblem, self).__init__(config)

        self.loss_name = 'NLL loss'
        self.out_channels = self.config.max_neighbors + 1

    def loss(self, inputs, targets):
        return F.nll_loss(inputs, targets, reduction='mean')

    def out_to_predictions(self, out):
        _, pred = out.max(dim=1)
        return pred

    def predictions_to_list(self, predictions):
        return predictions.tolist()

    def metric(self, predictions, targets):
        correct = predictions.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return acc

    def plot_targets_vs_predictions(self, targets, predictions):
        ch = chartify.Chart(blank_labels=True, x_axis_type='categorical', y_axis_type='categorical')
        ch.plot.heatmap(
            pd.DataFrame({'t': targets, 'p': predictions}).groupby(['t','p']).size().reset_index(name='count'),
            x_column='t', y_column='p', color_column='count', text_column='count'
        ).axes.set_xaxis_label('targets') \
            .axes.set_yaxis_label('predictions') \
            .set_title('Confusion matrix on test set')
        ch.save(filename=os.path.join(self.config.temp_dir, 'confusion_matrix_test.png'), format='png')
