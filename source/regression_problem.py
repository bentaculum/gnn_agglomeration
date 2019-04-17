import torch
import os
import tensorboardX
import torch.nn.functional as F

from model_type import ModelType

import chartify
import pandas as pd


class RegressionProblem(ModelType):
    def __init__(self, config):
        super(RegressionProblem, self).__init__(config)

        self.loss_name = 'MSE Loss'
        self.out_channels = 1

    def loss(self, inputs, targets):
        return F.mse_loss(inputs, targets.float(), reduction='mean')

    def out_to_predictions(self, out):
        return out.round()

    def metric(self, predictions, targets):
        correct = torch.squeeze(predictions).eq(targets.float()).sum().item()
        acc = correct / targets.size(0)
        return acc

    def predictions_to_list(self, predictions):
        return torch.squeeze(predictions).tolist()

    def plot_targets_vs_predictions(self, targets, predictions):
        ch = chartify.Chart(blank_labels=True, x_axis_type='numerical', y_axis_type='numerical')
        ch.plot.scatter(
            pd.DataFrame({'t': targets, 'p': predictions}).groupby(['t','p']).size().reset_index(name='count'),
            x_column='t', y_column='p', color_column='count', text_column='count'
        ).axes.set_xaxis_label('targets') \
            .axes.set_yaxis_label('predictions') \


