import torch
from torch.nn import functional as F
from abc import ABC, abstractmethod
import os
import tensorboardX
import chartify
import pandas as pd

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
    def loss(self, inputs, targets):
        pass

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
        ch = chartify.Chart(blank_labels=True, x_axis_type='categorical', y_axis_type='categorical')
        ch.plot.heatmap(
            pd.DataFrame({'t': targets, 'p': predictions}).groupby(['t','p']).size().reset_index(name='count'),
            x_column='t', y_column='p', color_column='count', text_column='count'
        ).axes.set_xaxis_label('targets') \
            .axes.set_yaxis_label('predictions') \
            .set_title('Confusion matrix on test set')
        ch.save(filename=os.path.join(self.config.temp_dir, 'confusion_matrix_test.png'), format='png')
