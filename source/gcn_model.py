import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_model import GnnModel


class GcnModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):
        super(GcnModel, self).__init__(
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            epoch=epoch,
            train_batch_iteration=train_batch_iteration,
            val_batch_iteration=val_batch_iteration,
            model_type=model_type
        )

    def layers(self):
        self.conv1 = GCNConv(self.config.feature_dimensionality, self.config.hidden_units)
        self.conv2 = GCNConv(self.config.hidden_units, self.model_type.out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.training:
            self.write_to_variable_summary(self.conv1.weight, 'conv1', 'params_weights')
            self.write_to_variable_summary(self.conv1.bias, 'conv1', 'params_bias')

        x = self.conv1(x, edge_index)
        self.write_to_variable_summary(x, 'conv1', 'preactivations')
        x = getattr(F, self.config.non_linearity)(x)
        self.write_to_variable_summary(x, 'conv1', 'outputs')
        x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)

        if self.training:
            self.write_to_variable_summary(self.conv2.weight, 'conv2', 'params_weights')
            self.write_to_variable_summary(self.conv2.bias, 'conv2', 'params_bias')
        x = self.conv2(x, edge_index)
        self.write_to_variable_summary(x, 'conv2', 'preactivations')

        return x
