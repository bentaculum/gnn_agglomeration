import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_model import GnnModel


class GcnRegression(GnnModel):
    def __init__(self, config, train_writer, val_writer):
        super(GcnRegression, self).__init__(config, train_writer, val_writer)
        self.loss_name = 'MSE loss'

    def layers(self):
        self.conv1 = GCNConv(self.config.feature_dimensionality, self.config.hidden_units)
        self.conv2 = GCNConv(self.config.hidden_units, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.training:
            self.write_to_variable_summary(self.conv1.weight, 'conv1', 'params_weights')
            self.write_to_variable_summary(self.conv1.bias, 'conv1', 'params_bias')

        x = self.conv1(x, edge_index)
        self.write_to_variable_summary(x, 'conv1', 'preactivations')
        x = getattr(F, self.config.hidden_activation)(x)
        self.write_to_variable_summary(x, 'conv1', 'outputs')
        x = F.dropout(x, training=self.training)

        if self.training:
            self.write_to_variable_summary(self.conv2.weight, 'conv2', 'params_weights')
            self.write_to_variable_summary(self.conv2.bias, 'conv2', 'params_bias')
        x = self.conv2(x, edge_index)
        self.write_to_variable_summary(x, 'conv2', 'preactivations')

        return x

    def loss(self, inputs, targets):
        self.current_loss =  F.mse_loss(inputs, targets.float(), reduction='mean')
        self.write_to_variable_summary(self.current_loss, 'out_layer', 'mse_loss')
        return self.current_loss

    def evaluate_metric(self, data):
        pred = self.forward(data).round()
        correct = torch.squeeze(pred).eq(data.y.float()).sum().item()
        acc = correct / data.num_nodes
        # print('\nAccuracy: {:.4f}'.format(acc))
        return acc

    def evaluate_as_list(self, data):
        pred = self.forward(data).round()
        return torch.squeeze(pred).tolist()

