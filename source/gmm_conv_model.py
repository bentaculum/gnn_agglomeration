import torch
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from gnn_model import GnnModel


class GmmConvModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):

        super(GmmConvModel, self).__init__(
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            epoch=epoch,
            train_batch_iteration=train_batch_iteration,
            val_batch_iteration=val_batch_iteration,
            model_type=model_type)

    def layers(self):
        self.conv_in = GMMConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=self.config.hidden_units,
            dim=self.config.pseudo_dimensionality)

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.config.hidden_layers):
            layer = GMMConv(
                in_channels=self.config.hidden_units,
                out_channels=self.config.hidden_units,
                dim=self.config.pseudo_dimensionality)
            self.hidden_layers.append(layer)

        self.conv_out = GMMConv(
            in_channels=self.config.hidden_units,
            out_channels=self.model_type.out_channels,
            dim=self.config.pseudo_dimensionality)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.training:
            self.write_to_variable_summary(self.conv_in.mu, 'in_layer', 'weights_mu')
            self.write_to_variable_summary(self.conv_in.sigma, 'in_layer', 'weights_sigma')
            self.write_to_variable_summary(self.conv_in.lin.weight, 'in_layer', 'weights_matmul')
            self.write_to_variable_summary(self.conv_in.lin.bias, 'in_layer', 'weights_matmul_bias')

        x = self.conv_in(x=x, edge_index=edge_index, pseudo=edge_attr)
        self.write_to_variable_summary(x, 'in_layer', 'preactivations')
        x = getattr(F, self.config.non_linearity)(x)
        self.write_to_variable_summary(x, 'in_layer', 'outputs')
        x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)

        for i, l in enumerate(self.hidden_layers):
            if self.training:
                self.write_to_variable_summary(l.mu, 'layer_{}'.format(i), 'weights_mu')
                self.write_to_variable_summary(l.sigma, 'layer_{}'.format(i), 'weights_sigma')
                self.write_to_variable_summary(l.lin.weight, 'layer_{}'.format(i), 'weights_matmul')
                self.write_to_variable_summary(l.lin.bias, 'layer_{}'.format(i), 'weights_matmul_bias')

            x = l(x=x, edge_index=edge_index, pseudo=edge_attr)
            self.write_to_variable_summary(x, 'layer_{}'.format(i), 'preactivations')
            x = getattr(F, self.config.non_linearity)(x)
            self.write_to_variable_summary(x, 'layer_{}'.format(i), 'outputs')
            x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)

        if self.training:
            self.write_to_variable_summary(self.conv_out.mu, 'out_layer', 'weights_mu')
            self.write_to_variable_summary(self.conv_out.sigma, 'out_layer', 'weights_sigma')
            self.write_to_variable_summary(self.conv_out.lin.weight, 'out_layer', 'weights_matmul')
            self.write_to_variable_summary(self.conv_out.lin.bias, 'out_layer', 'weights_matmul_bias')

        x = self.conv_out(x=x, edge_index=edge_index, pseudo=edge_attr)
        self.write_to_variable_summary(x, 'out_layer', 'preactivations')
        x = self.model_type.out_nonlinearity(x)
        self.write_to_variable_summary(x, 'out_layer', 'outputs')

        return x
