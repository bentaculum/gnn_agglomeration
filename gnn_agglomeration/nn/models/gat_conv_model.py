import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .gnn_model import GnnModel


class GatConvModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):

        super(GatConvModel, self).__init__(
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            epoch=epoch,
            train_batch_iteration=train_batch_iteration,
            val_batch_iteration=val_batch_iteration,
            model_type=model_type)

    def layers(self):
        # TODO adapt to per-layer configurability
        self.layers_list = torch.nn.ModuleList()

        conv_in = GATConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=self.config.hidden_units,
            heads=self.config.kernel_size,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            bias=self.config.use_bias)

        self.layers_list.append(conv_in)

        for i in range(self.config.hidden_layers):
            l = GATConv(
                in_channels=self.config.hidden_units,
                out_channels=self.config.hidden_units,
                heads=self.config.kernel_size,
                concat=True,
                negative_slope=0.2,
                dropout=0.0,
                bias=self.config.use_bias)
            self.layers_list.append(l)

        conv_out = GATConv(
            in_channels=self.config.hidden_units * self.config.kernel_size,
            out_channels=self.model_type.out_channels,
            heads=self.config.kernel_size,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            bias=self.config.use_bias)

        self.layers_list.append(conv_out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, l in enumerate(self.layers_list):
            if self.training:
                self.write_to_variable_summary(
                    l.weight, 'layer_{}'.format(i), 'weights')
                self.write_to_variable_summary(
                    l.att, 'layer_{}'.format(i), 'weights_attention')
                if self.config.use_bias:
                    self.write_to_variable_summary(
                        l.bias, 'layer_{}'.format(i), 'weights_bias')

            x = l(x=x, edge_index=edge_index)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'preactivations')

            if i < len(self.layers_list) - 1:
                x = getattr(F, self.config.non_linearity)(x)
                self.write_to_variable_summary(
                    x, 'layer_{}'.format(i), 'outputs')
                x = getattr(F, self.config.dropout_type)(
                    x, p=self.config.dropout_probs, training=self.training)
            else:
                x = self.model_type.out_nonlinearity(x)
                self.write_to_variable_summary(
                    x, 'layer_{}'.format(i), 'outputs')

        return x
