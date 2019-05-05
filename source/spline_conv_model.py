import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from gnn_model import GnnModel


class SplineConvModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):

        # Params from the SplineCNN paper
        config.non_linearity = 'elu'
        config.kernel_size = 4
        config.hidden_units = 16
        config.adam_weight_decay = 0.005

        super(SplineConvModel, self).__init__(
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            epoch=epoch,
            train_batch_iteration=train_batch_iteration,
            val_batch_iteration=val_batch_iteration,
            model_type=model_type)

    def layers(self):
        self.layers_list = torch.nn.ModuleList()

        conv_in = SplineConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=self.config.hidden_units,
            dim=self.config.pseudo_dimensionality,
            kernel_size=self.config.kernel_size,
            norm=False,
            root_weight=False,
            bias=self.config.use_bias)

        self.layers_list.append(conv_in)

        for i in range(self.config.hidden_layers):
            l = SplineConv(
                in_channels=self.config.hidden_units,
                out_channels=self.config.hidden_units,
                dim=self.config.pseudo_dimensionality,
                kernel_size=self.config.kernel_size,
                norm=False,
                root_weight=False,
                bias=self.config.use_bias)
            self.layers_list.append(l)

        conv_out = SplineConv(
            in_channels=self.config.hidden_units,
            out_channels=self.model_type.out_channels,
            dim=self.config.pseudo_dimensionality,
            kernel_size=self.config.kernel_size,
            norm=False,
            root_weight=False,
            bias=self.config.use_bias)

        self.layers_list.append(conv_out)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, l in enumerate(self.layers_list):
            if self.training:
                self.write_to_variable_summary(
                    l.weight, 'layer_{}'.format(i), 'weights')
                if self.config.use_bias:
                    self.write_to_variable_summary(
                        l.bias, 'layer_{}'.format(i), 'weights_bias')
                if l.root:
                    self.write_to_variable_summary(
                        l.root, 'layer_{}'.format(i), 'weights_root_mul')

            x = l(x=x, edge_index=edge_index, pseudo=edge_attr)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'preactivations')

            if i < len(self.layers_list) - 1:
                x = getattr(F, self.config.non_linearity)(x)
                self.write_to_variable_summary(
                    x, 'layer_{}'.format(i), 'outputs')
                x = getattr(F, self.config.dropout_type)(
                    x, p=self.config.dropout_prob, training=self.training)
            else:
                x = self.model_type.out_nonlinearity(x)
                self.write_to_variable_summary(
                    x, 'layer_{}'.format(i), 'outputs')

        return x
