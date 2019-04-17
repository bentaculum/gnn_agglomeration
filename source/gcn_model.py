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
        self.conv_in = GCNConv(self.config.feature_dimensionality, self.config.hidden_units)
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.config.hidden_layers):
            layer = GCNConv(self.config.hidden_units, self.config.hidden_units)
            self.hidden_layers.append(layer)
            
        self.conv_out = GCNConv(self.config.hidden_units, self.model_type.out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.training:
            self.write_to_variable_summary(self.conv_in.weight, 'conv_in', 'params_weights')
            self.write_to_variable_summary(self.conv_in.bias, 'conv_in', 'params_bias')

        x = self.conv_in(x, edge_index)
        self.write_to_variable_summary(x, 'conv_in', 'preactivations')
        x = getattr(F, self.config.non_linearity)(x)
        self.write_to_variable_summary(x, 'conv_in', 'outputs')
        x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)



        for i, l in enumerate(self.hidden_layers):
            if self.training:
                self.write_to_variable_summary(l.weight, 'layer_{}'.format(i), 'params_weights')
                self.write_to_variable_summary(l.bias, 'layer_{}'.format(i), 'params_bias')

            x = l(x, edge_index)
            self.write_to_variable_summary(x, 'layer_{}'.format(i), 'preactivations')
            x = getattr(F, self.config.non_linearity)(x)
            self.write_to_variable_summary(x, 'layer_{}'.format(i), 'outputs')
            x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)

        if self.training:
            self.write_to_variable_summary(self.conv_out.weight, 'conv_out', 'params_weights')
            self.write_to_variable_summary(self.conv_out.bias, 'conv_out', 'params_bias')
        x = self.conv_out(x, edge_index)
        self.write_to_variable_summary(x, 'conv_out', 'preactivations')

        return x
