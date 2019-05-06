import torch
import torch.nn.functional as F
from gnn_model import GnnModel

from our_conv import OurConv

class OurConvModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):

        super(OurConvModel, self).__init__(
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            epoch=epoch,
            train_batch_iteration=train_batch_iteration,
            val_batch_iteration=val_batch_iteration,
            model_type=model_type)

    def layers(self):
        self.layers_list = torch.nn.ModuleList()

        conv_in = OurConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=self.config.hidden_units,
            dim=self.config.pseudo_dimensionality,
            heads=self.config.kernel_size,
            concat=False,
            negative_slope=0.2,
            dropout=self.config.dropout_prob,
            bias=self.config.use_bias)

        self.layers_list.append(conv_in)

        # account for the attention heads sizing in the different layers
        for i in range(self.config.hidden_layers):
            l = OurConv(
                in_channels=self.config.hidden_units,
                out_channels=self.config.hidden_units,
                dim=self.config.pseudo_dimensionality,
                heads=self.config.kernel_size,
                concat=False,
                negative_slope=0.2,
                dropout=self.config.dropout_prob,
                bias=self.config.use_bias)
            self.layers_list.append(l)

        self.fc = torch.nn.Linear(
            in_features=self.config.hidden_units,
            out_features=self.model_type.out_channels,
            bias=self.config.use_bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, l in enumerate(self.layers_list):
            if self.training:
                self.write_to_variable_summary(
                    l.weight, 'layer_{}'.format(i), 'weights')
                self.write_to_variable_summary(
                    l.att, 'layer_{}'.format(i), 'weights_attention')
                if self.config.use_bias:
                    self.write_to_variable_summary(
                        l.bias, 'layer_{}'.format(i), 'weights_bias')

            x = l(x=x, edge_index=edge_index, pseudo=edge_attr)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'preactivations')

            x = getattr(F, self.config.non_linearity)(x)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'outputs')
            x = getattr(F, self.config.dropout_type)(
                x, p=self.config.dropout_prob, training=self.training)

        if self.training:
            self.write_to_variable_summary(
                self.fc.weight, 'out_layer', 'weights')
            if self.config.use_bias:
                self.write_to_variable_summary(
                    self.fc.bias, 'out_layer', 'bias')
        x = self.fc(x)
        self.write_to_variable_summary(x, 'out_layer', 'pre_activations')

        x = self.model_type.out_nonlinearity(x)
        self.write_to_variable_summary(x, 'out_layer', 'outputs')

        return x
