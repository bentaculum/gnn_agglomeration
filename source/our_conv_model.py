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
        # Assert some layer configs
        # By default, we need to specify the size of the representation between
        # input and output layer
        assert len(self.config.hidden_units) == self.config.hidden_layers + 1
        # Dropout should be there for input layer + all hidden layers
        assert len(self.config.dropout_probs) == self.config.hidden_layers + 1

        self.layers_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList()

        attention_nn_params = {
            'layers': self.config.att_layers,
            'layer_dims': self.config.att_layer_dims,
            'non_linearity': self.config.att_non_linearity,
            'batch_norm': self.config.att_batch_norm,
            'dropout_probs': self.config.att_dropout_probs,
            'bias': self.config.att_bias,
        }

        out_channels_in = self.config.hidden_units[0]
        conv_in = OurConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=out_channels_in,
            dim=self.config.pseudo_dimensionality,
            heads=self.config.kernel_size,
            concat=self.config.att_heads_concat,
            negative_slope=0.2,
            dropout=self.config.att_final_dropout,
            bias=self.config.use_bias,
            normalize_with_softmax=self.config.att_normalize,
            attention_nn_params=attention_nn_params
        )
        self.layers_list.append(conv_in)

        if self.config.batch_norm:
            if self.config.att_heads_concat:
                batch_norm_size_in = out_channels_in * self.config.kernel_size
            else:
                batch_norm_size_in = out_channels_in
            b = torch.nn.BatchNorm1d(batch_norm_size_in)

            self.batch_norm_list.append(b)

        for i in range(self.config.hidden_layers):
            if self.config.att_heads_concat:
                in_channels = self.config.hidden_units[i] * \
                    (self.config.kernel_size**(i + 1))
                out_channels = self.config.hidden_units[i + 1] * \
                    (self.config.kernel_size**(i + 1))
            else:
                in_channels = self.config.hidden_units[i]
                out_channels = self.config.hidden_units[i + 1]

            l = OurConv(
                in_channels=in_channels,
                out_channels=out_channels,
                dim=self.config.pseudo_dimensionality,
                heads=self.config.kernel_size,
                concat=self.config.att_heads_concat,
                negative_slope=0.2,
                dropout=self.config.att_final_dropout,
                bias=self.config.use_bias,
                attention_nn_params=attention_nn_params
            )
            self.layers_list.append(l)

            if self.config.batch_norm:
                if self.config.att_heads_concat:
                    batch_norm_size = out_channels * self.config.kernel_size
                else:
                    batch_norm_size = out_channels
                b = torch.nn.BatchNorm1d(batch_norm_size)
                self.batch_norm_list.append(b)

        if self.config.att_heads_concat:
            fc_in_features = self.config.hidden_units[-1] * \
                (self.config.kernel_size**(self.config.hidden_layers + 1))
        else:
            fc_in_features = self.config.hidden_units[-1]

        self.fc = torch.nn.Linear(
            in_features=fc_in_features,
            out_features=self.model_type.out_channels,
            bias=self.config.fc_bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, l in enumerate(self.layers_list):
            if self.training:
                self.write_to_variable_summary(
                    l.weight, 'layer_{}'.format(i), 'weights')
                if self.config.use_bias:
                    self.write_to_variable_summary(
                        l.bias, 'layer_{}'.format(i), 'weights_bias')

                for j in range(self.config.att_layers):
                    self.write_to_variable_summary(
                        l.att.weight_list[j],
                        'layer_{}'.format(i),
                        'att_mlp/weight_layer_{}'.format(j))
                    if self.config.att_bias:
                        self.write_to_variable_summary(
                            l.att.bias_list[j], 'layer_{}'.format(i), 'att_mlp/bias_layer_{}'.format(j))

            x = l(x=x, edge_index=edge_index, pseudo=edge_attr)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'preactivations')

            x = getattr(F, self.config.non_linearity)(x)
            self.write_to_variable_summary(
                x, 'layer_{}'.format(i), 'outputs')

            if self.config.batch_norm:
                x = self.batch_norm_list[i](x)
                self.write_to_variable_summary(
                    x, 'layer_{}'.format(i), 'outputs_batch_norm')

            x = getattr(F, self.config.dropout_type)(
                x, p=self.config.dropout_probs[i], training=self.training)

        if self.training:
            self.write_to_variable_summary(
                self.fc.weight, 'out_layer', 'weights')
            if self.config.fc_bias:
                self.write_to_variable_summary(
                    self.fc.bias, 'out_layer', 'bias')
        x = self.fc(x)
        self.write_to_variable_summary(x, 'out_layer', 'pre_activations')

        x = self.model_type.out_nonlinearity(x)
        self.write_to_variable_summary(x, 'out_layer', 'outputs')

        return x
