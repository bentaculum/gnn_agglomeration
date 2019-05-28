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
        assert len(self.config.fc_layer_dims) == self.config.fc_layers - 1

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
            local_layers=self.config.att_nodenet_layers,
            local_hidden_dims=self.config.att_nodenet_hidden_dims,
            non_linearity=self.config.att_non_linearity,
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
                normalize_with_softmax=self.config.att_normalize,
                local_layers=self.config.att_nodenet_layers,
                local_hidden_dims=self.config.att_nodenet_hidden_dims,
                non_linearity=self.config.att_non_linearity,
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

        if self.config.edge_labels:
            fc_in_features = 2 * \
                (fc_in_features + self.config.pseudo_dimensionality)

        self.fc_layers_list = torch.nn.ModuleList()
        fc_layer_dims = self.config.fc_layer_dims.copy()
        fc_layer_dims.insert(0, fc_in_features)
        fc_layer_dims.append(self.model_type.out_channels)
        for i in range(self.config.fc_layers):
            fc = torch.nn.Linear(
                in_features=fc_layer_dims[i],
                out_features=fc_layer_dims[i + 1],
                bias=self.config.fc_bias)
            self.fc_layers_list.append(fc)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, l in enumerate(self.layers_list):
            if self.training:
                for j, weight in enumerate(l.weight_list):
                    self.write_to_variable_summary(
                        weight, 'layer_{}_nodenet_{}'.format(i, j), 'weights')
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

        # TODO this is a quick fix implementation, with the assumption that
        # a pair of edges is next to each other in the edge index
        if self.config.edge_labels:
            x = x[edge_index[0]]
            # might be computationally expensive
            x = torch.cat([x, edge_attr], dim=-1)
            # One entry per edge, not per directed edge
            x = x.view(int(edge_index.size(1) / 2), -1)

        # TODO make dropout optional here
        for i, l in enumerate(self.fc_layers_list):
            if self.training:
                self.write_to_variable_summary(
                    l.weight, 'out_layer_fc_{}'.format(i), 'weights')
                if self.config.fc_bias:
                    self.write_to_variable_summary(
                        l.bias, 'out_layer_fc_{}'.format(i), 'bias')
            x = l(x)
            self.write_to_variable_summary(
                x, 'out_layer_fc_{}'.format(i), 'pre_activations')

            if i == self.config.fc_layers - 1:
                x = self.model_type.out_nonlinearity(x)
            else:
                x = getattr(F, self.config.non_linearity)(x)
            self.write_to_variable_summary(
                x, 'out_layer_fc_{}'.format(i), 'outputs')

        return x
