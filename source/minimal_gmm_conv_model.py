import torch
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from gnn_model import GnnModel


class MinimalGmmConvModel(GnnModel):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type=None):

        super(MinimalGmmConvModel, self).__init__(
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
            out_channels=self.model_type.out_channels,
            dim=self.config.pseudo_dimensionality,
            kernel_size=self.config.kernel_size,
            bias=self.config.use_bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if self.training:
            self.write_to_variable_summary(self.conv_in.mu, 'in_layer', 'weights_mu')
            self.write_to_variable_summary(self.conv_in.sigma, 'in_layer', 'weights_sigma')
            self.write_to_variable_summary(self.conv_in.lin.weight, 'in_layer', 'weights_matmul')
            if self.config.use_bias:
                self.write_to_variable_summary(self.conv_in.lin.bias, 'in_layer', 'weights_matmul_bias')

        x = self.conv_in(x=x, edge_index=edge_index, pseudo=edge_attr)
        self.write_to_variable_summary(x, 'in_layer', 'outputs')
        # x = getattr(F, self.config.dropout_type)(x, p=self.config.dropout_prob, training=self.training)

        # TODO add a separate fully connected layer over the output channels of the Conv Layer

        return x
