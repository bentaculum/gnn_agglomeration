import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AttentionMLP(torch.nn.Module):
    def __init__(
            self,
            heads,
            in_features,
            layers=1,
            layer_dims=[1],
            bias=True,
            non_linearity='relu',
            batch_norm=True,
            dropout_probs=[0.0]):
        super(AttentionMLP, self).__init__()
        assert len(layer_dims) == layers
        # Dropout should be there for input layer + all hidden layers
        assert len(dropout_probs) == layers

        self.layers = layers
        self.bias = bias
        self.non_linearity = non_linearity
        self.batch_norm = batch_norm
        self.dropout_probs = dropout_probs

        self.weight_list = torch.nn.ParameterList()
        self.bias_list = torch.nn.ParameterList()
        self.batch_norm_list = torch.nn.ModuleList()

        w_in = Parameter(torch.Tensor(1, heads, in_features, layer_dims[0]))
        self.weight_list.append(w_in)
        logger.debug(f'att layer {tuple(w_in.shape[1:])}')

        for i in range(layers - 1):
            w = Parameter(torch.Tensor(
                1, heads, layer_dims[i], layer_dims[i+1]))
            self.weight_list.append(w)
            logger.debug(f'att layer {tuple(w.shape[1:])}')

        if bias:
            for i in range(layers):
                b = Parameter(torch.Tensor(1, heads, layer_dims[i]))
                self.bias_list.append(b)
        if batch_norm:
            for i in range(layers):
                bn = torch.nn.BatchNorm1d(heads)
                self.batch_norm_list.append(bn)

        self.reset_parameters()

    def reset_parameters(self):
        # from torch.nn.Linear
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        # However, the sqrt(5) that appears there is legacy code and should not
        # be used
        for w in self.weight_list:
            init.kaiming_uniform_(w, nonlinearity='leaky_relu')

        for i, b in enumerate(self.bias_list):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_list[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)

    def forward(self, x):
        for i, w in enumerate(self.weight_list):

            # TODO this seems to be the memory bottleneck of the entire network
            # if torch.cuda.is_available():
            #     logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated(device='cuda')} B")

            # enable batched matrix multiplication with extra dim
            x = x.unsqueeze(-2)
            x = torch.matmul(x, w)
            # remove extra dim
            x = x.squeeze(-2)
            if self.bias:
                x += self.bias_list[i]
            x = getattr(F, self.non_linearity)(x)
            if self.batch_norm:
                x = self.batch_norm_list[i](x)
            x = F.dropout(x, p=self.dropout_probs[i], training=self.training)

        # after last layer, squeeze the last dimension, if it's of size 1
        x = x.squeeze(dim=-1)

        return x
