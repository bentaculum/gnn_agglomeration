import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F


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

        w_in = Parameter(torch.Tensor(1, heads, layer_dims[0], in_features))
        self.weight_list.append(w_in)

        for i in range(layers - 1):
            w = Parameter(torch.Tensor(
                1, heads, layer_dims[i + 1], layer_dims[i]))
            self.weight_list.append(w)

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
        for w in self.weight_list:
            init.kaiming_uniform_(w, a=math.sqrt(5))

        for i, b in enumerate(self.bias_list):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_list[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)

    def forward(self, x):
        for i, w in enumerate(self.weight_list):
            x = x.unsqueeze(-2)
            x = (x * w).sum(dim=-1)
            if self.bias:
                x += self.bias_list[i]
            x = getattr(F, self.non_linearity)(x)
            if self.batch_norm:
                x = self.batch_norm_list[i](x)
            x = F.dropout(x, p=self.dropout_probs[i], training=self.training)

        # after last layer, squeeze the last dimension, if it's of size 1
        x = x.squeeze(dim=-1)

        return x
