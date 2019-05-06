import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init


class AttentionMLP(torch.nn.Module):
    def __init__(self, heads, in_features, layers=0, hidden_units=None, bias=True):
        super(AttentionMLP, self).__init__()

        self.layers = layers
        self.bias = bias
        if hidden_units is None:
            hidden_units = in_features

        if layers > 1:
            raise NotImplementedError("Attention MLP with arbitrary number of hidden layers not implemented yet")

        if layers == 1:
            self.weight_in = Parameter(torch.Tensor(1, heads, hidden_units, in_features))
            self.weight_out = Parameter(torch.Tensor(1, heads, hidden_units))
            if bias:
                self.bias_in = Parameter(torch.Tensor(1, heads, hidden_units))
                self.bias_out = Parameter(torch.Tensor(1, heads))
            else:
                # from https://pytorch.org/docs/stable/notes/extending.html?highlight=module
                self.register_parameter('bias_in', None)
                self.register_parameter('bias_out', None)

        if layers == 0:
            self.weight_out = Parameter(torch.Tensor(1, heads, in_features))
            if bias:
                self.bias_out = Parameter(torch.Tensor(1, heads))
            else:
                self.register_parameter('bias_out', None)

        self.reset_parameters()

    def reset_parameters(self):
        # from torch.nn.Linear
        if self.layers == 1:
            init.kaiming_uniform_(self.weight_in, a=math.sqrt(5))
            if self.bias_in is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_in)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_in, -bound, bound)

        init.kaiming_uniform_(self.weight_out, a=math.sqrt(5))
        if self.bias_out is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_out)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_out, -bound, bound)

    def forward(self, x):
        if self.layers == 1:
            x = x.unsqueeze(-2)
            x = (x*self.weight_in).sum(dim=-1)
            if self.bias:
                x += self.bias_in
            # TODO parametrize non-linearity
            x = torch.nn.functional.relu(x)

        x = (x * self.weight_out).sum(dim=-1)
        if self.bias:
            x += self.bias_out
        return x

