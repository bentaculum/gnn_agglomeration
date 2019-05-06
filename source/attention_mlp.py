import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init


class AttentionMLP(torch.nn.Module):
    def __init__(self, heads, in_features, bias):
        super(AttentionMLP, self).__init__()
        self.weight_out = Parameter(torch.Tensor(1, heads, in_features))
        if bias:
            self.bias_out = Parameter(torch.Tensor(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        # from torch.nn.Linear
        init.kaiming_uniform_(self.weight_out, a=math.sqrt(5))
        if self.bias_out is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_out)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_out, -bound, bound)

    def forward(self, x):
        x = (x * self.weight_out).sum(dim=-1)
        x += self.bias_out
        return x

