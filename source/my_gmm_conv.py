import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform, reset


class MyGMMConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dim, kernel_size, bias=True):
        super(MyGMMConv, self).__init__('add')  # Fixed Bug here. The scatter operation is performed over the neighbors, so it should be additive

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        self.mu = Parameter(torch.Tensor(in_channels * kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(in_channels * kernel_size, dim))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        torch.nn.init.uniform_(self.mu, a=-0.01, b=0.01)
        torch.nn.init.uniform_(self.sigma, a=-0.01, b=0.01)
        # uniform(size, self.mu)
        # uniform(size, self.sigma)
        reset(self.lin)

    def forward(self, x, edge_index, pseudo):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.propagate(edge_index, x=x, pseudo=pseudo)
        return self.lin(out)

    def message(self, x_j, pseudo):
        F, (E, D), K = x_j.size(1), pseudo.size(), self.mu.size(0)

        # See: https://github.com/shchur/gnn-benchmark
        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D)) ** 2
        gaussian = gaussian / (1e-14 + self.sigma.view(1, K, D) ** 2)
        gaussian = torch.exp(gaussian.sum(dim=-1))

        return x_j * gaussian.view(E, F, -1).sum(dim=-1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

