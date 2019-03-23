import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GnnExample(torch.nn.Module):
    def __init__(self, config):
        super(GnnExample, self).__init__()
        self.config = config
        self.conv1 = GCNConv(self.config.dimensionality, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


