import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from . import gnn_model

class GnnExample(gnn_model.GnnModel):
    def __init__(self, config):
        super(GnnExample, self).__init__(config)

    def layers(self):
        self.conv1 = GCNConv(self.config.dimensionality, self.config.hidden_units)
        self.conv2 = GCNConv(self.config.hidden_units, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def loss(self, inputs, targets):
        return F.mse_loss(inputs, targets)

    def evaluate_metric(self, data):
        # put model in evaluation mode
        self.eval()
        pred = self.forward(data).round()
        correct = torch.squeeze(pred).eq(data.y).sum().item()
        acc = correct / data.num_nodes
        print('Accuracy: {:.4f}'.format(acc))




