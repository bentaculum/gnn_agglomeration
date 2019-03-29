import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_model import GnnModel


class GcnRegression(GnnModel):
    def __init__(self, config):
        super(GcnRegression, self).__init__(config)
        self.loss_name = 'MSE loss'

    def layers(self):
        self.conv1 = GCNConv(self.config.euclidian_dimensionality, self.config.hidden_units)
        self.conv2 = GCNConv(self.config.hidden_units, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = getattr(F, self.config.hidden_activation)(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def loss(self, inputs, targets):
        self.current_loss =  F.mse_loss(inputs, targets.float())
        return self.current_loss

    def evaluate_metric(self, data):
        # put model in evaluation mode
        self.eval()
        pred = self.forward(data).round()
        correct = torch.squeeze(pred).eq(data.y.float()).sum().item()
        acc = correct / data.num_nodes
        # print('\nAccuracy: {:.4f}'.format(acc))
        return acc

    def evaluate_as_list(self, data):
        self.eval()
        pred = self.forward(data).round()
        return torch.squeeze(pred).tolist()

