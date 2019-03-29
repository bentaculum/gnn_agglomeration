import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn_model import GnnModel


class GcnClassification(GnnModel):
    def __init__(self, config):
        super(GcnClassification, self).__init__(config)
        self.loss_name = 'NLL loss'

    def layers(self):
        self.conv1 = GCNConv(self.config.euclidian_dimensionality, self.config.hidden_units)
        self.conv2 = GCNConv(self.config.hidden_units, self.config.max_neighbors + 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = getattr(F, self.config.hidden_activation)(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets)
        return self.current_loss

    def evaluate_metric(self, data):
        # put model in evaluation mode
        self.eval()
        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes
        # print('\nACCURACY: {:.4f}'.format(acc))
        return acc

    def evaluate_as_list(self, data):
        self.eval()
        _, pred = self.forward(data).max(dim=1)
        return pred.tolist()




