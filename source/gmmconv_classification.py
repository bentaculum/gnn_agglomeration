import torch
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from gnn_model import GnnModel


class GmmConvClassification(GnnModel):
    def __init__(self, config):
        super(GmmConvClassification, self).__init__(config)
        self.loss_name = 'NLL loss'

    def layers(self):
        self.conv1 = GMMConv(
            in_channels=self.config.dimensionality,
            out_channels=self.config.hidden_units,
            dim=1)
        self.conv2 = GMMConv(
            in_channels=self.config.hidden_units,
            out_channels=self.config.max_neighbors + 1,
            dim=1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

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
        print('\nACCURACY: {:.4f}'.format(acc))

    def evaluate_as_list(self, data):
        self.eval()
        _, pred = self.forward(data).max(dim=1)
        return pred.tolist()




