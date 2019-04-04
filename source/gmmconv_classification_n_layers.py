import torch
import torch.nn.functional as F
from torch_geometric.nn import GMMConv
from gnn_model import GnnModel


class GmmConvClassification(GnnModel):
    def __init__(self, config):
        super(GmmConvClassification, self).__init__(config)
        self.loss_name = 'NLL loss'

    def layers(self):
        self.conv_in = GMMConv(
            in_channels=self.config.feature_dimensionality,
            out_channels=self.config.hidden_units,
            dim=self.config.pseudo_dimensionality)

        self.hidden_layers = []
        for i in range(self.config.hidden_layers):
            l = GMMConv(
                in_channels=self.config.hidden_units,
                out_channels=self.config.hidden_units,
                dim=self.config.pseudo_dimensionality)
            self.hidden_layers.append(l)

        self.conv_out = GMMConv(
            in_channels=self.config.hidden_units,
            out_channels=self.config.max_neighbors + 1,
            dim=self.config.pseudo_dimensionality)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv_in(x=x, edge_index=edge_index, pseudo=edge_attr)
        x = getattr(F, self.config.hidden_activation)(x)
        x = F.dropout(x, training=self.training)

        for l in self.hidden_layers:
            x = l(x=x, edge_index=edge_index, pseudo=edge_attr)
            x = getattr(F, self.config.hidden_activation)(x)
            x = F.dropout(x, training=self.training)

        x = self.conv_out(x=x, edge_index=edge_index, pseudo=edge_attr)

        return F.log_softmax(x, dim=1)

    def loss(self, inputs, targets):
        self.current_loss = F.nll_loss(inputs, targets)
        return torch.mean(self.current_loss)

    def evaluate_metric(self, data):
        # put model in evaluation mode
        self.eval()
        _, pred = self.forward(data).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes
        return acc

    def evaluate_as_list(self, data):
        self.eval()
        _, pred = self.forward(data).max(dim=1)
        return pred.tolist()




