import torch
from torch_geometric.data import Data


class RandomGraph():
    def __init__(self, config):
        self.config = config
        self.data = None

    def create_graph(self):
        x = torch.rand(self.config.nodes, self.config.dimensionality)
        # TODO might want to save the position in the pos attribute, not the node feature matrix

        # connect all edges within distance theta_max O(n^2)
        edges = []
        edge_attr = []
        y = torch.zeros(self.config.nodes)
        for i in range(x.size()[0]):
            for j in range(i+1, x.size()[0]):
                node1 = x[i]
                node2 = x[j]
                # print(torch.dist(node1, node2))
                if torch.dist(node1, node2) < self.config.theta_max:
                    edges.append([i,j])
                    edge_attr.append(torch.dist(node1, node2))
                    y[i] += 1
                    y[j] += 1

        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0,1)
        edge_attr = torch.tensor(edge_attr)
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
