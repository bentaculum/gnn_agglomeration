import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os

class MyGraph():
    # TODO let this class inherit directly from torch_geometric.data.Data

    def __init__(self, config, data=None):
        self.config = config
        self.data = data

    def create_random_graph(self):
        x = torch.rand(self.config.nodes, self.config.dimensionality)
        # TODO might want to save the position in the pos attribute, not the node feature matrix

        # connect all edges within distance theta_max O(n^2)
        edges = []
        edge_attr = []
        y = torch.zeros(self.config.nodes, dtype=torch.long)
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

    def plot(self):
        g = nx.Graph(incoming_graph_data=self.data.edge_index.transpose(0,1).tolist())
        # add the positions in euclidian space to the model
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        for i in range(self.data.x.size()[0]):
            pos_dict[i] = self.data.x[i].tolist()
            labels_dict[i] = int(self.data.y[i].item())

        self.set_plotting_style()
        nx.draw_networkx(g, pos_dict, labels=labels_dict)
        plt.title("Number of neighbors within euclidian distance {}".format(
            self.config.theta))
        plt.savefig(os.path.join(self.config.temp_dir, 'graph.png'))
        plt.show()

    def plot_predictions(self, pred):
        # transpose the edge matrix for format requirements
        g = nx.Graph(incoming_graph_data=self.data.edge_index.transpose(0,1).tolist())
        # add the positions in euclidian space to the model
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        for i in range(self.data.x.size()[0]):
            pos_dict[i] = self.data.x[i].tolist()
            labels_dict[i] = '{};{}'.format(int(pred[i]), int(self.data.y[i].item()))

        self.set_plotting_style()
        nx.draw_networkx(g, pos_dict, labels=labels_dict, font_size=10)
        plt.title("Number of neighbors within euclidian distance {}.\nEach node displays 'pred:target'".format(
            self.config.theta))
        plt.savefig(os.path.join(self.config.temp_dir, 'graph_with_predictions.png'))
        plt.show()

    def set_plotting_style(self):
        plt.figure(figsize=(8, 8))
        plt.xlabel('x (euclidian)')
        plt.ylabel('y (euclidian)')

