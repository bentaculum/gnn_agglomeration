import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os

from my_graph import MyGraph


class CountNeighborsGraph(MyGraph):

    # can't overwrite __init__ using different args than base class

    def create_random_graph(self, config):
        m = torch.distributions.beta.Beta(2, 3.5)
        pos = m.sample(torch.Size(
            [config.nodes, config.euclidian_dimensionality]))
        # pos = torch.rand(self.config.nodes,
        #                  self.config.euclidian_dimensionality)

        # connect all edges within distance theta_max O(n^2)
        edges = []
        y = torch.zeros(config.nodes, dtype=torch.long)
        x = torch.arange(
            config.nodes) % config.feature_dimensionality

        for i in range(config.nodes):
            for j in range(i + 1 - int(config.self_loops),
                           config.nodes):
                node1 = pos[i]
                node2 = pos[j]
                # print(torch.dist(node1, node2))
                if torch.dist(node1, node2) < config.theta_max:
                    # add bi-directed edges to use directed pseudo-coordinates
                    edges.append([i, j])
                    edges.append([j, i])
                    # if distance < theta, count the nodes as a neighbor in
                    # euclidian space
                    if torch.dist(node1, node2) < config.theta:
                        # Only if the two nodes belong to the same node class,
                        # and if it's not the same node,
                        # increase the target
                        if (x[i] == x[j]) and i != j:
                            y[i] += 1
                            y[j] += 1

        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1)

        # x = torch.ones(self.config.nodes, self.config.feature_dimensionality)
        # One hot encoded representation might be better, as this is
        # categorical data
        x = torch.nn.functional.one_hot(
            x, config.feature_dimensionality).float()

        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.pos = pos

    def plot(self):
        # TODO adapt
        g = nx.Graph(
            incoming_graph_data=self.data.edge_index.transpose(0, 1).tolist())
        # add the positions in euclidian space to the model
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        for i in range(self.data.x.size(0)):
            pos_dict[i] = self.data.x[i].tolist()
            labels_dict[i] = int(self.data.y[i].item())

        self.set_plotting_style()
        nx.draw_networkx(g, pos_dict, labels=labels_dict)
        plt.title("Number of neighbors within euclidian distance {}".format(
            self.config.theta))
        plt.savefig(os.path.join(self.config.run_abs_path, 'graph.png'))
        # plt.show()

    def plot_predictions(self, config, pred, graph_nr, run, acc):
        # TODO this is a quick fix for two node classes. Generalize!
        if config.classes > 2:
            raise NotImplementedError('Plotting not generalized to k classes')

        # transpose the edge matrix for format requirements
        g = nx.Graph(
            incoming_graph_data=self.edge_index.transpose(0, 1).tolist())
        # add the positions in euclidian space to the model
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        node_color = ['r' if features[0] ==
                      0 else 'y' for features in self.x]

        for i in range(self.pos.size(0)):
            pos_dict[i] = self.pos[i].tolist()
            if config.euclidian_dimensionality == 1:
                pos_dict[i].append(0)

            labels_dict[i] = '{};{}'.format(
                int(pred[i]), int(self.y[i].item()))

        self.set_plotting_style()
        nx.draw_networkx(
            g,
            pos_dict,
            labels=labels_dict,
            node_color=node_color,
            font_size=10)
        plt.title(
            "Number of neighbors within euclidian distance {}.\nEach node displays 'pred:target'".format(
                config.theta))

        self.add_to_plotting_style()
        img_path = os.path.join(
            config.run_abs_path,
            'graph_with_predictions_{}.png'.format(graph_nr))
        if os.path.isfile(img_path):
            os.remove(img_path)
        plt.savefig(img_path)
        run.add_artifact(filename=img_path,
                         name='graph_with_predictions_{}.png'.format(graph_nr))
        print('plotted the graph with predictions to {}'.format(img_path))

    def set_plotting_style(self):
        plt.figure(figsize=(8, 8))
        plt.xlabel('x (euclidian)')
        plt.ylabel('y (euclidian)')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

    def add_to_plotting_style(self):
        plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        plt.tick_params(axis='y', which='both', left=True, labelleft=True)
        plt.grid(linestyle='--', color='gray')
