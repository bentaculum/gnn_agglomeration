import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import matplotlib.pyplot as plt
import os


class DiameterGraph(Data):

    def __init__(self, config):
        super(DiameterGraph, self).__init__()
        self.config = config

        self.create_random_graph()

    def create_random_graph(self):
        pos_list = []
        diameter_list = []
        class_list = []
        noisy_class_list = []
        assert self.config.nodes % self.config.msts == 0
        # one hot encoded class labels + diameter are the nodes features
        assert self.config.feature_dimensionality == self.config.msts + 1

        # create multiple graphs
        for i in range(self.config.msts):
            nodes = self.config.nodes / self.config.msts
            pos = torch.rand(nodes,
                    self.config.euclidian_dimensionality)
            pos_list.extend(pos)

            g = nx.Graph(nodes)

            for i_n in range(nodes):
                for j_n in range(i_n + 1, nodes):
                    # fully connected graph
                    # add edge, with distance as edge weight
                    g.add_edge(i_n, j_n, weight=torch.dist(pos[i_n], pos[j_n]))

            # build an mst
            mst = nx.minimum_spanning_edges(g)
            edges = list(mst)
            sorted(edges)

            # iterate with BFS over the MST, assign descending diameters according to that
            diameters = np.full(nodes, None, dtype=float)
            diameters[0] = 1.0
            to_visit = [0]

            while len(to_visit) > 0:
                v = to_visit.pop(0)
                neighbors = [e[1] if e[0] == v for e in edges] + [e[0] if e[1] == v for e in edges]
                for n in neighbors:
                    if diameters[n]:
                        continue
                    # random multiplicator in [0.8,1.0)
                    diameters[n] = diameters[v] * (np.random.rand(1)[0] / 5 + 0.8)
                    to_visit.append(n)

            diameter_list.extend(list(diameters))
            class_list.extend([i] * nodes)

            # the root is fixed, not noisy
            root = np.zeros(self.config.msts)
            root[i] = 1
            noisy_class_list.append(root)

            # for all other nodes, the class label is drawn from a multinomial
            pvals = np.full(self.config.msts, self.config.class_noise / (self.config.msts - 1))
            pvals[i] = 1 - self.config.class_noise
            assert np.sum(pvals) == 1.0
            noisy_labels = np.random.multinomial(n=1, pvals=pvals, size = nodes-1)
            noisy_class_list.extend(list(noisy_labels))


        edges_list = []
        affinities_list = []
        # Beta distributions to sample affinities
        # TODO parametrize to config
        beta0 = torch.distributions.beta.Beta(1,4)
        beta1 = torch.distributions.beta.Beta(4,1)

        # connect all nodes within distance theta_max
        for i in range(self.config.nodes):
            for j in range(i + 1, self.config.nodes):
                node1 = pos[i]
                node2 = pos[j]
                if torch.dist(node1, node2) < self.config.theta_max:
                    # add bi-directed edges, sample affinity
                    edges_list.append([i, j])
                    edges_list.append([j, i])
                    if class_list[i] == class_list[j]:
                        affinities_list.append(beta1.sample(torch.Size([1])))
                    else:
                        affinities_list.append(beta0.sample(torch.Size([1])))


        # Cast all the data to torch tensors
        x_list = [noisy_class_list[i].append(diameter_list[i]) for i in range(nodes)]
        self.x = torch.tensor(x_list, dtype=torch.float)
        self.edge_index = torch.tensor(edges_list, dtype=torch.long).transpose(0, 1)
        self.edge_attr = torch.tensor(affinities_list, dtype=torch.float)
        self.y = torch.tensor(class_list, dtype=torch.long)
        self.pos = pos


    # TODO adapt these functions


    # def plot(self):
    #     g = nx.Graph(
    #         incoming_graph_data=self.data.edge_index.transpose(0, 1).tolist())
    #     # add the positions in euclidian space to the model
    #     pos_dict = {}
    #     # prepare the targets to be displayed
    #     labels_dict = {}
    #
    #     for i in range(self.data.x.size(0)):
    #         pos_dict[i] = self.data.x[i].tolist()
    #         labels_dict[i] = int(self.data.y[i].item())
    #
    #     self.set_plotting_style()
    #     nx.draw_networkx(g, pos_dict, labels=labels_dict)
    #     plt.title("Number of neighbors within euclidian distance {}".format(
    #         self.config.theta))
    #     plt.savefig(os.path.join(self.config.run_abs_path, 'graph.png'))
    #     # plt.show()
    #
    # def plot_predictions(self, pred, graph_nr):
    #     # transpose the edge matrix for format requirements
    #     g = nx.Graph(
    #         incoming_graph_data=self.data.edge_index.transpose(0, 1).tolist())
    #     # add the positions in euclidian space to the model
    #     pos_dict = {}
    #     # prepare the targets to be displayed
    #     labels_dict = {}
    #
    #     # TODO this is a quick fix for two node classes. Generalize!
    #     node_color = ['r' if features[0] == 0 else 'y' for features in self.data.x]
    #
    #     for i in range(self.data.pos.size(0)):
    #         pos_dict[i] = self.data.pos[i].tolist()
    #         if self.config.euclidian_dimensionality == 1:
    #             pos_dict[i].append(0)
    #
    #         labels_dict[i] = '{};{}'.format(
    #             int(pred[i]), int(self.data.y[i].item()))
    #
    #     self.set_plotting_style()
    #     nx.draw_networkx(g, pos_dict, labels=labels_dict, node_color=node_color, font_size=10)
    #     plt.title(
    #         "Number of neighbors within euclidian distance {}.\nEach node displays 'pred:target'".format(
    #             self.config.theta))
    #
    #     self.add_to_plotting_style()
    #     img_path = os.path.join(self.config.run_abs_path,
    #                             'graph_with_predictions_{}.png'.format(graph_nr))
    #     if os.path.isfile(img_path):
    #         os.remove(img_path)
    #     plt.savefig(img_path)
    #     print('plotted the graph with predictions to {}'.format(img_path))
    #     # plt.show()
    #
    # def set_plotting_style(self):
    #     plt.figure(figsize=(8, 8))
    #     plt.xlabel('x (euclidian)')
    #     plt.ylabel('y (euclidian)')
    #     plt.xlim(-0.1, 1.1)
    #     plt.ylim(-0.1, 1.1)
    #
    # def add_to_plotting_style(self):
    #     plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    #     plt.tick_params(axis='y', which='both', left=True, labelleft=True)
    #     plt.grid(linestyle='--', color='gray')
