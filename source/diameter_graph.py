import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os


# TODO write things into the torch_geometric.data.Data object, then call a function on it
def create_random_graph(config, data):
    pos_list = torch.rand(config.nodes, config.euclidian_dimensionality, dtype=torch.float)
    diameter_list = []
    class_list = []
    noisy_class_list = []
    assert config.nodes % config.msts == 0
    nodes = int(config.nodes / config.msts)
    # one hot encoded class labels + diameter are the nodes features
    assert config.feature_dimensionality == config.msts + 1

    ground_truth = []
    # create multiple graphs
    for i in range(config.msts):

        g = nx.empty_graph(n=nodes, create_using=nx.Graph())

        # label globally, don't start at 0
        numbering_dict = {}
        for j in range(nodes):
            numbering_dict[j] = i * nodes + j
        g = nx.relabel.relabel_nodes(g, numbering_dict)

        for i_n in range(nodes):
            for j_n in range(i_n + 1, nodes):
                # fully connected graph
                # add edge, with distance as edge weight
                g.add_edge(i_n, j_n, weight=torch.dist(
                    pos_list[i * nodes + i_n], pos_list[i * nodes + j_n]))

        # build an mst
        # TODO save it
        # mst = nx.minimum_spanning_edges(g, data=False)
        mst = nx.minimum_spanning_tree(g)
        # edges = list(mst)
        edges = mst.edges(data=False)
        sorted(edges)

        # iterate with DFS over the MST, assign descending diameters according to that
        diameters = [None] * nodes
        diameters[0] = 1.0

        def dfs(v):
            nonlocal diameters
            neighbors = [e[1] for e in edges if e[0] == v] + [e[0] for e in edges if e[1] == v]
            children = [n for n in neighbors if diameters[n] is None]
            if len(children) == 0:
                return
            for c in children:
                # random multiplicator in [0.8,1.0)
                diameters[c] = diameters[v] * (np.random.rand(1)[0] / 5 + 0.8)
                # diameters
                dfs(c)

        dfs(0)

        # rename all nodes in MST for global numbering
        mst_relabeled = nx.relabel.relabel_nodes(mst, numbering_dict)
        edges_relabeld = sorted(mst_relabeled.edges(data=False))
        ground_truth.extend(edges_relabeld)

        diameter_list.extend(diameters)
        class_list.extend([i] * nodes)

        # the root is fixed, not noisy
        root = np.zeros(config.msts)
        root[i] = 1
        noisy_class_list.append(root)

        # for all other nodes, the class label is drawn from a multinomial
        pvals = np.full(config.msts, config.class_noise / (config.msts - 1))
        pvals[i] = 1 - config.class_noise
        assert np.sum(pvals) == 1.0
        noisy_labels = np.random.multinomial(n=1, pvals=pvals, size=nodes-1)
        noisy_class_list.extend(list(noisy_labels))

    data.ground_truth = torch.tensor(ground_truth, dtype=torch.long)

    edges_list = []
    affinities_list = []
    # Beta distributions to sample affinities
    beta0 = torch.distributions.beta.Beta(config.affinity_dist_alpha, config.affinity_dist_beta)
    beta1 = torch.distributions.beta.Beta(config.affinity_dist_beta, config.affinity_dist_alpha)

    # connect all nodes, regardless of subgraph, within distance theta_max
    for i in range(config.nodes):
        for j in range(i + 1, config.nodes):
            node1 = pos_list[i]
            node2 = pos_list[j]
            if torch.dist(node1, node2) < config.theta_max:
                # add bi-directed edges, sample affinity
                edges_list.append([i, j])
                edges_list.append([j, i])
                if class_list[i] == class_list[j]:
                    aff = beta1.sample(torch.Size([1]))
                    # append twice, as the graph is bi-directed
                    affinities_list.append(aff)
                    affinities_list.append(aff)
                else:
                    aff = beta0.sample(torch.Size([1]))
                    # append twice, as the graph is bi-directed
                    affinities_list.append(aff)
                    affinities_list.append(aff)

    # Cast all the data to torch tensors
    x_list = []
    for i in range(config.nodes):
        li = list(noisy_class_list[i])
        li.append(diameter_list[i])
        x_list.append(li)

    data.x = torch.tensor(x_list, dtype=torch.float)
    data.edge_index = torch.tensor(edges_list, dtype=torch.long).transpose(0, 1)
    data.edge_attr = torch.tensor(affinities_list, dtype=torch.float)
    data.y = torch.tensor(class_list, dtype=torch.long)
    data.pos = pos_list

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


def plot_predictions(config, data, pred, graph_nr):
    # add the positions in euclidian space to the model
    pos_dict = {}
    # prepare the targets to be displayed
    labels_dict = {}

    # TODO this is a quick fix for two node classes. Generalize!
    node_color = ['r' if i.item() == 0 else 'y' for i in data.y]
    # This should be all the same per MST

    for i in range(data.pos.size(0)):
        pos_dict[i] = data.pos[i].tolist()
        if config.euclidian_dimensionality == 1:
            pos_dict[i].append(0)

        # pred vs input noisy label
        labels_dict[i] = '{}:{}'.format(
            int(pred[i]), int(data.x[i][:config.msts].max(0)[1]))

    set_plotting_style()
    g = nx.empty_graph(n=config.nodes, create_using=nx.Graph())
    g.add_edges_from(data.ground_truth.tolist())
    nx.draw_networkx(g, pos_dict, labels=labels_dict, node_color=node_color, font_size=10)
    plt.title(
        """Recovery of class label, based on 'descending diameter' and noisy affinities.
        Input class labels are correct with prob {}. Red is 0, Yellow is 1.
        All nodes within distance {} are connected in input graph.
        Color and the MSTs represent ground truth, node label is of format 'pred:noisy_input'""".format(
            1 - config.class_noise, config.theta_max))

    add_to_plotting_style()
    img_path = os.path.join(config.run_abs_path,
                            'graph_with_predictions_{}.png'.format(graph_nr))
    if os.path.isfile(img_path):
        os.remove(img_path)
    plt.savefig(img_path)
    print('plotted the graph with predictions to {}'.format(img_path))
    # plt.show()


def set_plotting_style():
    plt.figure(figsize=(8, 8))
    plt.xlabel('x (euclidian)')
    plt.ylabel('y (euclidian)')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)


def add_to_plotting_style():
    plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True, labelleft=True)
    plt.grid(linestyle='--', color='gray')
