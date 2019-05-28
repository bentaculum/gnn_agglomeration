import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

from my_graph import MyGraph


class IterativeGraph(MyGraph):

    # can't overwrite __init__ using different args than base class

    # Restricted to 2D euclidian space at the moment

    # TODO might not need this one
    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        deg = np.arctan2(y, x) * 180 / np.pi
        return rho, deg

    def pol2cart(self, rho, deg):
        x = rho * np.cos(deg / 180 * np.pi)
        y = rho * np.sin(deg / 180 * np.pi)
        return x, y

    def close_to_borders(self, node, theta):
        for i in node:
            if i < theta or 1 - i < theta:
                return True
        return False

    def create_single_component(self, config):
        # TODO This is a chain, for now, extend this to tree

        # variable number of nodes in graph now. We need indices of the start
        # nodes
        pos = []
        degs = []
        diams = []

        # first position at least theta_max away from all unit square borders
        pos.append(np.random.rand(2) *
                   (1 - 2 * config.theta_max) + config.theta_max)
        degs.append(0)
        diams.append(1)

        next_rho = np.random.rand() * config.theta
        next_deg = np.random.rand() * 360
        next_node = pos[-1] + np.array(self.pol2cart(next_rho, next_deg))
        pos.append(next_node)
        degs.append(next_deg)

        # Random multiplier between 0.8 and 1.0
        diams.append(diams[-1] * (np.random.rand() / 5 + 0.8))

        more_nodes = not self.close_to_borders(pos[-1], config.theta)
        while more_nodes:
            next_rho = np.random.rand() * config.theta
            next_deg = (degs[-1] + (np.random.rand() - 0.5) *
                        2 * config.curvature_degree_limit) % 360
            next_node = pos[-1] + np.array(self.pol2cart(next_rho, next_deg))
            pos.append(next_node)
            degs.append(next_deg)
            diams.append(diams[-1] * (np.random.rand() / 5 + 0.8))

            more_nodes = not self.close_to_borders(pos[-1], config.theta)

        return pos, diams

    def create_random_graph(self, config):

        if config.euclidian_dimensionality is not 2:
            raise NotImplementedError(
                "IterativeGraph is only implemented for 2D euclidian")

        pos_list = []
        diameter_list = []
        class_list = []
        noisy_class_list = []
        if config.class_label_feature:
            # one hot encoded class labels + diameter are the nodes features
            assert config.feature_dimensionality == config.msts + 1
        else:
            assert config.feature_dimensionality == 1
            # TODO think about writing the euclidian coordinates to the nodes

        # TODO update for n-D edge features
        if config.data_transform == 'Distance':
            assert config.pseudo_dimensionality == 2
        else:
            assert config.pseudo_dimensionality == 3

        ground_truth = []
        roots_list = []

        # create multiple graphs
        for i in range(config.msts):
            pos, diams = self.create_single_component(config)

            # create edges, globally numbered
            offset = len(pos_list)
            roots_list.append(offset)
            edges = [[x + offset, x + 1 + offset] for x in range(len(pos) - 1)]

            # add positions and diameters
            pos_list.extend(pos)
            diameter_list.extend(diams)

            # rename all nodes in MST for global numbering
            ground_truth.extend(edges)

            class_list.extend([i] * len(pos))

            if config.class_label_feature:
                # the root is fixed, not noisy
                root = np.zeros(config.msts)
                root[i] = 1
                noisy_class_list.append(root)

                # for all other nodes, the class label is drawn from a
                # multinomial
                pvals = np.full(
                    config.msts, config.class_noise / (config.msts - 1))
                pvals[i] = 1 - config.class_noise
                noisy_labels = np.random.multinomial(
                    n=1, pvals=pvals, size=len(pos) - 1)
                noisy_class_list.extend(list(noisy_labels))

        ###########################

        self.ground_truth = torch.tensor(ground_truth, dtype=torch.long)
        self.roots = torch.tensor(roots_list, dtype=torch.long)

        edges_list = []
        affinities_list = []
        # Beta distributions to sample affinities
        beta0 = torch.distributions.beta.Beta(
            config.affinity_dist_alpha, config.affinity_dist_beta)
        beta1 = torch.distributions.beta.Beta(
            config.affinity_dist_beta, config.affinity_dist_alpha)

        def all_affinities(n1, n2):
            if class_list[n1] == class_list[n2]:
                aff = beta1.sample(torch.Size([1]))
                gt = 1
            else:
                aff = beta0.sample(torch.Size([1]))
                gt = 0
            # append twice, as the graph is bi-directed
            return [aff, aff], gt

        def only_gt_affinities(n1, n2):
            if [n1, n2] in ground_truth:
                aff = beta1.sample(torch.Size([1]))
                gt = 1
            else:
                aff = beta0.sample(torch.Size([1]))
                gt = 0
            return [aff, aff], gt

        # TODO adapt this for trees
        def only_gt_dir_affinities(n1, n2):
            if [n1, n2] in ground_truth:
                aff = beta1.sample(torch.Size([1]))
                gt = 1
            else:
                aff = beta0.sample(torch.Size([1]))
                gt = 0
            return [aff, 0], gt

        ground_truth_affinities = np.zeros(len(ground_truth))
        ground_truth_edge_labels = []

        # connect all nodes, regardless of subgraph, within distance theta_max
        total_nodes = len(pos_list)
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                node1 = torch.tensor(pos_list[i], dtype=torch.float)
                node2 = torch.tensor(pos_list[j], dtype=torch.float)
                if torch.dist(node1, node2) < config.theta_max:
                    # add bi-directed edges, sample affinity
                    edges_list.append([i, j])
                    edges_list.append([j, i])
                    new_aff, edge_label = locals()[config.affinities](i, j)
                    affinities_list.extend(new_aff)
                    ground_truth_edge_labels.append(edge_label)

                    # Save ground truth affinities for plotting
                    if [i, j] in ground_truth:
                        index = ground_truth.index([i, j])
                        ground_truth_affinities[index] = new_aff[0]

        self.ground_truth_affinities = torch.tensor(
            ground_truth_affinities, dtype=torch.float)
        ########################################

        # Cast all the data to torch tensors
        if config.class_label_feature:
            x_list = []
            for i in range(total_nodes):
                li = list(noisy_class_list[i])
                li.append(diameter_list[i])
                x_list.append(li)
        else:
            x_list = diameter_list

        self.x = torch.tensor(x_list, dtype=torch.float)
        if self.x.dim() == 1:
            self.x = self.x.unsqueeze(-1)

        self.edge_index = torch.tensor(
            edges_list, dtype=torch.long).transpose(0, 1)
        self.edge_attr = torch.tensor(affinities_list, dtype=torch.float)

        if config.edge_labels:
            self.y = torch.tensor(ground_truth_edge_labels, dtype=torch.long)
        else:
            self.y = torch.tensor(class_list, dtype=torch.long)

        self.pos = torch.tensor(pos_list, dtype=torch.float)

    def plot_predictions(self, config, pred, graph_nr, run, acc, logger):
        # add the positions in euclidian space to the model
        if config.edge_labels:
            self.plot_predictions_on_edges(
                config=config,
                pred=pred,
                graph_nr=graph_nr,
                run=run,
                acc=acc,
                logger=logger
            )
        else:
            self.plot_predictions_on_nodes(
                config=config,
                pred=pred,
                graph_nr=graph_nr,
                run=run,
                acc=acc,
                logger=logger
            )
        # TODO move duplicate code here

    def plot_predictions_on_edges(
            self,
            config,
            pred,
            graph_nr,
            run,
            acc,
            logger):
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        if self.x.dim() == 1:
            node_size = (self.x * 500).tolist()
        else:
            node_size = (self.x[:, -1].squeeze() * 500).tolist()

        node_color = np.zeros(self.pos.size(0), dtype=np.int_)
        # node_size = [200] * self.pos.size(0)

        prev_root = self.roots[0].item()
        roots_list = self.roots[1:].tolist()
        roots_list.append(self.pos.size(0))
        for i, r in enumerate(roots_list):
            node_color[prev_root:r] = i
            # node_size[prev_root:r.item()] = list(np.linspace(500, 200, r.item() - prev_root))

            prev_root = r
        node_color = node_color.tolist()

        for i in range(self.pos.size(0)):
            pos_dict[i] = self.pos[i].tolist()
            if config.euclidian_dimensionality == 1:
                pos_dict[i].append(0)

            if config.class_label_feature:
                labels_dict[i] = '{}'.format(
                    int(self.x[i][:config.msts].max(0)[1]))

        ax = self.set_plotting_style(config=config)
        g = nx.empty_graph(n=len(self.pos), create_using=nx.Graph())

        slicing_list = np.array(pred).astype(np.bool_).tolist()
        # TODO quick fix: edges are assumed to be ordered
        every_other = [True, False] * int(self.edge_index.size(1) / 2)
        unique_edges = self.edge_index.transpose(0, 1)[every_other]
        unique_edges = unique_edges.cpu().numpy().astype(np.int_)
        pred_edges = unique_edges[slicing_list]
        g.add_edges_from(pred_edges.tolist())

        nx.draw_networkx_edges(
            g,
            pos=pos_dict,
            edgelist=self.ground_truth.tolist(),
            edge_color='r',
            width=5)
        nx.draw_networkx(
            g,
            pos_dict,
            labels=labels_dict,
            node_color=node_color,
            cmap=cm.Paired,
            vmin=0.0,
            vmax=float(
                config.msts),
            font_size=10,
            ax=ax,
            with_labels=True,
            node_size=node_size)

        plt.title(
            """Recovery of ground truth edges, based on 'descending diameter' and noisy affinities(edge widths).
            Input class labels are correct with prob {}. All nodes within distance {} are
            connected in input graph. The shown colors depict ground truth,
            each root is brown. Node label shows the noisy_input""".format(
                1 - config.class_noise, config.theta_max))
        plt.text(0.6, 1.0, 'Accuracy: {0:.3f}'.format(acc), fontsize=16)
        plt.legend(loc='upper left', fontsize=12)

        self.add_to_plotting_style()
        img_path = os.path.join(
            config.run_abs_path,
            'graph_with_predictions_{}.png'.format(graph_nr))
        if os.path.isfile(img_path):
            os.remove(img_path)
        plt.savefig(img_path)
        run.add_artifact(filename=img_path,
                         name='graph_with_predictions_{}.png'.format(graph_nr))
        logger.debug(
            'plotted the graph with predictions to {}'.format(img_path))

    def plot_predictions_on_nodes(
            self,
            config,
            pred,
            graph_nr,
            run,
            acc,
            logger):
        pos_dict = {}
        # prepare the targets to be displayed
        labels_dict = {}

        node_color = [int(i.item()) for i in self.y]
        # Special color for all roots
        for i in self.roots:
            node_color[i] = config.msts

        for i in range(self.pos.size(0)):
            pos_dict[i] = self.pos[i].tolist()
            if config.euclidian_dimensionality == 1:
                pos_dict[i].append(0)

            labels_dict[i] = '{}:{}'.format(
                int(pred[i]), int(self.x[i][:config.msts].max(0)[1]))

        ax = self.set_plotting_style(config=config)
        g = nx.empty_graph(n=len(self.pos), create_using=nx.Graph())
        g.add_edges_from(self.ground_truth.tolist())

        widths = (self.ground_truth_affinities * 2).tolist()

        nx.draw_networkx(
            g,
            pos_dict,
            labels=labels_dict,
            node_color=node_color,
            cmap=cm.Paired,
            vmin=0.0,
            vmax=float(
                config.msts),
            font_size=10,
            ax=ax,
            with_labels=True,
            width=widths)
        plt.title(
            """Recovery of class label, based on 'descending diameter' and noisy affinities(edge widths).
            Input class labels are correct with prob {}. All nodes within distance {} are
            connected in input graph. The shown MSTS and colors depict ground truth,
            each root is brown. Node label is of format 'pred:noisy_input'""".format(
                1 - config.class_noise, config.theta_max))
        plt.text(0.6, 1.0, 'Accuracy: {0:.3f}'.format(acc), fontsize=16)
        plt.legend(loc='upper left', fontsize=12)

        self.add_to_plotting_style()
        img_path = os.path.join(
            config.run_abs_path,
            'graph_with_predictions_{}.png'.format(graph_nr))
        if os.path.isfile(img_path):
            os.remove(img_path)
        plt.savefig(img_path)
        run.add_artifact(filename=img_path,
                         name='graph_with_predictions_{}.png'.format(graph_nr))
        logger.debug(
            'plotted the graph with predictions to {}'.format(img_path))

    def set_plotting_style(self, config):
        f = plt.figure(figsize=(8, 8))
        plt.xlabel('x (euclidian)')
        plt.ylabel('y (euclidian)')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        ax = f.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Paired')
        cNorm = colors.Normalize(vmin=0, vmax=float(config.msts))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        for i in range(config.msts):
            ax.plot([0], [0], color=scalarMap.to_rgba(i), label=i)
        return ax

    def add_to_plotting_style(self):
        plt.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        plt.tick_params(axis='y', which='both', left=True, labelleft=True)
        plt.grid(linestyle='--', color='gray')
