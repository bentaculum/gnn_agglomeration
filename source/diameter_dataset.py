from torch_geometric.data import Data

from random_graph_dataset import RandomGraphDataset
# from diameter_graph import create_random_graph
from diameter_graph import DiameterGraph


class DiameterDataset(RandomGraphDataset):
    def __init__(self, root, config):

        self.check_config_vars = [
            'samples',
            'nodes',
            'self_loops',
            'feature_dimensionality',
            'euclidian_dimensionality',
            'theta_max',
            'theta',
            'msts',
            'affinity_dist_beta',
            'affinity_dist_alpha',
            'class_noise',
        ]

        super(DiameterDataset, self).__init__(
            root=root, config=config)

    def create_datapoint(self):
        graph = DiameterGraph()
        graph.create_random_graph(config=self.config)
        return graph

    def update_config(self, config):
        # TODO check if local update necessary
        config.classes = config.msts
        self.config.classes = config.classes

