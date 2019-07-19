from .random_graph_dataset import RandomGraphDataset
from .iterative_graph import IterativeGraph


class IterativeDataset(RandomGraphDataset):
    def __init__(self, root, config):

        self.check_config_vars = [
            'samples',
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

        super(IterativeDataset, self).__init__(
            root=root, config=config)

    def create_datapoint(self):
        graph = IterativeGraph()
        graph.create_random_graph(config=self.config)
        return graph

    def update_config(self, config):
        # TODO check if local update necessary
        if config.edge_labels:
            config.classes = 2
            self.config.classes = 2
        else:
            config.classes = config.msts
            self.config.classes = config.classes
