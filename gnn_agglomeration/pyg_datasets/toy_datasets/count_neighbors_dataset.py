import torch

from .random_graph_dataset import RandomGraphDataset
from .count_neighbors_graph import CountNeighborsGraph


class CountNeighborsDataset(RandomGraphDataset):
    def __init__(self, root, config):

        self.check_config_vars = [
            'samples',
            'nodes',
            'self_loops',
            'feature_dimensionality',
            'euclidian_dimensionality',
            'theta_max',
            'theta',
        ]

        super(CountNeighborsDataset, self).__init__(
            root=root, config=config)

    def create_datapoint(self):
        graph = CountNeighborsGraph()
        graph.create_random_graph(config=self.config)
        return graph

    def update_config(self, config):
        # TODO check if local update necessary
        config.max_neighbors = self.max_neighbors()
        config.classes = config.max_neighbors + 1
        self.config.max_neighbors = config.max_neighbors
        self.config.classes = config.classes

    def print_summary(self):
        self.neighbors_distribution()

    def max_neighbors(self):
        # Detect maximum number of neighbors
        neighbors = 0
        for i in range(self.__len__()):
            neighbors = max(neighbors, torch.max(self.get(i).y).item())

        return int(neighbors)

    def neighbors_distribution(self):
        # histogram for no of neighbors within distance theta
        dic = {}
        for i in range(self.__len__()):
            graph_targets = self.get(i).y
            for t in graph_targets:
                t_int = t.item()
                if t_int in dic:
                    dic[t_int] += 1
                else:
                    dic[t_int] = 1

        print('Maximum # of neighbors within distance {} in dataset: {}'.format(
            self.config.theta, self.config.max_neighbors))
        print('# of neighbors, distribution:')
        for key, value in sorted(dic.items(), key=lambda x: x[0]):
            print("{} : {}".format(key, value))
        print('')
