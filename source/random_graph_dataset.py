import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import numpy as np

from my_graph import MyGraph
from diameter_graph import DiameterGraph


class RandomGraphDataset(InMemoryDataset):
    def __init__(self, root, config, transform=None, pre_transform=None):
        self.config = config
        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(RandomGraphDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        raise NotImplementedError('Dataset not available for download')

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        print('Creating {} new random graphs ... '.format(self.config.samples))
        for i in range(self.config.samples):
            # TODO parametrize
            print('Create graph {} ...'.format(i))
            graph = DiameterGraph(config=self.config)
            graph.create_random_graph()
            data_list.append(graph.data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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

        return dic

    def targets_mean_std(self):
        # TODO this should be preprocessed and saved to file for large datasets
        targets = []
        for i in range(self.__len__()):
            targets.extend(self.get(i).y)

        targets = np.array(targets)
        return np.mean(targets), np.std(targets)
