import torch
from torch_geometric.data import InMemoryDataset
from source.my_graph import MyGraph

class RandomGraphDataset(InMemoryDataset):
    def __init__(self, root, config, transform=None, pre_transform=None):
        self.config = config
        super(RandomGraphDataset, self).__init__(root, transform, pre_transform)
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
        graph = MyGraph(self.config)
        print('Creating a new random graph ... ')
        graph.create_random_graph()
        data_list = [graph.data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

