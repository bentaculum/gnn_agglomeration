import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import json
import os
from abc import ABC, abstractmethod


class RandomGraphDataset(InMemoryDataset, ABC):
    def __init__(self, root, config):
        self.config = config
        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(RandomGraphDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.check_dataset_vs_config()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        raise NotImplementedError('Dataset not available for download')

    @abstractmethod
    def create_datapoint(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        #TODO use sacred logger
        print('Creating {} new random graphs ... '.format(self.config.samples))
        for i in range(self.config.samples):
            print('Create graph {} ...'.format(i))
            graph = self.create_datapoint()
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)

    def check_dataset_vs_config(self):
        with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'r') as json_file:
            data_config = json.load(json_file)
        run_conf_dict = vars(self.config)
        for key in self.check_config_vars:
            if key in data_config:
                assert run_conf_dict[key] == data_config[key],\
                    'Run config does not match dataset config\nrun_conf_dict[{}]={}, data_config[{}]={}'.format(
                    key, run_conf_dict[key], key, data_config[key])

    def update_config(self, config):
        pass

    def print_summary(self):
        pass

    def targets_mean_std(self):
        # TODO this should be preprocessed and saved to file for large datasets
        targets = []
        for i in range(self.__len__()):
            targets.extend(self.get(i).y)

        targets = np.array(targets)
        return np.mean(targets), np.std(targets)
