import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import json
import os
# from abc import ABC, abstractmethod

# TODO for testing
from .iterative_dataset import IterativeDataset

class HemibrainDataset(Dataset):
    def __init__(self, root, config, length):
        self.config = config
        self.len = length
        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(HemibrainDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)

        # TODO necessary?
        # self.data, self.slices = torch.load(self.processed_paths[0])

        # TODO possible?
        # self.check_dataset_vs_config()

        # TODO for testing
        self.ds = IterativeDataset(root=config.dataset_abs_path, config=config)
        with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)

    def __len__(self):
        return self.len

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        g = self.ds.create_datapoint()
        return g

    # def process(self):
    #     # Read data into huge `Data` list.
    #     data_list = []
    #     # TODO use sacred logger
    #     print('Creating {} new random graphs ... '.format(self.config.samples))
    #     for i in range(self.config.samples):
    #         print('Create graph {} ...'.format(i))
    #         graph = self.create_datapoint()
    #         data_list.append(graph)
    #
    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])
    #
    #     with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'w') as f:
    #         json.dump(vars(self.config), f)

    # TODO do after every get?
    def check_dataset_vs_config(self):
        with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'r') as json_file:
            data_config = json.load(json_file)
        run_conf_dict = vars(self.config)
        for key in self.check_config_vars:
            if key in data_config:
                assert run_conf_dict[key] == data_config[key],\
                    'Run config does not match dataset config\nrun_conf_dict[{}]={}, data_config[{}]={}'.format(
                    key, run_conf_dict[key], key, data_config[key])

    # TODO for testing
    def update_config(self, config):
        # TODO check if local update necessary
        if config.edge_labels:
            config.classes = 2
            self.config.classes = 2
        else:
            config.classes = config.msts
            self.config.classes = config.classes


    # TODO figure out what to do with these extra methods I have defined for previous datasets
    # def update_config(self, config):
    #     pass

    def print_summary(self):
        pass

    def targets_mean_std(self):
        # TODO this should be preprocessed and saved to file for large datasets
        targets = []
        for i in range(self.__len__()):
            targets.extend(self.get(i).y)

        targets = np.array(targets)
        return np.mean(targets), np.std(targets)
