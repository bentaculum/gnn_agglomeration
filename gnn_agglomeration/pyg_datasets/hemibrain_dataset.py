import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import json
import os
import daisy
import configparser

from .hemibrain_graph import HemibrainGraph

class HemibrainDataset(Dataset):
    def __init__(self, root, config, length, roi_offset, roi_shape):
        self.config = config
        self.len = length
        self.roi_offset = roi_offset
        self.roi_shape = roi_shape
        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(HemibrainDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)

        with open(config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        # Graph provider
        # TODO fully parametrize once necessary
        self.graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name=config.db_name,
            host=pw_parser['DEFAULT']['db_host'],
            mode='r',
            nodes_collection=config.nodes_collection,
            edges_collection=config.edges_collection,
            endpoint_names=['u', 'v'],
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'])

        # TODO possible?
        # self.check_dataset_vs_config()

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
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        random_offset = np.zeros(3, dtype=np.int_)
        random_offset[0] = np.random.randint(
            low=0, high=self.roi_shape[0] - self.config.block_size[0])
        random_offset[1] = np.random.randint(
            low=0, high=self.roi_shape[1] - self.config.block_size[1])
        random_offset[2] = np.random.randint(
            low=0, high=self.roi_shape[2] - self.config.block_size[2])
        total_offset = self.roi_offset + random_offset

        graph = HemibrainGraph()
        graph.read_and_process(
            graph_provider=self.graph_provider,
            block_offset=total_offset,
            block_shape=self.config.block_size
        )

        # TODO dynamic data augmentation

        return graph

    # TODO not necessary unless I save the processed graphs to file again
    def check_dataset_vs_config(self):
        pass

    # def check_dataset_vs_config(self):
    #     with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'r') as json_file:
    #         data_config = json.load(json_file)
    #     run_conf_dict = vars(self.config)
    #     for key in self.check_config_vars:
    #         if key in data_config:
    #             assert run_conf_dict[key] == data_config[key],\
    #                 'Run config does not match dataset config\nrun_conf_dict[{}]={}, data_config[{}]={}'.format(
    #                 key, run_conf_dict[key], key, data_config[key])

    def update_config(self, config):
        # TODO check if local update necessary
        if config.edge_labels:
            config.classes = 2
            self.config.classes = 2
        else:
            config.classes = config.msts
            self.config.classes = config.classes

        # TODO do this again once processed dataset is saved to file
        # with open(os.path.join(self.config.dataset_abs_path, 'config.json'), 'w') as f:
        #     json.dump(vars(self.config), f)

    def print_summary(self):
        pass

    def targets_mean_std(self):
        # TODO this should be preprocessed and saved to file for large datasets
        targets = []
        for i in range(self.__len__()):
            targets.extend(self.get(i).y)

        targets = np.array(targets)
        return np.mean(targets), np.std(targets)
