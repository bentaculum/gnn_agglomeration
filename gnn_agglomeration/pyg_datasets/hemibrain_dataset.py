import torch
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
import numpy as np
import daisy
import configparser
from abc import ABC, abstractmethod


class HemibrainDataset(Dataset, ABC):
    def __init__(self, root, config, roi_offset, roi_shape, length=None):
        self.config = config
        self.roi_offset = roi_offset
        self.roi_shape = roi_shape
        self.len = length

        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(HemibrainDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)

        self.pad_total_roi()
        self.connect_to_db()

        # TODO possible?
        # self.check_dataset_vs_config()

    def pad_total_roi(self):
        # pad the entire volume, padded area not part of total roi any more
        self.roi_offset = np.array(self.roi_offset) + np.array(self.config.block_padding)
        self.roi_shape = np.array(self.roi_shape) - 2 * np.array(self.config.block_padding)

    def pad_block(self, offset, shape):
        # enlarge the block with padding in all dimensions
        offset_padded = np.array(offset) - np.array(self.config.block_padding)
        shape_padded = np.array(shape) + 2 * np.array(self.config.block_padding)
        return offset_padded, shape_padded

    def connect_to_db(self):
        with open(self.config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        # Graph provider
        # TODO fully parametrize once necessary
        self.graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name=self.config.db_name,
            host=pw_parser['DEFAULT']['db_host'],
            mode='r',
            nodes_collection=self.config.nodes_collection,
            edges_collection=self.config.edges_collection,
            endpoint_names=['u', 'v'],
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'])

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

    @abstractmethod
    def get(self, idx):
        pass

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
