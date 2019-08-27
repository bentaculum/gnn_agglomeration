import torch
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
import numpy as np
import daisy
import configparser
from abc import ABC, abstractmethod
import logging
import os
import pymongo
import time
import bson
import multiprocessing
from time import time as now
import pickle

from ..data_transforms import *
from gnn_agglomeration import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# hack to make daisy logging indep from from sacred logging
logging.getLogger(
    'daisy.persistence.mongodb_graph_provider').setLevel(logging.INFO)


class HemibrainDataset(Dataset, ABC):
    def __init__(
            self,
            root,
            config,
            db_name,
            embeddings_collection,
            roi_offset,
            roi_shape,
            length=None,
            save_processed=False):
        self.config = config
        self.db_name = db_name
        self.embeddings_collection = embeddings_collection
        self.roi_offset = np.array(roi_offset, dtype=np.int_)
        self.roi_shape = np.array(roi_shape, dtype=np.int_)
        self.len = length
        self.save_processed = save_processed

        self.connect_to_db()
        self.load_node_embeddings()
        self.load_all_nodes()
        self.prepare()

        data_augmentation = globals()[config.data_augmentation](config=config)
        coordinate_transform = getattr(
            T, config.data_transform)(norm=True, cat=True)
        transform = T.Compose([data_augmentation, coordinate_transform])
        super(HemibrainDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)

    def load_all_nodes(self):
        """
        Needed to add node position to edges that go out of the cube
        """
        start = now()
        with open(self.config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        client = pymongo.MongoClient(pw_parser['DEFAULT']['db_host'])
        db = client[self.db_name]
        collection = db[self.config.nodes_collection]

        # TODO parametrize field names
        self.all_nodes = {}
        for line in collection.find():
            self.all_nodes[line['id']] = {
                'center_z': line['center_z'],
                'center_y': line['center_y'],
                'center_x': line['center_x']
            }

        logger.info(f'load all nodes from db in {now() - start} s')

    def load_node_embeddings(self):

        if self.embeddings_collection is None:
            self.embeddings = None
            return

        start = now()
        with open(self.config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        client = pymongo.MongoClient(pw_parser['DEFAULT']['db_host'])
        db = client[self.db_name]
        collection = db[self.embeddings_collection]

        # TODO parametrize field names
        self.embeddings = {}
        for line in collection.find():
            self.embeddings[line['id']] = pickle.loads(line['embedding'])

        logger.info(f'load all node embeddings in {now() - start} s')

    def prepare(self):
        pass

    def pad_block(self, offset, shape):
        """
        Enlarge the block with padding in all dimensions.
        Crop the enlarged block if the new block is not contained in the ROI
        """
        offset_padded = np.array(offset, dtype=np.int_) - np.array(self.config.block_padding, dtype=np.int_)
        shape_padded = np.array(shape, dtype=np.int_) + 2 * \
            np.array(self.config.block_padding, dtype=np.int_)
        logger.debug(
            f'offset padded: {offset_padded}, shape padded: {shape_padded}')
        return self.crop_block(offset_padded, shape_padded)

    def crop_block(self, offset, shape):
        """

        Args:
            offset (numpy.array): padded offset
            shape (numpy.array): padded shape

        Returns:
            cropped offset, cropped shape

        """
        # lower corner
        cropped_offset = np.maximum(self.roi_offset, offset)
        # correct shape for cropping
        cropped_shape = shape - (cropped_offset - offset)

        # upper corner
        cropped_shape = np.minimum(
            self.roi_offset + self.roi_shape,
            cropped_offset + cropped_shape) - cropped_offset

        logger.debug(
            f'offset cropped: {cropped_offset}, shape cropped: {cropped_shape}')
        return cropped_offset, cropped_shape

    def connect_to_db(self):
        with open(self.config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        # Graph provider
        # TODO fully parametrize once necessary
        self.graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name=self.db_name,
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
        return [f'processed_data_{i}.pt' for i in range(self.len)]

    def process_one(self, idx):
        if not os.path.isfile(self.processed_paths[idx]):
            data = self.get_from_db(idx)
            torch.save(data, self.processed_paths[idx])

    def process(self):
        logger.info(f'Trying to load data from {self.root} ...')
        start = time.time()

        pool = multiprocessing.Pool(
            processes=self.config.num_workers,
            initializer=np.random.seed,
            initargs=())
        pool.map_async(
            func=self.process_one,
            iterable=range(self.len))
        pool.close()
        pool.join()

        logger.info(f'processed {self.len} in {time.time() - start}s')

        # with open(
        # os.path.join(
        # self.config.dataset_abs_path, 'config.json'), 'w') as f:
        #     json.dump(vars(self.config), f)

    def _download(self):
        pass

    def _process(self):
        if self.save_processed:
            if not os.path.isdir(self.processed_dir):
                os.makedirs(self.processed_dir)

            self.process()

    def get(self, idx):
        if self.save_processed:
            return torch.load(self.processed_paths[idx])
        else:
            start = now()
            g = self.get_from_db(idx)
            logger.debug(f'get graph from db in {now() - start} s')
            return g

    @abstractmethod
    def get_from_db(self, idx):
        pass

    def write_outputs_to_db(self, outputs_dict, collection_name):
        logger.info('writing to db ...')
        start = now()
        outputs_dict = {(min(k), max(k)): v for k, v in outputs_dict.items()}
        logger.info(
            f'lower id first in all edges in outputs_dict in {now() - start} s')

        start = now()
        roi = daisy.Roi(list(self.roi_offset), list(self.roi_shape))
        # TODO parametrize block size
        block_size = (np.array(roi.get_shape())/2).astype(np.int_)

        orig_node_attrs, orig_edge_attrs = self.graph_provider.read_blockwise(
            roi=roi,
            block_size=daisy.Coordinate(block_size),
            num_workers=self.config.num_workers
        )

        # TODO parametrize the used names
        id_field = 'id'
        node1_field = 'u'
        node2_field = 'v'

        # keep only needed columns
        nodes_cols = [id_field, 'center_z', 'center_y', 'center_x']
        orig_node_attrs = {k: orig_node_attrs[k] for k in nodes_cols}

        edges_cols = [node1_field, node2_field]
        orig_edge_attrs = {k: orig_edge_attrs[k] for k in edges_cols}

        logger.info(
            f'edges before dropping edges going out of the dataset: {len(orig_edge_attrs[node1_field])}')
        # drop edges at the border
        # there shouldn't be any, as the center of mass of the fragments cannot be outside of the ROI
        utils.drop_outgoing_edges(
            node_attrs=orig_node_attrs,
            edge_attrs=orig_edge_attrs,
            id_field=id_field,
            node1_field=node1_field,
            node2_field=node2_field
        )
        logger.info(
            f'edges after dropping edges going out of the dataset: {len(orig_edge_attrs[node1_field])}')
        logger.info(f'load original RAG, drop edges in {now() - start} s')

        # lower id in all edges first
        start = now()
        for i, t in enumerate(zip(orig_edge_attrs[node1_field], orig_edge_attrs[node2_field])):
            orig_edge_attrs[node1_field][i] = min(t)
            orig_edge_attrs[node2_field][i] = max(t)
        logger.info(
            f'lower id first in all edges in orig_edge_attrs in {now() - start} s')

        logger.info(
            f'num edges in ROI {len(orig_edge_attrs[node1_field])}, num outputs {len(outputs_dict)}')
        assert len(orig_edge_attrs[node1_field]) >= len(outputs_dict)

        # TODO insert dummy value 1 for all edges that are not in outputs_dict,
        # but part of full RAG
        start = now()
        counter = 0
        missing_edges_pos = []
        for e_tuple in zip(orig_edge_attrs[node1_field], orig_edge_attrs[node2_field]):
            if e_tuple not in outputs_dict:
                # TODO this is just for debugging, terrible style
                u_idx = np.where(orig_node_attrs[id_field] == [
                                 e_tuple[0]])[0][0]
                u_pos = (orig_node_attrs['center_z'][u_idx],
                         orig_node_attrs['center_y'][u_idx],
                         orig_node_attrs['center_x'][u_idx])
                v_idx = np.where(orig_node_attrs[id_field] == [
                                 e_tuple[1]])[0][0]
                v_pos = (orig_node_attrs['center_z'][v_idx],
                         orig_node_attrs['center_y'][v_idx],
                         orig_node_attrs['center_x'][v_idx])
                missing_edges_pos.append((u_pos, v_pos))

                # TODO parametrize the dummy value 1
                outputs_dict[e_tuple] = torch.tensor(
                    1, dtype=torch.float).item()
                counter += 1

        np.savez_compressed(
            os.path.join(self.config.run_abs_path, "missing_edges_pos.npz"),
            missing_edges_pos=np.array(missing_edges_pos))
        logger.info(f'added {counter} dummy values in {now() - start} s')

        assert len(orig_edge_attrs[node1_field]) == len(outputs_dict),\
            f'num edges in ROI {len(orig_edge_attrs[node1_field])}, num outputs including dummy values {len(outputs_dict)}'

        with open(self.config.db_host, 'r') as f:
            pw_parser = configparser.ConfigParser()
            pw_parser.read_file(f)

        client = pymongo.MongoClient(pw_parser['DEFAULT']['db_host'])
        db = client[self.db_name]
        collection = db[collection_name]

        start = now()
        insertion_elems = []
        # TODO parametrize field names
        for (u, v), merge_score in outputs_dict.items():
            insertion_elems.append(
                {'u': bson.Int64(u), 'v': bson.Int64(v), 'merge_score': float(merge_score)})
        collection.insert_many(insertion_elems, ordered=False)
        logger.info(
            f'insert predicted merge_scores in {now() - start}s')

    def targets_mean_std(self):
        """
        Not possible to estimate target mean and variance for a dataset that
        is not in memory without doing a preliminary pass. For randomly
        fetched RAG snippets you could estimate on n pre-fetched graphs
        """
        raise NotImplementedError(
            'Online mean and variance estimation not implemented')

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
