import logging
import torch
from gunpowder import *
import daisy
import numpy as np
from time import time as now
import math
from abc import ABC, abstractmethod
import pickle

# TODO how to import from beyond top level path?
from . import utils  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.getLogger('gunpowder.nodes.').setLevel(logging.DEBUG)


class SiameseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
            self,
            patch_size,
            raw_channel,
            mask_channel,
            raw_mask_channel,
            num_workers=5,
            in_memory=True,
            rag_block_size=None,
            rag_from_file=None,
            dump_rag=None):
        """
        connect to db, load and weed out edges, define gunpowder pipeline
        Args:
            patch_size:
            raw_channel (bool): if set true, load patch from raw volumetric data
            mask_channel (bool): if set true, load patch from fragments volumetric data
            num_workers (int): number of workers available, e.g. for loading RAG
        """
        self.patch_size = patch_size
        self.raw_channel = raw_channel
        self.mask_channel = mask_channel
        self.raw_mask_channel = raw_mask_channel
        self.num_workers = num_workers
        self.in_memory = in_memory
        assert raw_channel or mask_channel or raw_mask_channel

        self.load_rag(
            rag_block_size=rag_block_size,
            rag_from_file=rag_from_file,
            dump_rag=dump_rag
        )
        self.init_pipeline()
        self.build_pipeline()

    def load_rag(self, rag_block_size, rag_from_file, dump_rag):

        # TODO parametrize the used names
        self.id_field = 'id'
        self.node1_field = 'u'
        self.node2_field = 'v'

        if rag_from_file:
            start = now()
            self.nodes_attrs, self.edges_attrs = pickle.load(
                open(rag_from_file, 'rb'))
            logger.info(f'load rag from {rag_from_file} in {now() - start} s')
            return

        # connect to one RAG DB
        logger.debug('ready to connect to RAG db')
        start = now()
        graph_provider = daisy.persistence.MongoDbGraphProvider(
            db_name=config.db_name,
            host=config.db_host,
            mode='r',
            nodes_collection=config.nodes_collection,
            edges_collection=config.edges_collection,
            endpoint_names=['u', 'v'],
            position_attribute=[
                'center_z',
                'center_y',
                'center_x'])
        logger.debug(f'connect to RAG db in {now() - start} s')

        # get all edges, including gt_merge_score, as dict of numpy arrays
        start = now()
        roi = daisy.Roi(offset=config.roi_offset, shape=config.roi_shape)
        logger.info(roi)
        rag_block_size = daisy.Coordinate(rag_block_size)
        logger.info(f'rag_block_size {rag_block_size}')

        nodes_attrs, edges_attrs = graph_provider.read_blockwise(
            roi=roi,
            block_size=rag_block_size,
            num_workers=self.num_workers
        )
        logger.info(f'read whole graph in {now() - start} s')

        start = now()
        nodes_cols = [self.id_field, 'center_z', 'center_y', 'center_x']
        self.nodes_attrs = {k: nodes_attrs[k] for k in nodes_cols}

        edges_cols = [self.node1_field, self.node2_field,
                      config.new_edge_attr_trinary]
        edges_attrs = {k: edges_attrs[k] for k in edges_cols}

        logger.debug(
            f'num edges before dropping: {len(edges_attrs[self.node1_field])}')

        edges_attrs = utils.drop_outgoing_edges(
            node_attrs=self.nodes_attrs,
            edge_attrs=edges_attrs,
            id_field=self.id_field,
            node1_field=self.node1_field,
            node2_field=self.node2_field,
        )

        # filter edges, we only want edges labeled 0 or 1
        edge_filter = (edges_attrs[config.new_edge_attr_trinary] == 1) | \
            (edges_attrs[config.new_edge_attr_trinary] == 0)

        for attr, vals in edges_attrs.items():
            edges_attrs[attr] = vals[edge_filter]
        self.edges_attrs = edges_attrs

        logger.debug(f'num nodes: {len(self.nodes_attrs[self.id_field])}')
        logger.debug(f'num edges: {len(self.edges_attrs[self.node1_field])}')
        logger.debug(
            f'''convert graph to numpy arrays, drop outgoing edges
            and filter edges in {now() - start} s''')

        if dump_rag:
            start_dump = now()
            pickle.dump((self.nodes_attrs, self.edges_attrs),
                        open(dump_rag, 'wb'))
            logger.info(f'dump rag to {dump_rag} in {now() - start_dump} s')

    @abstractmethod
    def init_pipeline(self):
        pass

    def build_pipeline(self):
        logger.info('start building pipeline')
        start = now()
        self.built_pipeline = build(self.pipeline)
        self.batch_provider = self.built_pipeline.__enter__()
        logger.info(f'built pipeline in {now() - start} s')

    def __len__(self):
        return self.len

    @abstractmethod
    def get_batch(self, center, node_id):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
