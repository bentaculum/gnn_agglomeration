import logging
import torch
from gunpowder import *
import daisy
import numpy as np
from time import time as now
import math
from abc import ABC, abstractmethod

# TODO how to import from beyond top level path?
from . import utils  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.getLogger('gunpowder.nodes.').setLevel(logging.DEBUG)


class SiameseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, patch_size, raw_channel, mask_channel, num_workers=5):
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
        self.num_workers = num_workers
        assert raw_channel or mask_channel

        self.load_rag()
        self.init_pipeline()

    def load_rag(self):
        # TODO parametrize the used names
        self.id_field = 'id'
        self.node1_field = 'u'
        self.node2_field = 'v'

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

        # TODO parametrize block size
        block_size = (np.array(roi.get_shape())/2).astype(np.int_)

        nodes_attrs, edges_attrs = graph_provider.read_blockwise(
            roi=roi,
            block_size=daisy.Coordinate(block_size),
            num_workers=self.num_workers
        )
        logger.debug(f'read whole graph in {now() - start} s')

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

    @abstractmethod
    def init_pipeline(self):
        pass

    def __len__(self):
        return self.len

    def get_patch(self, center, node_id):
        """
        get volumetric patch using gunpowder
        Args:
            center(tuple of ints): center of mass for the node
            node_id(int): node id from db

        Returns:
            patch as torch.Tensor, with one or more channels

        """
        offset = np.array(center) - np.array(self.patch_size) / 2
        roi = Roi(offset=offset, shape=self.patch_size)
        roi = roi.snap_to_grid(Coordinate(config.voxel_size), mode='closest')
        # logger.debug(f'ROI snapped to grid: {roi}')
        # with build(self.pipeline) as p:
        request = BatchRequest()
        if self.raw_channel:
            request[self.raw_key] = ArraySpec(roi=roi)
        if self.mask_channel:
            request[self.labels_key] = ArraySpec(roi=roi)

        batch = self.built_pipeline.request_batch(request)

        channels = []
        if self.raw_channel:
            raw_array = batch[self.raw_key].data
            # logger.debug(f'raw_array shape {raw_array.shape}')
            channels.append(raw_array)
        if self.mask_channel:
            labels_array = batch[self.labels_key].data
            # logger.debug(f'labels_array shape {labels_array.shape}')
            labels_array = (labels_array == node_id).astype(np.float32)
            # sanity check: is there overlap?
            # TODO request new pair if fragment not contained?
            # No,because that's also going to appear in the inference
            logger.debug(f'overlap: {labels_array.sum()} voxels')
            channels.append(labels_array)

        tensor = torch.tensor(channels, dtype=torch.float)

        # Not necessary: Add the `channel`-dimension
        # if len(channels) == 1:
        # tensor = tensor.unsqueeze(0)

        return tensor

    @abstractmethod
    def __getitem__(self, index):
        pass
