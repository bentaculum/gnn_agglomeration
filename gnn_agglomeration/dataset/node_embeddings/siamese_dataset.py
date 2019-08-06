import logging
import torch
import daisy
import time
import numpy as np

from gnn_agglomeration import utils
# dataset configs for all many params
from ..config import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SiameseDataset(torch.utils.data.Data):
    """
    Each data point is actually a mini-batch of volume pairs
    """

    def __init__(self, length, patch_size, raw_channel, mask_channel):
        """

        Args:
            length:
            patch_size:
            raw_channel:
            mask_channel:
        """
        self.len = length
        self.patch_size = patch_size
        self.raw_channel = raw_channel
        self.mask_channel = mask_channel
        assert raw_channel or mask_channel

        # TODO parametrize the used names
        self.id_field = 'id'
        self.node1_field = 'u'
        self.node2_field = 'v'

        # connect to one RAG DB
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

        # get all edges, including gt_merge_score, as dict of numpy arrays
        start = time.time()
        roi = daisy.Roi(offset=config.roi_offset, shape=config.roi_shape)
        nodes = graph_provider.read_nodes(roi=roi)
        edges = graph_provider.read_edges(roi=roi, nodes=nodes)
        self.nodes_attrs = utils.to_np_arrays(nodes)
        self.edges_attrs = utils.to_np_arrays(edges)
        logger.info(f'read whole graph in {time.time() - start}s')

    def __len__(self):
        return self.len

    def get_patch(self, center, node_id):
        offset = np.array(center) - np.array(self.patch_size)/2
        roi = daisy.Roi(offset=offset, shape=self.patch_size)

        channels = []
        if self.raw_channel:
            ds = daisy.open_ds(config.groundtruth_zarr, 'volumes/raw/s0')
            if not ds.roi.contains(roi):
                logger.warning(f'location {center} is not fully contained in dataset')
            patch = (ds[roi].to_ndarray()/255.0).astype(np.float32)
            channels.append(patch)

        if self.mask_channel:
            ds = daisy.open_ds(config.fragments_zarr, config.fragments_ds)
            if not ds.roi.contains(roi):
                logger.warning(f'location {center} is not fully contained in dataset')
            patch = ds[roi].to_ndarray()
            mask = (patch == node_id).astype(np.float32)
            channels.append(mask)

        return torch.tensor(channels, dtype=torch.float)

    def __getitem__(self, index):
        """

        Args:
            index(int): not used here, needed for inheritance

        Returns:

            a mini-batch of volume pairs

        """
        # pick random node
        index = np.random.randint(0, len(self.nodes_attrs[self.id_field]))
        node_id = self.nodes_attrs[self.id_field][index]
        node_center = (self.nodes_attrs['center_z'][index], self.nodes_attrs['center_y'][index], self.nodes_attrs['center_x'][index])
        # get raw data for the node
        node_patch = self.get_patch(center=node_center)

        # get all neighbors of random node in RAG, considering edges in both `directions`
        neighbor_mask1 = self.edges_attrs[self.node1_field] == node_id
        neighbors1 = self.edges_attrs[self.node2_field][neighbor_mask1]
        neighbor_mask2 = self.edges_attrs[self.node2_field] == node_id
        neighbors2 = self.edges_attrs[self.node1_field][neighbor_mask2]

        neighbor_ids = np.unique(np.concatenate((neighbors1, neighbors2)))
        neighbor_indices = np.in1d(self.nodes_attrs[self.id_field], neighbor_ids).nonzero()[0]
        # TODO parametrize
        centers = [
            (self.nodes_attrs['center_z'][i], self.nodes_attrs['center_y'][i], self.nodes_attrs['center_x'][i])
            for i in neighbor_indices
        ]

        # get the raw data for each neighbor, with daisy, using Nils's synful script
        patches = []
        for c, i in zip(centers, neighbor_ids):
            patches.append(self.get_patch(center=c, node_id=i))

        # TODO make pairs of data, plus corresponding labels
        # use torch Data class

        return None
