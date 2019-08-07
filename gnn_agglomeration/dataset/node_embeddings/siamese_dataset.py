import logging
import torch
import daisy
from time import time as now
import numpy as np
from time import time as now

import sys
# sys.path.insert(1, '..')
from . import utils  # noqa
# sys.path.remove('..')

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SiameseDataset(torch.utils.data.Dataset):
    """
    Each data point is actually a mini-batch of volume pairs
    """

    def __init__(self, length, patch_size, raw_channel, mask_channel, transform=None):
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
        self.transform = transform
        assert raw_channel or mask_channel

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
        nodes = graph_provider.read_nodes(roi=roi)
        edges = graph_provider.read_edges(roi=roi, nodes=nodes)
        logger.debug(f'read whole graph in {now() - start} s')

        start = now()
        nodes_attrs = utils.to_np_arrays(nodes)
        nodes_cols = [self.id_field, 'center_z', 'center_y', 'center_x']
        self.nodes_attrs = {k: nodes_attrs[k] for k in nodes_cols}

        edges_attrs = utils.to_np_arrays(edges)
        edges_cols = [self.node1_field, self.node2_field, config.new_edge_attr_trinary]
        edges_attrs = {k: edges_attrs[k] for k in edges_cols}

        self.edges_attrs = utils.drop_outgoing_edges(
            node_attrs=self.nodes_attrs,
            edge_attrs=edges_attrs,
            id_field=self.id_field,
            node1_field=self.node1_field,
            node2_field=self.node2_field,
        )

        logger.debug(
            f'convert graph to numpy arrays and drop outgoing edges in {now() - start} s')

    def __len__(self):
        return self.len

    def get_patch(self, center, node_id):
        offset = np.array(center) - np.array(self.patch_size) / 2
        roi = daisy.Roi(offset=offset, shape=self.patch_size)
        # center might not be on the grid defined by the voxel_size
        roi = roi.snap_to_grid(daisy.Coordinate(config.voxel_size), mode='closest')

        if roi.get_shape()[0] != self.patch_size[0]:
            logger.warning(
                f'''correct roi shape {roi.get_shape()} to 
                {self.patch_size} after snapping to grid''')
            roi.set_shape(self.patch_size)

        channels = []
        if self.raw_channel:
            ds = daisy.open_ds(config.groundtruth_zarr, 'volumes/raw/s0')
            if not ds.roi.contains(roi):
                logger.warning(f'location {center} is not fully contained in dataset')
                return None
            patch = (ds[roi].to_ndarray() / 255.0).astype(np.float32)
            channels.append(patch)

        if self.mask_channel:
            ds = daisy.open_ds(config.fragments_zarr, config.fragments_ds)
            if not ds.roi.contains(roi):
                logger.warning(f'location {center} is not fully contained in dataset')
                return None
            patch = ds[roi].to_ndarray()
            mask = (patch == node_id).astype(np.float32)
            channels.append(mask)

        tensor = torch.tensor(channels, dtype=torch.float)

        # Add the `channel`-dimension
        if len(channels) == 1:
            tensor = tensor.unsqueeze(0)

        return tensor

    def __getitem__(self, index):
        """

        Args:
            index(int): not used here, needed for inheritance

        Returns:

            a mini-batch of volume pairs

        """
        start_getitem = now()
        # pick random node
        index = np.random.randint(0, len(self.nodes_attrs[self.id_field]))
        node_id = self.nodes_attrs[self.id_field][index]
        node_center = (
            self.nodes_attrs['center_z'][index],
            self.nodes_attrs['center_y'][index],
            self.nodes_attrs['center_x'][index])

        # get raw data for the node
        node_patch = self.get_patch(center=node_center, node_id=node_id)
        if node_patch is None:
            logger.warning(f'patch for center node is not fully contained in ROI, try again')
            return self.__getitem__(index=index)

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
        # handle edges with label unknown
        patches = []
        labels = []
        for c, i, idx in zip(centers, neighbor_ids, neighbor_indices):
            if self.edges_attrs[config.new_edge_attr_trinary][idx] is not 1 and \
                    self.edges_attrs[config.new_edge_attr_trinary][idx] is not 0:
                continue

            neighbor_patch = self.get_patch(center=c, node_id=i)
            if neighbor_patch is None:
                continue
            else:
                patches.append(neighbor_patch)

            # assign labels: 1 if the two fragments belong to same neuron, -1 if not
            edge_score = self.edges_attrs[config.new_edge_attr_trinary][idx]
            if edge_score == 0:
                labels.append(torch.tensor(1))
            elif edge_score == 1:
                labels.append(torch.tensor(-1))
            else:
                raise ValueError(f'Value {edge_score} cannot be transformed into a valid label')

        if len(patches) == 0:
            logger.warning(f'all neighbor patches are not fully contained in ROI, try again')
            return self.__getitem__(index=index)

        # make pairs of data, plus corresponding labels
        # TODO check if torch.Tensor.expand(len_patches, -1, -1, -1, -1)
        #  works as well, it should use less memory

        input0 = torch.tensor(node_patch).float()
        input0 = input0.repeat(len(patches), 1, 1, 1, 1)
        # input0 = input0.expand(len(patches), 1, 1 ,1 ,1)
        input1 = torch.stack(patches).float()
        labels = torch.stack(labels).float()

        if self.transform is not None:
            input0 = self.transform(input0)
            input1 = self.transform(input1)

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return input0, input1, labels
