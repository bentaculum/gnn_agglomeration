import logging
import torch
from gunpowder import *
import daisy
import numpy as np
from time import time as now
import math

# TODO how to import from beyond top level path?
from . import utils  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SiameseDataset(torch.utils.data.Dataset):
    """
    Each data point is actually a mini-batch of volume pairs
    """

    def __init__(self, patch_size, raw_channel, mask_channel, num_workers=5, transform=None):
        """

        Args:
            patch_size:
            raw_channel:
            mask_channel:
        """
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
        logger.info(roi)

        # TODO parametrize block size
        block_size = (np.array(roi.get_shape())/2).astype(np.int_)

        nodes_attrs, edges_attrs = graph_provider.read_blockwise(
            roi=roi,
            block_size=daisy.Coordinate(block_size),
            num_workers=num_workers
        )
        logger.debug(f'read whole graph in {now() - start} s')

        start = now()
        nodes_cols = [self.id_field, 'center_z', 'center_y', 'center_x']
        self.nodes_attrs = {k: nodes_attrs[k] for k in nodes_cols}

        edges_cols = [self.node1_field, self.node2_field, config.new_edge_attr_trinary]
        edges_attrs = {k: edges_attrs[k] for k in edges_cols}

        logger.debug(f'num edges before dropping: {len(edges_attrs[self.node1_field])}')

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

        # assign dataset length
        self.len = len(self.edges_attrs[config.new_edge_attr_trinary])

        logger.debug(f'num nodes: {len(self.nodes_attrs[self.id_field])}')
        logger.debug(f'num edges: {len(self.edges_attrs[self.node1_field])}')
        logger.debug(
            f'''convert graph to numpy arrays, drop outgoing edges 
            and filter edges in {now() - start} s''')

        # get weights
        start = now()
        targets = self.edges_attrs[config.new_edge_attr_trinary]
        class_sample_count = np.array(
            [len(np.where(targets == t)[0]) for t in np.unique(targets)]
        )
        logger.debug(f'class sample counts {class_sample_count}')
        weights = 1.0 / class_sample_count
        samples_weights = weights[targets.astype(np.int_)]
        self.samples_weights = torch.from_numpy(samples_weights).float()
        logger.debug(f'assign sample weights in {now() - start} s')

        # gunpowder init
        self.raw_key = ArrayKey('RAW')
        self.labels_key = ArrayKey('LABELS')

        self.source = (
            ZarrSource(
                config.groundtruth_zarr,
                datasets={self.raw_key: config.groundtruth_ds},
                array_specs={self.raw_key: ArraySpec(interpolatable=True)}) +
            Normalize(self.raw_key) +
            Pad(self.raw_key, None),
            ZarrSource(
                config.fragments_zarr,
                datasets={self.labels_key: config.fragments_ds},
                array_specs={self.labels_key: ArraySpec(interpolatable=True)}) +
            Normalize(self.labels_key) +
            Pad(self.labels_key, None),
        )

        self.pipeline = (
            self.source +
            ElasticAugment(
                # TODO check all these params
                control_point_spacing=[4, 40, 40],
                jitter_sigma=[0, 2, 2],
                rotation_interval=[0, math.pi / 2.0],
                prob_slip=0.05,  # check if needed
                prob_shift=0.05,  # check if needed
                max_misalign=10,
                subsample=8) +
            SimpleAugment(transpose_only=[1, 2]) +
            IntensityAugment(self.raw_key, 0.9, 1.1, -0.1, 0.1) +
            IntensityScaleShift(self.raw_key, 2, -1) +  # check if needed here
            # at least for debugging:
            Snapshot({
                self.raw_key: 'volumes/raw',
                self.labels_key: 'volumes/labels'
            },
                every=100,
                output_filename='batch_{iteration}.hdf')
        )

    def __len__(self):
        return self.len

    def get_patch_gunpowder(self, center, node_id):
        """
        generator function
        TODO
        Args:
            center:
            node_id:

        Returns:

        """
        offset = np.array(center) - np.array(self.patch_size) / 2
        roi = daisy.Roi(offset=offset, shape=self.patch_size)
        roi = roi.snap_to_grid(daisy.Coordinate(config.voxel_size), mode='closest')
        with build(self.pipeline) as p:

            request = BatchRequest()
            if self.raw_channel:
                request.add(self.raw_key, roi)
            if self.mask_channel:
                request.add(self.labels_key, roi)

            batch = p.request_batch(request)

            channels = []
            if self.raw_channel:
                raw_array = batch[self.raw_key].data
                channels.append(raw_array)
            if self.mask_channel:
                labels_array = batch[self.labels_key].data
                labels_array = (labels_array == node_id).astype(np.float32)
                channels.append(labels_array)

            tensor = torch.tensor(channels, dtype=torch.float)
            # Add the `channel`-dimension
            if len(channels) == 1:
                tensor = tensor.unsqueeze(0)

            yield tensor

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

        # TODO padding
        channels = []
        if self.raw_channel:
            ds = daisy.open_ds(config.groundtruth_zarr, config.groundtruth_ds)
            if not ds.roi.contains(roi):
                logger.warning(f'patch {roi} is not fully contained in {ds.roi}')
                return None
            patch = (ds[roi].to_ndarray() / 255.0).astype(np.float32)
            channels.append(patch)

        if self.mask_channel:
            ds = daisy.open_ds(config.fragments_zarr, config.fragments_ds)
            if not ds.roi.contains(roi):
                logger.warning(f'patch {roi} is not fully contained in {ds.roi}')
                return None
            patch = ds[roi].to_ndarray()
            mask = (patch == node_id).astype(np.float32)
            channels.append(mask)

        tensor = torch.tensor(channels, dtype=torch.float)

        # Add the `channel`-dimension
        if len(channels) == 1:
            tensor = tensor.unsqueeze(0)

        # try generator
        return tensor

    def __getitem__(self, index):
        """
        TODO
        Args:
            index:

        Returns:

        """
        start_getitem = now()

        # TODO this could be stored in memory
        edge_score = self.edges_attrs[config.new_edge_attr_trinary][index]

        # get the two incident nodes
        node1_id = self.edges_attrs[self.node1_field][index]
        node2_id = self.edges_attrs[self.node2_field][index]
        # weird numpy syntax
        node1_index = np.where(self.nodes_attrs[self.id_field] == node1_id)[0][0]
        node2_index = np.where(self.nodes_attrs[self.id_field] == node2_id)[0][0]

        node1_center = (
            self.nodes_attrs['center_z'][node1_index],
            self.nodes_attrs['center_y'][node1_index],
            self.nodes_attrs['center_x'][node1_index])
        node2_center = (
            self.nodes_attrs['center_z'][node2_index],
            self.nodes_attrs['center_y'][node2_index],
            self.nodes_attrs['center_x'][node2_index])

        node1_patch = self.get_patch(center=node1_center, node_id=node1_id)
        node2_patch = self.get_patch(center=node2_center, node_id=node2_id)

        if node1_patch is None or node2_patch is None:
            logger.warning(f'patch for one of the nodes is not fully contained in ROI, try again')
            # new_index = np.random.randint(0, len(self.edges_attrs[self.node1_field]))
            # Sample a new index
            new_index = torch.multinomial(
                input=self.samples_weights,
                num_samples=1,
                replacement=True).item()
            return self.__getitem__(index=new_index)

        # input0 = torch.tensor(node1_patch).float()
        # input1 = torch.tensor(node2_patch).float()
        input0 = node1_patch.float()
        input1 = node2_patch.float()
        label = torch.tensor(edge_score).float()

        # if self.transform is not None:
        #     input0 = self.transform(input0)
        #     input1 = self.transform(input1)

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return input0, input1, label
