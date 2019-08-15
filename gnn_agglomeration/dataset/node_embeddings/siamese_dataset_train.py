import logging
import torch
from gunpowder import *
import numpy as np
from time import time as now
import math

from .siamese_dataset import SiameseDataset  # noqa
from .hdf5_like_in_memory import InMemZarrSource  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)


class SiameseDatasetTrain(SiameseDataset):
    """
    Each data point is actually a mini-batch of volume pairs
    """

    def __init__(self, patch_size, raw_channel, mask_channel, num_workers=5):
        """
        connect to db, load and weed out edges, define gunpowder pipeline
        Args:
            patch_size:
            raw_channel (bool): if set true, load patch from raw volumetric data
            mask_channel (bool): if set true, load patch from fragments volumetric data
            num_workers (int): number of workers available, e.g. for loading RAG
        """

        super().__init__(
            patch_size=patch_size,
            raw_channel=raw_channel,
            mask_channel=mask_channel,
            num_workers=num_workers
        )

        # assign dataset length
        self.len = len(self.edges_attrs[config.new_edge_attr_trinary])

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

    def init_pipeline(self):
        # gunpowder init
        self.raw_key = ArrayKey('RAW')
        self.labels_key = ArrayKey('LABELS')

        self.sources = (
            InMemZarrSource(
                config.groundtruth_zarr,
                datasets={self.raw_key: config.raw_ds},
                array_specs={self.raw_key: ArraySpec(interpolatable=True)}) +
            Normalize(self.raw_key) +
            Pad(self.raw_key, None, value=0),
            InMemZarrSource(
                config.fragments_zarr,
                datasets={self.labels_key: config.fragments_ds},
                array_specs={self.labels_key: ArraySpec(interpolatable=True)}) +
            Pad(self.labels_key, None, value=0),
        )

        self.pipeline = (
            self.sources +
            MergeProvider() +
            ElasticAugment(
                # copied from /groups/funke/funkelab/sheridana/lsd_experiments/hemi/02_train/setup01/train.py
                control_point_spacing=[40, 40, 40],
                # copied from /groups/funke/funkelab/sheridana/lsd_experiments/hemi/02_train/setup01/train.py
                jitter_sigma=[2, 2, 2],
                # indep. rotation of two cropouts does not help
                rotation_interval=[0, 0],
                prob_slip=0.0,
                prob_shift=0.0,
                max_misalign=0,
                # TODO adjust subsample value for speed
                subsample=8)  # +
            # TODO do not use transpose, currently buggy
            # SimpleAugment(transpose_only=[]) +
            # PrintProfilingStats(every=1)
        )

        if self.raw_channel:
            self.pipeline + \
                IntensityAugment(self.raw_key, 0.9, 1.1, - 0.1, 0.1) + \
                IntensityScaleShift(self.raw_key, 2, -1)

        # at least for debugging:
        # Snapshot({
        # self.raw_key: 'volumes/raw',
        # self.labels_key: 'volumes/labels'
        # },
        # every=100,
        # output_dir='snapshots',
        # output_filename=f'sample_{now()}.hdf')
        # )

    def __getitem__(self, index):
        """
        Args:

            index(int): number of edge in dataset to load

        Returns:
            a pair of volumetric patches for the two incident nodes,
            plus the corresponding label

        """
        start_getitem = now()

        edge_score = self.edges_attrs[config.new_edge_attr_trinary][index]

        # get the two incident nodes
        node1_id = self.edges_attrs[self.node1_field][index]
        node2_id = self.edges_attrs[self.node2_field][index]
        # weird numpy syntax
        node1_index = np.where(
            self.nodes_attrs[self.id_field] == node1_id)[0][0]
        node2_index = np.where(
            self.nodes_attrs[self.id_field] == node2_id)[0][0]

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
            logger.warning(
                f'patch for one of the nodes is not fully contained in ROI, try again')
            # Sample a new index, using the sample weights again
            new_index = torch.multinomial(
                input=self.samples_weights,
                num_samples=1,
                replacement=True).item()
            return self.__getitem__(index=new_index)

        input0 = node1_patch.float()
        input1 = node2_patch.float()

        if edge_score == 0:
            label = torch.tensor(1.0)
        elif edge_score == 1:
            label = torch.tensor(-1.0)
        else:
            raise ValueError(
                f'Value {edge_score} cannot be transformed into a valid label')

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return input0, input1, label
