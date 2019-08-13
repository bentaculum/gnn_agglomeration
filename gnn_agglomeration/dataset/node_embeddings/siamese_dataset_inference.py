import logging
import torch
from gunpowder import *
import numpy as np
from time import time as now

from .siamese_dataset import SiameseDataset  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)


class SiameseDatasetInference(SiameseDataset):
    """
    Each data point is one patch + node id
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
        self.len = len(self.nodes_attrs[self.id_field])

    def init_pipeline(self):
        # gunpowder init
        self.raw_key = ArrayKey('RAW')
        self.labels_key = ArrayKey('LABELS')

        self.sources = (
            ZarrSource(
                config.groundtruth_zarr,
                datasets={self.raw_key: config.raw_ds},
                array_specs={self.raw_key: ArraySpec(interpolatable=True)}) +
            Normalize(self.raw_key) +
            Pad(self.raw_key, None, value=0),
            # interpolatable?
            ZarrSource(
                config.fragments_zarr,
                datasets={self.labels_key: config.fragments_ds},
                array_specs={self.labels_key: ArraySpec(interpolatable=True)}) +
            Pad(self.labels_key, None, value=0),
        )

        self.pipeline = (
            self.sources +
            MergeProvider() +
            IntensityScaleShift(self.raw_key, 2, -1) +
            # at least for debugging:
            Snapshot({
                self.raw_key: 'volumes/raw',
                self.labels_key: 'volumes/labels'
            },
                every=100,
                output_dir='snapshots',
                output_filename=f'sample_{now()}.hdf')
            # PrintProfilingStats(every=1)
        )

    def __getitem__(self, index):
        """
        Args:

            index(int): number of edge in dataset to load

        Returns:
            a volumetric patches for the node at position `index`
            in self.nodes_attrs, plus the corresponding node id

        """
        start_getitem = now()

        node_id = self.nodes_attrs[self.id_field][index]
        node_center = (
            self.nodes_attrs['center_z'][index],
            self.nodes_attrs['center_y'][index],
            self.nodes_attrs['center_x'][index])

        patch = self.get_patch(center=node_center, node_id=node_id)

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return patch.float(), node_id
