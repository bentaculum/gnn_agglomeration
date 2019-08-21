import logging
import torch
from gunpowder import *
import numpy as np
from time import time as now
import pymongo
import bson
import pickle

from .siamese_dataset import SiameseDataset  # noqa

# dataset configs for many params
from config import config  # noqa


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)


class SiameseDatasetInference(SiameseDataset):
    """
    Each data point is one patch + node id
    """

    def __init__(
            self,
            patch_size,
            raw_channel,
            mask_channel,
            raw_mask_channel,
            num_workers=5,
            in_memory=True,
            inference_samples='all',
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
        self.inference_samples = inference_samples
        super().__init__(
            patch_size=patch_size,
            raw_channel=raw_channel,
            mask_channel=mask_channel,
            raw_mask_channel=raw_mask_channel,
            num_workers=num_workers,
            in_memory=in_memory,
            rag_block_size=rag_block_size,
            rag_from_file=rag_from_file,
            dump_rag=dump_rag
        )

        # assign dataset length
        self.len = len(self.nodes_attrs[self.id_field])

    def init_pipeline(self):
        self.raw_key = ArrayKey('RAW')
        self.labels_key = ArrayKey('LABELS')

        if self.in_memory:
            from .hdf5_like_in_memory import InMemZarrSource as ZarrSource  # noqa
        else:
            from gunpowder.nodes.zarr_source import ZarrSource  # noqa

        self.sources = (
            ZarrSource(
                config.groundtruth_zarr,
                datasets={self.raw_key: config.raw_ds_emb},
                array_specs={self.raw_key: ArraySpec(interpolatable=True)}) +
            Normalize(self.raw_key) +
            Pad(self.raw_key, None, value=0),
            ZarrSource(
                config.fragments_zarr,
                datasets={self.labels_key: config.fragments_ds_emb},
                array_specs={self.labels_key: ArraySpec(interpolatable=False)}) +
            Pad(self.labels_key, None, value=0),
        )

        self.pipeline = (
            self.sources +
            MergeProvider()
        )
        # PrintProfilingStats(every=1)

    def get_batch(self, center, node_id):
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
        roi = roi.snap_to_grid(Coordinate(config.voxel_size_emb), mode='closest')
        # logger.debug(f'ROI snapped to grid: {roi}')

        request = BatchRequest()
        if self.raw_channel or self.raw_mask_channel:
            request[self.raw_key] = ArraySpec(roi=roi)
        if self.mask_channel or self.raw_mask_channel:
            request[self.labels_key] = ArraySpec(roi=roi)

        batch = self.batch_provider.request_batch(request)

        channels = []
        if self.raw_mask_channel:
            raw_array = batch[self.raw_key].data
            labels_array = batch[self.labels_key].data
            assert raw_array.shape == labels_array.shape, \
                f'raw shape {raw_array.shape}, labels shape {labels_array.shape}'
            mask = labels_array == node_id

            raw_mask_array = raw_array * mask
            channels.append(raw_mask_array.astype(np.float32))
            if self.raw_channel:
                channels.append(raw_array)
            if self.mask_channel:
                channels.append(mask.astype(np.float32))

        else:
            if self.raw_channel:
                raw_array = batch[self.raw_key].data
                channels.append(raw_array)
            if self.mask_channel:
                labels_array = batch[self.labels_key].data
                labels_array = (labels_array == node_id).astype(np.float32)
                # sanity check: is there overlap?
                # logger.debug(f'overlap: {labels_array.sum()} voxels')
                channels.append(labels_array)

        tensor = torch.tensor(channels, dtype=torch.float)

        return tensor

    def __getitem__(self, index):
        """
        Args:

            index(int): number of node in dataset to load

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

        patch = self.get_batch(center=node_center, node_id=node_id)

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return patch.float(), torch.tensor(node_id.astype(np.int64))

    def write_embeddings_to_db(self, node_ids, embeddings, collection_name):
        start = now()
        client = pymongo.MongoClient(config.db_host)
        db = client[config.db_name]

        logger.info(
            f'''num nodes in ROI {len(self.nodes_attrs[self.id_field])},
            num embeddings {len(embeddings)}''')
        if self.inference_samples == 'all':
            assert len(self.nodes_attrs[self.id_field]) == len(embeddings)

        collection = db[collection_name]

        insertion_elems = []
        # TODO parametrize field name
        for i, e in zip(node_ids, embeddings):
            insertion_elems.append(
                {self.id_field: bson.Int64(i),
                 'embedding': bson.Binary(pickle.dumps(e))})
        collection.insert_many(insertion_elems, ordered=False)
        logger.info(
            f'write embeddings to db in {now() - start}s')
