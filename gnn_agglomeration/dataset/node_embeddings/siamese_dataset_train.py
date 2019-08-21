import logging
import torch
from gunpowder import *
import numpy as np
from time import time as now
import math
import os
import os.path as osp
import json
import h5py
import datetime
import pytz

from .siamese_dataset import SiameseDataset  # noqa
from .merge_fragments import MergeFragments  # noqa

# dataset configs for many params
from config import config  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)


class SiameseDatasetTrain(SiameseDataset):
    """
    Each data point is actually a mini-batch of volume pairs
    """

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
            dump_rag=None,
            snapshots=False):
        """
        connect to db, load and weed out edges, define gunpowder pipeline
        Args:
            patch_size:
            raw_channel (bool): if set true, load patch from raw volumetric data
            mask_channel (bool): if set true, load patch from fragments volumetric data
            num_workers (int): number of workers available, e.g. for loading RAG
        """

        self.snapshots = snapshots
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

        timestamp = datetime.datetime.now(
            pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')
        self.snapshot_dir = os.path.join('snapshots', str(timestamp))
        if not osp.isdir(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

        # 5 control points (4 intervals) per axis
        adaptive_control_point_spacing = np.array(
            self.patch_size)/(4 * np.array(config.voxel_size_emb, dtype=np.int_))
        logger.info(
            f'ElasticAugment control point spacing: {adaptive_control_point_spacing}')
        adaptive_jitter_sigma = adaptive_control_point_spacing / 16
        logger.info(f'ElasticAugment jitter sigma: {adaptive_jitter_sigma}')

        self.pipeline = (
            self.sources +
            MergeProvider() +
            MergeFragments() +
            ElasticAugment(
                control_point_spacing=adaptive_control_point_spacing,
                jitter_sigma=adaptive_jitter_sigma,
                # indep. rotation of two cropouts does not help
                rotation_interval=[0, math.pi/2],
                prob_slip=0.0,
                prob_shift=0.0,
                max_misalign=0,
                # TODO adjust subsample value for speed
                subsample=8) +
            SimpleAugment()
            # PrintProfilingStats(every=1)
        )

        if self.raw_channel or self.raw_mask_channel:
            self.pipeline + \
                IntensityAugment(self.raw_key, 0.9, 1.1, - 0.1, 0.1)

    def get_batch(self, center, node_id):
        """
        TODO
        Args:
            center:
            node_id:

        Returns:

        """
        center_u, center_v = center
        roi = Roi(offset=(0, 0, 0), shape=self.patch_size)
        roi = roi.snap_to_grid(Coordinate(config.voxel_size_emb), mode='closest')

        request = BatchRequest()
        request.thaw()
        request.center_u = Coordinate(center_u)
        request.center_v = Coordinate(center_v)
        request.freeze()

        if self.raw_channel or self.raw_mask_channel:
            request[self.raw_key] = ArraySpec(
                roi=roi,
                voxel_size=Coordinate(config.voxel_size_emb))
        if self.mask_channel or self.raw_mask_channel:
            request[self.labels_key] = ArraySpec(
                roi=roi,
                voxel_size=Coordinate(config.voxel_size_emb))

        batch = self.batch_provider.request_batch(request)

        batch_torch = []
        mask_sizes = []
        fragment_arrays_for_snapshot = []
        for i in range(0, 2):
            # u=0, v=1
            channels = []
            if self.raw_mask_channel:
                if self.snapshots:
                    raise NotImplementedError(
                        'Snapshots not implemented for raw_mask_channel')

                raw_array = batch[self.raw_key].data[i]
                labels_array = batch[self.labels_key].data[i]
                assert raw_array.shape == labels_array.shape, \
                    f'raw shape {raw_array.shape}, labels shape {labels_array.shape}'
                mask = labels_array == node_id[i]

                raw_mask_array = raw_array * mask

                channels.append(raw_mask_array.astype(np.float32))
                if self.raw_channel:
                    channels.append(raw_array)
                if self.mask_channel:
                    channels.append(mask.astype(np.float32))

            else:
                if self.raw_channel:
                    raw_array = batch[self.raw_key].data[i]
                    channels.append(raw_array)
                if self.mask_channel:
                    labels_array = batch[self.labels_key].data[i]
                    mask = (
                        labels_array == node_id[i]).astype(np.float32)

                    if self.snapshots:
                        fragment_arrays_for_snapshot.append(labels_array)
                        # sanity check: fragment size, according to the mask
                        overlap = mask.sum()
                        mask_sizes.append(overlap)

                    channels.append(mask)

            tensor = torch.tensor(channels, dtype=torch.float)
            batch_torch.append(tensor)

        return batch_torch, batch.id, mask_sizes, fragment_arrays_for_snapshot

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

        (node1_patch, node2_patch), batch_id, mask_sizes, fragment_arrays = self.get_batch(
            center=(node1_center, node2_center),
            node_id=(node1_id, node2_id)
        )

        input0 = node1_patch.float()
        input1 = node2_patch.float()

        if edge_score == 0:
            label = torch.tensor(1.0)
        elif edge_score == 1:
            label = torch.tensor(-1.0)
        else:
            raise ValueError(
                f'Value {edge_score} cannot be transformed into a valid label')

        if self.snapshots:
            self.snapshot_batch(
                batch_id=batch_id,
                label=label,
                mask_sizes=mask_sizes,
                inputs=(input0, input1),
                centers=(node1_center, node2_center),
                fragment_arrays=fragment_arrays,
                node_ids=(node1_id, node2_id)
            )

        logger.debug(f'__getitem__ in {now() - start_getitem} s')
        return input0, input1, label

    def snapshot_batch(self, batch_id, label, mask_sizes, inputs, centers, fragment_arrays, node_ids):
        """
        adapted from https://github.com/funkey/gunpowder/blob/53a7961c273ee70c0b8e5c95835609cb04decba3/gunpowder/nodes/snapshot.py
        TODO clean this up 
        """
        # meta data
        with open(os.path.join(self.snapshot_dir, f'{batch_id}.json'), 'w') as f:
            json.dump({'label': str(label.item()),
                       'fragment_pixels': str(mask_sizes),
                       'node_ids': str(node_ids)},
                      f)

        input0, input1 = inputs
        input0, input1 = input0.cpu().numpy(), input1.cpu().numpy()

        arrays = [
            [input0[0], input0[1], fragment_arrays[0]],
            [input1[0], input1[1], fragment_arrays[1]]
        ]
        ds_names = ['raw', 'mask', 'fragments']
        for i, arr in enumerate(arrays):
            with h5py.File(osp.join(self.snapshot_dir, f'{batch_id}_{i}.hdf'), 'w') as f:
                for name, block in zip(ds_names, arr):
                    dataset = f.create_dataset(name=name, data=block)

                    # calculate the offset, which the gp pipeline does not have
                    offset = np.array(centers[i]) - np.array(self.patch_size)/2
                    roi = Roi(offset=offset, shape=self.patch_size)
                    roi = roi.snap_to_grid(Coordinate(
                        config.voxel_size_emb), mode='closest')

                    dataset.attrs['offset'] = roi.get_offset()
                    dataset.attrs['resolution'] = Coordinate(config.voxel_size_emb)
                    dataset.attrs['value_range'] = (
                        np.asscalar(block.min()),
                        np.asscalar(block.max())
                    )
