import daisy
import zarr
import numpy as np
import time
import logging
from collections import Counter
import pickle
import sys
import json
import shutil
import os


logging.basicConfig(level=logging.INFO)

# TODO adjust logging levels
# logging.getLogger('daisy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO move to config
fragments_zarr = '/nrs/funke/sheridana/hemi/setup02/400000/cropout_1.zarr'
fragments_ds = 'volumes/fragments'

groundtruth_zarr = '/groups/funke/funkelab/sheridana/lsd_experiments/hemi/01_data/hemi_testing_roi.zarr'
groundtruth_ds = 'volumes/labels/relabelled_eroded_ids/s0'

fragments = daisy.open_ds(fragments_zarr, fragments_ds)
groundtruth = daisy.open_ds(groundtruth_zarr, groundtruth_ds)

# Parametrize
background_id = 0
threshold_overlap = 0.5

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)


def overlap_in_block(block, fragments, groundtruth, tmp_path):
    # logger.debug("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.read_roi)
    groundtruth = groundtruth.to_ndarray(block.read_roi)
    # logger.debug("Copying took {0:.3f}".format(time.time() - start))

    # get all fragment ids in this block
    frag_ids = np.unique(fragments)
    logger.debug('num of fragment IDs: {}'.format(len(frag_ids)))

    frag_dict = dict()
    # for each of them, create a boolean mask and count the remaining elems
    for i in frag_ids:
        # start = time.time()
        # TODO is numpy.ma faster?
        masked_gt = groundtruth[fragments == i]
        unique, counts = np.unique(masked_gt, return_counts=True)
        # logger.debug('counts {}'.format(counts))

        # write counter into dict frag:Counter
        counter = Counter(dict(zip(unique, counts)))
        # logger.debug("Count fragment {0} took {1:.3f}".format(
        # i, time.time() - start))

        max_count = counter.most_common(1)[0][1]
        all_counts = sum(counter.values())
        if max_count/all_counts > threshold_overlap:
            # most common elem
            frag_dict[i] = int(counter.most_common(1)[0][0])
        else:
            frag_dict[i] = int(background_id)

        # logger.debug('most common gt id: {}'.format(frag_dict[i]))

    logger.debug(
        'write Counter dict for block {} to file'.format(block.block_id))
    pickle.dump(frag_dict, open(os.path.join(
        tmp_path, '{}.pickle'.format(block.block_id)), 'wb'))


def overlap_reduce(tmp_path):
    block_dicts = []
    for f in os.listdir(tmp_path):
        if f.endswith(".pickle"):
            block_dicts.append(pickle.load(
                open(os.path.join(tmp_path, f), 'rb')))
    logger.info('Found {} block results in {}'.format(
        len(block_dicts), tmp_path))

    # keys = []
    # for b in block_dicts:
    # keys.extend(b.keys())
    # keys = set(keys)

    merged_dicts = dict()
    for b in block_dicts:
        for k in b.keys():
            if k in merged_dicts:
                logger.warning(
                    'fragment id {} already exists in previous block'.format(k))

        merged_dicts.update(b)

    return merged_dicts


# TODO parametrize
output_path = '../temp/overlap_counts'
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)


# TODO parametrize block size
block_size = config['block_size']
total_roi = daisy.Roi(offset=config['roi_offset'], shape=config['roi_shape'])

logger.info('Start blockwise processing')
start = time.time()
daisy.run_blockwise(
    total_roi=total_roi,
    read_roi=daisy.Roi(offset=(0, 0, 0), shape=block_size),
    write_roi=daisy.Roi(offset=(0, 0, 0), shape=block_size),
    process_function=lambda block: overlap_in_block(
        block=block,
        fragments=fragments,
        groundtruth=groundtruth,
        tmp_path=output_path),
    fit='shrink',
    num_workers=config['num_workers'],
    read_write_conflict=False,
    max_retries=0)

# TODO parametrize
logger.debug('num blocks: {}'.format(
    np.prod(np.ceil(np.array(config['roi_shape']) / np.array(block_size)))))

frag_to_gt = overlap_reduce(output_path)
pickle.dump(frag_to_gt, open('frag_to_gt.pickle', 'wb'))
