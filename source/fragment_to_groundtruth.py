import daisy
import zarr
import numpy as np
import time
import logging
from collections import Counter
import pickle
import sys
import json
import multiprocessing as mp


# logging.basicConfig(level=logging.DEBUG)

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

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)


def overlap_in_block(block, fragments, groundtruth, output):
    # logger.debug("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    # logger.debug("Copying took {0:.3f}".format(time.time() - start))

    # get all fragment ids in this block
    frag_ids = np.unique(fragments)
    # logger.debug('num of fragment IDs: {}'.format(len(frag_ids)))

    frag_dict = dict()
    # for each of them, create a boolean mask and count the remaining elems
    for i in frag_ids:
        start = time.time()
        # TODO is numpy.ma faster?
        masked_gt = groundtruth[fragments == i]
        unique, counts = np.unique(masked_gt, return_counts=True)

        # write counter into dict frag:Counter
        frag_dict[i] = Counter(dict(zip(unique, counts)))
        # logger.debug("Count fragment {0} took {1:.3f}".format(
        # i, time.time() - start))

    output.put(frag_dict)
    logger.debug('num of dicts in output list: {}'.format(output.qsize()))

    # Successful exit
    return 1


def merge_blocks(blocks_queue):
    # put all dictionaries from mp.Queue into a list
    block_dicts = []
    while not blocks_queue.empty():
        block_dicts.append(blocks_queue.get())

    # TODO replace this with a simple check for duplicates
    keys = []
    for b in block_dicts:
        keys.extend(b.keys())
    keys = set(keys)

    totals = dict()
    for k in keys:
        totals[k] = Counter()

    for b in block_dicts:
        for k, v in b:
            totals[k] += b

    # TODO set thresholds here
    # TODO check if 0 is background
    # most common elem
    frag_to_gt = dict()
    for k, v in totals:
        frag_to_gt[k] = v.most_common(1)[0][0] if v else 0

    return frag_to_gt


output = mp.Queue()

# TODO parametrize block size
# TODO make sure blockwise
block_size = 1024
# block_size = 3000
total_roi = daisy.Roi(offset=config['roi_offset'], shape=(2048, 2048, 2048))
# total_roi = daisy.Roi(offset=config['roi_offset'], shape=config['roi_shape'])

logger.info('Start blockwise processing')
start = time.time()
daisy.run_blockwise(
    total_roi=total_roi,
    read_roi=daisy.Roi(offset=(0, 0, 0), shape=(
        block_size, block_size, block_size)),
    write_roi=daisy.Roi(offset=(0, 0, 0), shape=(
        block_size, block_size, block_size)),
    process_function=lambda block: overlap_in_block(
        block=block,
        fragments=fragments,
        groundtruth=groundtruth,
        output=output),
    fit='shrink',
    num_workers=2,
    # num_workers=config['num_workers'],
    # processes=True,
    read_write_conflict=True,
    max_retries=0)

logger.debug('num output dicts: {}'.format(len(output)))
logger.debug('num blocks: {}'.format(
    np.prod(np.ceil(np.array(config['roi_shape']) / np.array([block_size, block_size, block_size])))))

frag_to_gt = merge_blocks(output)

pickle.dump(frag_to_gt, open('frag_to_gt.pickle', 'wb'))
