import daisy
import zarr
import numpy as np
import time
import logging
from collections import Counter


logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy.datasets').setLevel(logging.DEBUG)

# TODO move to config
fragments_zarr = '/nrs/funke/sheridana/hemi/setup02/400000/cropout_1.zarr'
fragments_ds = 'volumes/fragments'

groundtruth_zarr = '/groups/funke/funkelab/sheridana/lsd_experiments/hemi/01_data/hemi_testing_roi.zarr'
groundtruth_ds = 'volumes/labels/relabelled_eroded_ids/s0'

fragments = daisy.open_ds(fragments_zarr, fragments_ds)
groundtruth = daisy.open_ds(groundtruth_zarr, groundtruth_ds)

# Try brute force version
# 25G of memory, takes 50 sec to load

# logging.info("Loading fragments to memory...")
# start = time.time()
# np_fragments = fragments.to_ndarray()
# logging.info("%.3fs" % (time.time() - start))

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)


def overlap_in_block(block, fragments, groundtruth, output):
    logging.info("Copying fragments to memory...")
    start = time.time()
    fragments = fragments.to_ndarray(block.write_roi)
    logging.info("%.3fs" % (time.time() - start))

    # get all fragment ids in this block
    frag_ids = np.unique(fragments)
    frag_dict = dict()
    # for each of them, create a boolean mask and count the remaining elems
    for i in frag_ids:
        # TODO is numpy.ma faster?
        masked_gt = groundtruth[fragments == i]
        unique, counts = numpy.unique(masked_gt, return_counts=True)

        # write counter into dict frag:Counter
        frag_dict[i] = Counter(dict(zip(unique, counts)))

    # TODO somehow return
    output.append(frag_dict)


def reduce(block_dicts):
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


# TODO parametrize block size
daisy.run_blockwise(
    total_roi=daisy.Roi(
        offset=config['roi_offset'], shape=config['roi_shape']),
    read_roi=daisy.Roi(offset=(0, 0, 0), (500, 500, 500)),
    write_roi=daisy.Roi(offset=(0, 0, 0), (500, 500, 500)),
    lambda b: overlap_in_block(
        b,
        fragments_file,
        segmentation,
        fragments,
        lut),
    fit='shrink',
    num_workers=config['num_workers'],
    processes=True,
    # TODO check if that solves the race condition
    read_write_conflict=True)
