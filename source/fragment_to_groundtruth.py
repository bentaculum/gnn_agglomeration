import daisy
import zarr
import numpy as np
import time
import logging


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
