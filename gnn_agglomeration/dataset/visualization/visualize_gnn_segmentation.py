import daisy
import neuroglancer
import numpy as np
import sys
import zarr

from funlib.show.neuroglancer import add_layer, ScalePyramid

neuroglancer.set_server_bind_address('0.0.0.0')

input_file = sys.argv[1]

raw = [
    daisy.open_ds(input_file, 'volumes/raw/s%d' % s)
    for s in range(6)
]

relabelled = [
    daisy.open_ds(input_file, 'volumes/labels/relabelled_ids/s%d' % s)
    for s in range(6)
]
relabelled_mesh = [
    daisy.open_ds(input_file, 'volumes/labels/relabelled_ids/s%d' % s)
    for s in range(4, 5)
]


output_file = sys.argv[2]


segmentation_gnn = [
    daisy.open_ds(
        output_file,
        f'volumes/segmentation_gnn/setup28_20/s{i}') for i in range(6)]

segmentation_gnn_mesh = [
    daisy.open_ds(
        output_file,
        f'volumes/segmentation_gnn/setup28_20/s{i}') for i in range(4, 5)]

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, raw, 'raw')
    add_layer(s, relabelled, 'relabelled gt')
    add_layer(s, relabelled_mesh, 'relabelled gt mesh')
    add_layer(s, segmentation_gnn, 'segmentation')
    add_layer(s, segmentation_gnn_mesh, 'segmentation mesh')
print(viewer)
