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
    for s in range(9)
]

relabelled = [
    daisy.open_ds(input_file, 'volumes/labels/relabelled_ids/s%d' % s)
    for s in range(9)
]

output_file = sys.argv[2]

frag_embeddings = [
    daisy.open_ds(output_file, f'volumes/embeddings/siamese_uint8/s{i}') for i in range(9)
]

gnn_embeddings = [
    daisy.open_ds(output_file, f'volumes/embeddings/gnn_setup26/s{i}') for i in range(9)
]

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(
        s,
        frag_embeddings,
        'fragments embeddings',
        shader='rgb'
    )
    add_layer(
        s,
        gnn_embeddings,
        'gnn embeddings',
        shader='rgb'
    )

    add_layer(s, raw, 'raw')
    add_layer(s, relabelled, 'relabelled gt')

print(viewer)
