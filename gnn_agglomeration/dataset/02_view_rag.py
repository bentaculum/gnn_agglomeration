from funlib.show.neuroglancer import add_layer, ScalePyramid
import argparse
import daisy
import glob
import neuroglancer
import numpy as np
import os
import configargparse
import sys
import time
import logging
from matplotlib import colors
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = configargparse.ArgumentParser()
parser.add_argument(
    '--file',
    '-f',
    type=str,
    action='append',
    required=True,
    help="The path to the container to show")
parser.add_argument(
    '--datasets',
    '-d',
    type=str,
    nargs='+',
    action='append',
    required=True,
    help="The datasets in the container to show")


args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining_argv]
from config import config  # noqa


neuroglancer.set_server_bind_address('0.0.0.0')
viewer = neuroglancer.Viewer()

for f, datasets in zip(args.file, args.datasets):

    arrays = []
    for ds in datasets:
        try:

            print("Adding %s, %s" % (f, ds))
            a = daisy.open_ds(f, ds)

        except:

            print("Didn't work, checking if this is multi-res...")

            scales = glob.glob(os.path.join(f, ds, 's*'))
            print("Found scales %s" % ([
                os.path.relpath(s, f)
                for s in scales
            ],))
            a = [
                daisy.open_ds(f, os.path.relpath(scale_ds, f))
                for scale_ds in scales
            ]
        arrays.append(a)

    with viewer.txn() as s:
        for array, dataset in zip(arrays, datasets):
            add_layer(s, array, dataset)

start = time.time()
graph_provider = daisy.persistence.MongoDbGraphProvider(
    config.db_name,
    config.db_host,
    mode='r',
    nodes_collection=config.nodes_collection,
    edges_collection=config.edges_collection,
    endpoint_names=['u', 'v'],
    position_attribute=[
        'center_z',
        'center_y',
        'center_x']
)

roi = daisy.Roi(list(config.roi_offset), list(config.roi_shape))
nodes_attrs, edges_attrs = graph_provider.read_blockwise(
    roi=roi,
    block_size=daisy.Coordinate((10000, 10000, 10000)),
    num_workers=5
)
logger.info(f'read graph blockwise in {time.time() - start}s')

start = time.time()
nodes = {node_id: (z, y, x) for z, y, x, node_id in zip(
    nodes_attrs["center_z"],
    nodes_attrs["center_y"],
    nodes_attrs["center_x"],
    nodes_attrs['id']
)
}

edges = {(u, v): score for u, v, score in
         zip(edges_attrs["u"], edges_attrs["v"], edges_attrs[config.new_edge_attr_trinary])}
logger.info(f'write nodes and edges to dicts in {time.time() - start}s')

start = time.time()
lines = {}
for i, ((u, v), score) in enumerate(edges.items()):
    try:
        l = neuroglancer.LineAnnotation(
            point_a=nodes[u],
            point_b=nodes[v],
            id=i
        )
        lines.setdefault(score, []).append(l)
    except KeyError:
        pass
logger.info(f'create LineAnnotations in {time.time() - start}s')

start = time.time()
colors_list = list(colors.CSS4_COLORS.values())
with viewer.txn() as s:
    for k, v in lines.items():
        # TODO adapt parameters
        s.layers[str(k)] = neuroglancer.AnnotationLayer(
            voxel_size=(1, 1, 1),
            filter_by_segmentation=False,
            annotation_color=random.choice(colors_list),
            annotations=v)
logger.info(f'add annotation layers in {time.time() - start}s')

url = str(viewer)
print(url)

print("Press ENTER to quit")
input()
