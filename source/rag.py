from pymongo import MongoClient
import sys
import numpy as np
import pandas as pd
import time
import json

import daisy

# client_address = sys.argv[1]

# client = MongoClient(client_address)

# db = client.hemi_mtlsd_400k_roi_1
# db_nodes = db.nodes
# db_edges = db.edges_hist_quant_50

# print('Loading RAG nodes from DB ...')
# cursor = db_nodes.find()
# nodes = pd.DataFrame(list(cursor))

# print('Loading RAG edges from DB ...')
# cursor = db_edges.find()
# edges = pd.DataFrame(list(cursor))

# sys.exit(0)


config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)


db_host = config['db_host']
db_name = config['db_name']
nodes_collection = 'nodes'
edges_collection = config['edges_collection']


print("Reading graph from DB ", db_name, edges_collection)
start = time.time()

graph_provider = daisy.persistence.MongoDbGraphProvider(
    db_name,
    db_host,
    mode='r',
    nodes_collection=nodes_collection,
    edges_collection=edges_collection,
    endpoint_names=['u', 'v'],
    position_attribute=[
        'center_z',
        'center_y',
        'center_x'])


roi_offset = config['roi_offset']
roi_shape = config['roi_shape']

dim = 3
if dim != 3:
    raise NotImplementedError('Works only in 3D')

blocks_per_dim = [3, 3, 3]
overlap_pct = 0.1
# TODO check whether the overlap contains the longest edge for sure

block_offsets = []
block_shapes = []
# Create Rois for all blocks
for i in range(blocks_per_dim[0]):
    for j in range(blocks_per_dim[1]):
        for k in range(blocks_per_dim[2]):
            # TODO overlapping
            block_offsets.append(
                np.array(roi_offset) +
                np.array([i, j, k]) * (np.array(roi_shape) / np.array(blocks_per_dim)).astype(int))
            block_shapes.append((
                np.array(roi_shape) / np.array(blocks_per_dim)).astype(int))

node_blocks = []
edge_blocks = []
for i in range(int(np.prod(blocks_per_dim))):
    roi = daisy.Roi(list(block_offsets[i]), list(block_shapes[i]))
    node_attrs = graph_provider.read_nodes(roi=roi)
    node_blocks.append(node_attrs)
    edge_attrs = graph_provider.read_edges(roi=roi)
    edge_blocks.append(edge_attrs)

sys.exit(0)


block_size = 5900
num_workers = 4
node_attrs, edge_attrs = graph_provider.read_blockwise(
    roi,
    block_size=daisy.Coordinate((block_size, block_size, block_size)),
    num_workers=num_workers)

print("Read graph in %.3fs" % (time.time() - start))

if 'id' not in node_attrs:
    print('No nodes found in roi %s' % roi)
    sys.exit(0)

print('id dtype: ', node_attrs['id'].dtype)
print('edge u  dtype: ', edge_attrs['u'].dtype)
print('edge v  dtype: ', edge_attrs['v'].dtype)

nodes = node_attrs['id']
edges = np.stack([edge_attrs['u'].astype(np.uint64),
                  edge_attrs['v'].astype(np.uint64)], axis=1)
scores = edge_attrs['merge_score'].astype(np.float32)

print('Nodes dtype: ', nodes.dtype)
print('edges dtype: ', edges.dtype)
print('scores dtype: ', scores.dtype)

print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))
