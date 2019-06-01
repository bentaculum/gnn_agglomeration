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
# TODO test train split in 0th dimension of the cube

dim = 3
if dim != 3:
    raise NotImplementedError('Works only in 3D')

blocks_per_dim = [3, 3, 3]
overlap_pct = 0.1
overlap_abs = int(overlap_pct *
                  np.max((np.array(roi_shape) / np.array(blocks_per_dim)).astype(int)))
print('overlap voxels {}'.format(overlap_abs))
# TODO check whether the overlapping region is guaranteed to contain the longest edge

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
            # overlapping
            if i < blocks_per_dim[0] - 1:
                block_shapes[-1][0] += overlap_abs
            if j < blocks_per_dim[1] - 1:
                block_shapes[-1][1] += overlap_abs
            if k < blocks_per_dim[2] - 1:
                block_shapes[-1][2] += overlap_abs

node_blocks = []
edge_blocks = []
roi_blocks = []
for i in range(int(np.prod(blocks_per_dim))):
    roi = daisy.Roi(list(block_offsets[i]), list(block_shapes[i]))
    roi_blocks.append(roi)
    node_attrs = graph_provider.read_nodes(roi=roi)
    node_blocks.append(node_attrs)
    edge_attrs = graph_provider.read_edges(roi=roi)
    edge_blocks.append(edge_attrs)

print("Read graph in %.3fs" % (time.time() - start))

for i, (n, e) in enumerate(zip(node_blocks, edge_blocks)):
    print('prepare block {} ...'.format(i))
    if len(n) == 0:
        print('No nodes found in roi %s' % roi_blocks[i])
        sys.exit(0)
    if len(e) == 0:
        print('No edges found in roi %s' % roi_blocks[i])
        sys.exit(0)

    # TODO remap local indices per block to [0, num_nodes]
    # to transform the edges, you need to do lookup of the node ids. Use a dict. Store in pyg graph as torch tensor

    # node_attrs, edge_attrs = graph_provider.read_blockwise(
    # roi,
    # block_size=daisy.Coordinate((block_size, block_size, block_size)),
    # num_workers=num_workers)

    df_nodes = pd.DataFrame(n)
    # columns in desired order
    df_nodes = df_nodes[['id', 'center_z',
                         'center_y', 'center_x']].astype(np.uint64)

    df_edges = pd.DataFrame(e)
    # columns in desired order
    df_edges = df_edges[['u', 'v', 'merge_score']]
    df_edges['u'] = df_edges['u'].astype(np.uint64)
    df_edges['v'] = df_edges['v'].astype(np.uint64)
    df_edges['merge_score'] = df_edges['merge_score'].astype(np.float32)

    nodes_remap = dict(zip(df_nodes['id'], range(len(df_nodes))))
    # TODO save this as torch tensor
    node_ids = df_nodes['id'].values
    edges_remapped = []
    df_edges['u'] = df_edges['u'].map(nodes_remap)
    df_edges['v'] = df_edges['v'].map(nodes_remap)

    # TODO convert to torch tensors
    edge_index = df_edges[['u', 'v']].values
    edge_attr = df_edges['merge_score'].values
    pos = df_nodes[['center_z', 'center_y', 'center_x']].values

# TODO sanity check: Does the union of all blocks contain all edges and nodes?
# print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))
