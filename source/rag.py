# from pymongo import MongoClient
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

block_size_euclidian = (np.array(roi_shape) / 10).astype(int)
print('block size: {}'.format(block_size_euclidian))

# TODO which split dimension for anistropic data?
validation_split = 0.1
test_split = 0.1

roi_offset_train = roi_offset
roi_shape_train = (roi_shape *
                   np.array([1.0 - validation_split - test_split, 1.0, 1.0])).astype(int)

roi_offset_val = (
    roi_offset + np.array([roi_shape[0] * (1.0 - validation_split - test_split), 0, 0])).astype(int)
roi_shape_val = (
    roi_shape * np.array([validation_split, 1.0, 1.0])).astype(int)

roi_offset_test = (
    roi_offset + np.array([roi_shape[0] * (1.0 - test_split), 0, 0])).astype(int)
roi_shape_test = (roi_shape * np.array([test_split, 1.0, 1.0])).astype(int)


def create_graphs_blockwise(roi_offset, roi_shape, block_size):
    blocks_per_dim = (roi_shape / block_size).astype(int)
    overlap_pct = 0.1
    overlap_abs = int(overlap_pct *
                      np.max((np.array(roi_shape) / np.array(blocks_per_dim)).astype(int)))
    print('overlap voxels {}'.format(overlap_abs))
    # TODO check whether the overlapping region is guaranteed to contain the longest edge

    block_offsets = []
    block_shapes = []
    block_shape_default = (np.array(roi_shape) /
                           np.array(blocks_per_dim)).astype(int)
    # Create Rois for all blocks
    for i in range(blocks_per_dim[0]):
        for j in range(blocks_per_dim[1]):
            for k in range(blocks_per_dim[2]):
                block_offsets.append(
                    np.array(roi_offset) +
                    np.array([i, j, k]) * (np.array(roi_shape) / np.array(blocks_per_dim)).astype(int))
                block_shapes.append(block_shape_default)
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
    sub_blocks_per_block = [3, 3, 3]

    for i in range(int(np.prod(blocks_per_dim))):
        print('read block {} ...'.format(i))
        roi = daisy.Roi(list(block_offsets[i]), list(block_shapes[i]))
        roi_blocks.append(roi)
        # TODO enable using sub-blocks for large blocks
        # node_attrs, edge_attrs = graph_provider.read_blockwise(
        # roi=roi, block_size=daisy.Coordinate((block_shape_default / sub_blocks_per_block).astype(int)), num_workers=config['num_workers'])
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


def create_graphs_random(roi_offset, roi_shape, block_size, num_graphs):
    print('reading in {} random blocks'.format(num_graphs))
    block_offsets = []
    block_shapes = []

    # Create random Rois
    for i in range(num_graphs):
        random_offset = np.zeros(3, dtype=np.int_)
        random_offset[0] = np.random.randint(
            low=0, high=roi_shape[0] - block_size[0])
        random_offset[1] = np.random.randint(
            low=0, high=roi_shape[1] - block_size[1])
        random_offset[2] = np.random.randint(
            low=0, high=roi_shape[2] - block_size[2])
        total_offset = roi_offset + random_offset

        block_offsets.append(total_offset)
        block_shapes.append(block_size)

    node_blocks = []
    edge_blocks = []
    roi_blocks = []
    sub_blocks_per_block = [3, 3, 3]

    for i in range(len(block_offsets)):
        print('read block {} ...'.format(i))
        roi = daisy.Roi(list(block_offsets[i]), list(block_shapes[i]))
        roi_blocks.append(roi)
        # TODO enable using sub-blocks for large blocks
        # node_attrs, edge_attrs = graph_provider.read_blockwise(
        # roi=roi, block_size=daisy.Coordinate((block_shape_default / sub_blocks_per_block).astype(int)), num_workers=config['num_workers'])
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


# TODO finalize graph
create_graphs_random(roi_offset=roi_offset_train,
                     roi_shape=roi_shape_train, block_size=block_size_euclidian, num_graphs=5)
create_graphs_blockwise(roi_offset=roi_offset_val,
                        roi_shape=roi_shape_val, block_size=block_size_euclidian)
create_graphs_blockwise(roi_offset=roi_offset_test,
                        roi_shape=roi_shape_test, block_size=block_size_euclidian)
# TODO sanity check: Does the union of all blocks contain all edges and nodes for blockwise loading?
# print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))
