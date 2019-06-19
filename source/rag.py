# from pymongo import MongoClient
import sys
import numpy as np
import pandas as pd
import time
import json
import os
import logging

import daisy

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('daisy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


logger.info("Reading graph from DB {}, collection {}".format(db_name, edges_collection))
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
# roi_shape = config['roi_shape']
roi_shape = [11800, 11800, 1180]
# TODO test train split in 0th dimension of the cube

dim = 3
if dim != 3:
    raise NotImplementedError('Works only in 3D')

# block_size_euclidian = (np.array(roi_shape) / 10).astype(int)
block_size_euclidian = config['pyg_block_size']
# print('block size: {}'.format(block_size_euclidian))

# TODO which split dimension for anisotropic data?
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


def parse_rag_excerpt(node_attrs, edge_attrs):
    df_nodes = pd.DataFrame(node_attrs)
    # columns in desired order
    df_nodes = df_nodes[['id', 'center_z',
                         'center_y', 'center_x']].astype(np.uint64)

    df_edges = pd.DataFrame(edge_attrs)
    # columns in desired order
    # TODO account for directed edges
    df_edges = df_edges[['u', 'v', 'merge_score', 'merge_ground_truth', 'merge_labeled']]
    df_edges['merge_score'] = df_edges['merge_score'].astype(np.float32)
    df_edges['merge_ground_truth'] = df_edges['merge_ground_truth'].astype(np.int_)
    df_edges['merge_labeled'] = df_edges['merge_ground_truth'].astype(np.int_)

    nodes_remap = dict(zip(df_nodes['id'], range(len(df_nodes))))
    node_ids = df_nodes['id'].values

    df_edges['u'] = df_edges['u'].map(nodes_remap)
    df_edges['v'] = df_edges['v'].map(nodes_remap)

    # Drop edges for which one of the incident nodes is not in the extracted node set
    df_edges = df_edges[np.isfinite(df_edges['u']) & np.isfinite(df_edges['v'])]

    df_edges['u'] = df_edges['u'].astype(np.uint64)
    df_edges['v'] = df_edges['v'].astype(np.uint64)

    edge_index = df_edges[['u', 'v']].values
    edge_attr = df_edges['merge_score'].values
    pos = df_nodes[['center_z', 'center_y', 'center_x']].values
    mask = df_edges['merge_labeled'].values
    y = df_edges['merge_ground_truth'].values

    return edge_index, edge_attr, pos, node_ids, mask, y


def mask_target_edges(edge_index_padded, node_ids_padded, inner_roi, mask):
    # parse inner block
    inner_nodes = graph_provider.read_nodes(roi=inner_roi)
    inner_edges = graph_provider.read_edges(roi=inner_roi, nodes=inner_nodes)

    inner_edge_index, _, _, inner_node_ids = parse_rag_excerpt(inner_nodes, inner_edges)

    # remap inner and outer node ids to original ids
    inner_orig_edge_index = list(inner_node_ids[inner_edge_index.flatten()].reshape((-1, 2)))
    outer_orig_edge_index = list(node_ids_padded[edge_index_padded.flatten()].reshape((-1, 2)))

    # convert to sets, make sure that directedness is not a problem
    inner_list = [tuple([min(i), max(i)]) for i in inner_orig_edge_index]
    outer_list = [tuple([min(i), max(i)]) for i in outer_orig_edge_index]

    for i, edge in enumerate(outer_list):
        if edge not in inner_list:
            mask[i] = 0

    return mask


def parse_rois(block_offsets, block_shapes, padded_offsets=None, padded_shapes=None):
    sub_blocks_per_block = [3, 3, 3]

    edge_index_list = []
    edge_attr_list = []
    pos_list = []
    node_ids_list = []
    mask_list = []
    y_list = []

    for i in range(len(block_offsets)):
        logger.info('read block {} ...'.format(i))
        # print('block_offset {} block_shape {}'.format(
        # block_offsets[i], block_shapes[i]))

        # Load the padded block if given
        if padded_offsets and padded_shapes:
            roi = daisy.Roi(list(padded_offsets[i]), list(padded_shapes[i]))
        else:
            roi = daisy.Roi(list(block_offsets[i]), list(block_shapes[i]))

        # node_attrs, edge_attrs = graph_provider.read_blockwise(
        # roi=roi, block_size=daisy.Coordinate((block_shape_default / sub_blocks_per_block).astype(int)), num_workers=config['num_workers'])
        node_attrs = graph_provider.read_nodes(roi=roi)
        edge_attrs = graph_provider.read_edges(roi=roi, nodes=node_attrs)

        # print('prepare block {} ...'.format(i))
        if len(node_attrs) == 0:
            raise ValueError('No nodes found in roi %s' % roi)
        if len(edge_attrs) == 0:
            raise ValueError('No edges found in roi %s' % roi)

        edge_index, edge_attr, pos, node_ids, mask, y = parse_rag_excerpt(node_attrs, edge_attrs)

        if padded_offsets and padded_shapes:
            mask = mask_target_edges(
                edge_index_padded=edge_index,
                node_ids_padded=node_ids,
                inner_roi=daisy.Roi(list(block_offsets[i]), list(block_shapes[i])),
                mask=mask
            )
        mask_list.append(mask)

        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        pos_list.append(pos)
        node_ids_list.append(node_ids)
        y_list.append(y)

    logger.info("Parse set of ROIs in %.3fs" % (time.time() - start))
    return edge_index_list, edge_attr_list, pos_list, node_ids_list, mask_list, y_list


def pad_total_roi(roi_offset, roi_shape, padding):
    # pad the entire volume, padded area not part of total roi any more
    roi_offset_padded = np.array(roi_offset) + np.array(padding)
    roi_shape_padded = np.array(roi_shape) - 2 * np.array(padding)
    return roi_offset_padded, roi_shape_padded


def pad_block(offset, shape, padding):
    # enlarge the block with padding in all dimensions
    offset_padded = np.array(offset) - np.array(padding)
    shape_padded = np.array(shape) + 2 * np.array(padding)
    return offset_padded, shape_padded


def create_graphs_blockwise_padded(roi_offset, roi_shape, block_size, padding):
    logger.debug('total roi_offset: {}, total roi_shape: {}'.format(roi_offset, roi_shape))

    roi_offset, roi_shape = pad_total_roi(roi_offset, roi_shape, padding)
    logger.debug('padded roi_offset: {}, padded roi_shape: {}'.format(roi_offset, roi_shape))
    logger.info('block size: {}'.format(block_size))
    logger.debug('padding: {}'.format(padding))

    blocks_per_dim = (np.array(roi_shape) / np.array(block_size)).astype(int)
    logger.info('blocks per dim: {}'.format(blocks_per_dim))

    block_offsets = []
    block_shapes = []
    block_offsets_padded = []
    block_shapes_padded = []
    block_shape_default = np.array(block_size, dtype=np.int_)
    # Create Rois for all blocks
    for i in range(blocks_per_dim[0]):
        for j in range(blocks_per_dim[1]):
            for k in range(blocks_per_dim[2]):
                block_offset_new = np.array(roi_offset) + np.array([i, j, k]) * np.array(block_size, dtype=np.int_)
                block_shape_new = block_shape_default.copy()
                # padding
                bo_padded, bs_padded = pad_block(offset=block_offset_new, shape=block_shape_new, padding=padding)

                block_offsets.append(block_offset_new)
                block_shapes.append(block_shape_new)
                block_offsets_padded.append(bo_padded)
                block_shapes_padded.append(bs_padded)

    return parse_rois(
        block_offsets=block_offsets,
        block_shapes=block_shapes,
        padded_offsets=block_offsets_padded,
        padded_shapes=block_shapes_padded)


def create_graphs_blockwise(roi_offset, roi_shape, block_size):
    print('block size: {}'.format(block_size))
    overlap_pct = 0.1
    overlap_abs = int(overlap_pct *
                      np.max((np.array(block_size)).astype(int)))
    print('overlap voxels {}'.format(overlap_abs))

    blocks_per_dim = (np.array(roi_shape) / np.array(block_size)).astype(int)
    print('blocks per dim: {}'.format(blocks_per_dim))

    block_offsets = []
    block_shapes = []
    block_shape_default = np.array(block_size, dtype=np.int_)
    # Create Rois for all blocks
    for i in range(blocks_per_dim[0]):
        for j in range(blocks_per_dim[1]):
            for k in range(blocks_per_dim[2]):
                block_offsets.append(
                    np.array(roi_offset) +
                    np.array([i, j, k]) * np.array(block_size, dtype=np.int_))
                block_shape_new = block_shape_default.copy()
                # overlapping
                if i < blocks_per_dim[0] - 1:
                    block_shape_new[0] += overlap_abs
                if j < blocks_per_dim[1] - 1:
                    block_shape_new[1] += overlap_abs
                if k < blocks_per_dim[2] - 1:
                    block_shape_new[2] += overlap_abs
                block_shapes.append(block_shape_new)

    return parse_rois(block_offsets=block_offsets, block_shapes=block_shapes)


def create_graphs_random_padded(roi_offset, roi_shape, block_size, padding, num_graphs):
    print('reading in {} random blocks'.format(num_graphs))
    roi_offset, roi_shape = pad_total_roi(roi_offset, roi_shape, padding)

    # Check if the decrease of total roi is valid
    assert np.all(roi_shape > 0)

    block_offsets = []
    block_shapes = []
    block_offsets_padded = []
    block_shapes_padded = []

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
        bo_padded, bs_padded = pad_block(offset=total_offset, shape=block_size, padding=padding)
        block_offsets_padded.append(bo_padded)
        block_shapes_padded.append(bs_padded)

    return parse_rois(
        block_offsets=block_offsets,
        block_shapes=block_shapes,
        padded_offsets=block_offsets_padded,
        padded_shapes=block_shapes_padded
    )


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

    return parse_rois(block_offsets=block_offsets, block_shapes=block_shapes)


def save_graphs_to_npz(edge_index, edge_attr, pos, node_ids, edge_mask, y, path, split_name, graph_nr):
    p = os.path.join(path, split_name)
    if not os.path.isdir(p):
        os.makedirs(p)
    logger.debug('saving {} ...'.format(os.path.join(p, 'graph{}'.format(graph_nr))))
    np.savez_compressed(os.path.join(p, 'graph{}'.format(graph_nr)), edge_index=edge_index,
                        edge_attr=edge_attr, pos=pos, node_ids=node_ids, edge_mask=edge_mask, y=y)


def load_graphs_from_npz(path, split_name):
    np.load(os.path.join(path, split_name), allow_pickle=True)


output_data_path = '../data/debug_output_masked'
# output_data_path = '../data/hemi/12_micron_cube'

if not os.path.isdir(output_data_path):
    os.makedirs(output_data_path)
padding = config['pyg_padding']
num_graphs = 10

edge_index, edge_attr, pos, node_ids, edge_mask, targets = create_graphs_random(roi_offset=roi_offset_train,
                                                            roi_shape=roi_shape_train, block_size=block_size_euclidian, num_graphs=num_graphs)
for i, (ei, ea, po, ni, em, t) in enumerate(zip(edge_index, edge_attr, pos, node_ids, edge_mask, targets)):
    save_graphs_to_npz(edge_index=ei, edge_attr=ea, pos=po,
                       node_ids=ni, edge_mask=em, y=t, path=output_data_path, split_name='train', graph_nr=i)

edge_index, edge_attr, pos, node_ids, edge_mask, targets = create_graphs_blockwise_padded(roi_offset=roi_offset_val,
                                                               roi_shape=roi_shape_val, block_size=block_size_euclidian, padding=padding)
for i, (ei, ea, po, ni, em, t) in enumerate(zip(edge_index, edge_attr, pos, node_ids, edge_mask, targets)):
    save_graphs_to_npz(edge_index=ei, edge_attr=ea, pos=po,
                       node_ids=ni, edge_mask=em, y=t, path=output_data_path, split_name='val', graph_nr=i)

edge_index, edge_attr, pos, node_ids, edge_mask, targets = create_graphs_blockwise(roi_offset=roi_offset_test,
                                                               roi_shape=roi_shape_test, block_size=block_size_euclidian)
for i, (ei, ea, po, ni, em, t) in enumerate(zip(edge_index, edge_attr, pos, node_ids, edge_mask, targets)):
    save_graphs_to_npz(edge_index=ei, edge_attr=ea, pos=po,
                       node_ids=ni, edge_mask=em, y=t, path=output_data_path, split_name='test', graph_nr=i)

# print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))
