from pymongo import MongoClient
import sys
import numpy as np
import pandas as pd
import time
import json

import daisy

# client_address = sys.argv[1]

# client = MongoClient(client_address)

# db = client.hemi_mtlsd_400k
# db_nodes = db.nodes
# db_edges = db.edges_hist_quant_50

# print('Loading RAG nodes from DB ...')
# cursor = db_nodes.find()
# nodes = pd.DataFrame(list(cursor))

# print('Loading RAG edges from DB ...')
# cursor = db_edges.find()

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)

# edges = pd.DataFrame(list(cursor))

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


# None values for unboundedness
roi_offset = config['roi_offset']
roi_shape = config['roi_shape']
roi = daisy.Roi(
    roi_offset,
    roi_shape)

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
