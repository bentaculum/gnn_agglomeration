import sys
import json
import logging
import daisy
import time
import pickle
import networkx as nx

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_file = sys.argv[1]

with open(config_file, 'r') as f:
    config = json.load(f)

graph_provider = daisy.persistence.MongoDbGraphProvider(
    config['db_name'],
    config['db_host'],
    mode='r+',
    nodes_collection=config['nodes_collection'],
    edges_collection=config['edges_collection'],
    endpoint_names=['u', 'v'],
    position_attribute=[
        'center_z',
        'center_y',
        'center_x'])

roi_offset = config['roi_offset']
roi_shape = config['roi_shape']
roi = daisy.Roi(offset=roi_offset, shape=roi_shape)

start = time.time()
# Get all node and edge attributes
graph = graph_provider.get_graph(roi=roi)
logger.debug("Loaded graph in {0:.3f} seconds".format(time.time() - start))

gt = pickle.load(open(config['frag_to_gt_path'], "rb"))

# Convert all values from np.uint64 to dtype processable by mongodb
for k, v in gt.items():
    gt[k] = int(v)

new_node_attr = 'segment_id'
new_edge_attr = 'merge_ground_truth'
new_edge_masking = 'merge_labeled'

nx.set_node_attributes(graph, values=gt, name=new_node_attr)

# Two binary values:
# - merge_ground_truth is 1 if two fragments have the same id, 0 otherwise
# - merge_labeled is 1 if two fragment ids are non-zero and the same, 0 otherwise
edge_gt = {}
edge_labeled = {}
background_id = 0

start = time.time()
for u, v in graph.edges(data=False):
    if u not in gt or v not in gt:
        edge_label = 0
        labeled = 0
    else:
        if gt[u] == gt[v]:
            edge_label = 1
        else:
            edge_label = 0

        if gt[u] == background_id and gt[v] == background_id:
            labeled = 0
        else:
            labeled = 1

    edge_gt[(u, v)] = edge_label
    edge_labeled[(u, v)] = labeled

logger.debug('Computed edge ground truth in {0:.3f} seconds'.format(
    time.time() - start))

nx.set_edge_attributes(graph, values=edge_gt, name=new_edge_attr)
nx.set_edge_attributes(graph, values=edge_labeled, name=new_edge_masking)

start = time.time()
graph.update_node_attrs(roi=roi, attributes=[new_node_attr])
logger.debug('Updated nodes in {0:.3f} s'.format(time.time() - start))

start = time.time()
graph.update_edge_attrs(roi=roi, attributes=[new_edge_attr, new_edge_masking])
logger.debug('Updated edges in {0:.3f} s'.format(time.time() - start))
