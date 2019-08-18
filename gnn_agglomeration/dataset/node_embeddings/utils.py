import time
import numpy as np
import logging
import os
import os.path as osp
import re
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def drop_outgoing_edges(node_attrs, edge_attrs, id_field, node1_field, node2_field):
    """
    drop edges for which at least one node is not part of the node dictionary

    Args:
        node_attrs (dict):

            dictionary of node features, where each feature is a numpy array of equal length

        edge_attrs(dict):

            dictionary of edge features, where each feature is a numpy array of equal length

        id_field (str):

            dict key for the id in node_attrs

        node1_field(str):

            dict key for node 1 in edge_attrs

        node2_field:

            dict key for node 2 in edge_attrs

    Returns:
        dict: clean edge dictionary
    """

    start = time.time()
    u_in = np.isin(edge_attrs[node1_field], node_attrs[id_field])
    v_in = np.isin(edge_attrs[node2_field], node_attrs[id_field])
    edge_in = np.logical_and(u_in, v_in)
    for attr, vals in edge_attrs.items():
        edge_attrs[attr] = vals[edge_in]

    logger.debug(f'drop edges at the border in {time.time() - start}s')
    return edge_attrs


def to_np_arrays(list_of_dicts):
    """
    transforms a list of dictionaries with identical keys into one dict of numpy arrays

    Args:
        list_of_dicts(list): list of dictionaries with identical keys

    Returns:
        dict: one numpy array per key in input dicts
    """

    d = {}
    for i in list_of_dicts:
        for k, v in i.items():
            d.setdefault(k, []).append(v)
    for k, v in d.items():
        d[k] = np.array(v)
    return d


def load_checkpoint(load_model, load_model_version, runs_dir):
    if load_model == 'latest':
        # find latest model in the runs path
        # TODO filter for correct format of directory name, instead of
        # '2019'
        runs = sorted([name for name in os.listdir(
            runs_dir) if name.startswith('2019')])

        # take second last, as the newest dir is the one for the current run
        run_path = osp.join(runs_dir, runs[-2])
    else:
        run_path = load_model

    # find latest state of the model
    def extract_number(f):
        s = re.findall(r'\d+', f)
        return int(s[0]) if s else -1, f

    load_model_dir = os.path.join(run_path, 'model')
    checkpoint_versions = [name for name in os.listdir(
        load_model_dir) if name.endswith('.tar')]
    if load_model_version == 'latest':
        if 'final.tar' in checkpoint_versions:
            checkpoint_to_load = 'final.tar'
        else:
            checkpoint_versions = [
                v for v in checkpoint_versions if v.startswith('iteration')]
            checkpoint_to_load = max(checkpoint_versions, key=extract_number)
    else:
        checkpoint_to_load = load_model_version

    logger.info(
        f'Load model {load_model_dir}, checkpoint {checkpoint_to_load}')
    checkpoint = torch.load(os.path.join(
        load_model_dir, checkpoint_to_load))

    return checkpoint
