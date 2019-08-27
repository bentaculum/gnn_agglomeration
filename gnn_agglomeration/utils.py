import torch
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def log_max_memory_allocated(device):
    if torch.cuda.is_available():
        logger.debug(
            f'max GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / (2**30):.3f} GiB')
        torch.cuda.reset_max_memory_allocated(device=device)


def output_similarities_split(writer, iteration, out0, out1, labels):
    mask_split = labels == 1
    mask_merge = labels == 0
    assert torch.all(mask_split | mask_merge)
    output_similarities = torch.nn.functional.cosine_similarity(
        out0, out1, dim=1)

    if len(output_similarities[mask_split]) > 0:
        writer.add_histogram(
            '01/output_similarities/split_similarity_should_be_-1',
            output_similarities[mask_split],
            iteration
        )
    if len(output_similarities[~mask_split]) > 0:
        writer.add_histogram(
            '01/output_similarities/merge_similarity_should_be_1',
            output_similarities[~mask_split],
            iteration
        )


class TooManyEdgesException(Exception):
    pass
