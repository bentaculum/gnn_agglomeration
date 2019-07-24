import daisy
import numpy as np
import time
import logging
from collections import Counter
import pickle
import shutil
import os
import networkx as nx
import datetime
import pytz

from config import config

logging.basicConfig(level=getattr(logging, config.logging_level))
# logging.getLogger('daisy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def overlap_in_block(block, fragments, groundtruth, tmp_path):
    fragments = fragments.to_ndarray(block.read_roi)
    groundtruth = groundtruth.to_ndarray(block.read_roi)

    # get all fragment ids in this block
    frag_ids = np.unique(fragments)
    logger.debug(f"num of fragment IDs: {(len(frag_ids))}")

    frag_dict = dict()
    # for each of them, create a boolean mask and count the remaining elems
    for i in frag_ids:
        masked_gt = groundtruth[fragments == i]
        unique, counts = np.unique(masked_gt, return_counts=True)

        # write counter into dict as frag:Counter
        counter = Counter(dict(zip(unique, counts)))

        max_count = counter.most_common(1)[0][1]
        all_counts = sum(counter.values())
        if max_count / all_counts > config.threshold_overlap:
            # most common elem
            frag_dict[i] = int(counter.most_common(1)[0][0])
        else:
            frag_dict[i] = int(config.background_id)

    logger.debug(
        f"write Counter dict for block {block.block_id} to file")
    pickle.dump(frag_dict, open(os.path.join(
        tmp_path, f"{block.block_id}.pickle"), 'wb'))


def overlap_reduce(tmp_path):
    block_dicts = []
    for f in os.listdir(tmp_path):
        if f.endswith(".pickle"):
            block_dicts.append(pickle.load(
                open(os.path.join(tmp_path, f), 'rb')))

    logger.info(
        f"Found {len(block_dicts)} block results in {tmp_path}")

    merged_dicts = dict()
    for b in block_dicts:
        for k in b.keys():
            if k in merged_dicts:
                logger.warning(
                    f"fragment id {k} already exists in previous block")

        merged_dicts.update(b)

    return merged_dicts


def overlap():
    run_datetime = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')
    temp_dir = os.path.join(config.temp_path, run_datetime)
    os.makedirs(temp_dir)

    fragments = daisy.open_ds(config.fragments_zarr, config.fragments_ds)
    groundtruth = daisy.open_ds(config.groundtruth_zarr, config.groundtruth_ds)
    total_roi = daisy.Roi(offset=config.roi_offset, shape=config.roi_shape)

    start = time.time()
    daisy.run_blockwise(
        total_roi=total_roi,
        read_roi=daisy.Roi(offset=(0, 0, 0), shape=config.block_size),
        write_roi=daisy.Roi(offset=(0, 0, 0), shape=config.block_size),
        process_function=lambda block: overlap_in_block(
            block=block,
            fragments=fragments,
            groundtruth=groundtruth,
            tmp_path=temp_dir),
        fit='shrink',
        num_workers=config.num_workers,
        read_write_conflict=False,
        max_retries=1)

    logger.info(
        f"Blockwise overlapping of fragments and ground truth in {time.time() - start:.3f}s")
    logger.debug(
        f"num blocks: {np.prod(np.ceil(np.array(config.roi_shape) / np.array(config.block_size)))}")

    frag_to_gt = overlap_reduce(tmp_path=temp_dir)

    pickle.dump(frag_to_gt, open(os.path.join(
        temp_dir, 'frag_to_gt.pickle'), 'wb'))
    return frag_to_gt


def update_rag_db_with_gt(gt):
    graph_provider = daisy.persistence.MongoDbGraphProvider(
        config.db_name,
        config.db_host,
        mode='r+',
        nodes_collection=config.nodes_collection,
        edges_collection=config.edges_collection,
        endpoint_names=['u', 'v'],
        position_attribute=[
            'center_z',
            'center_y',
            'center_x'])

    roi = daisy.Roi(offset=config.roi_offset, shape=config.roi_shape)

    start = time.time()
    # Get all node and edge attributes
    graph = graph_provider.get_graph(roi=roi)
    logger.debug(f"Loaded graph in {time.time() - start:.3f} s")

    # Convert all values from np.uint64 to dtype processable by mongodb
    for k, v in gt.items():
        gt[k] = int(v)

    nx.set_node_attributes(graph, values=gt, name=config.new_node_attr)

    # Two binary values:
    # - gt_merge_score is 0 if two fragments have the same id, 1 otherwise (lower means merge earlier)
    # - merge_labeled is 0 if both fragments are background, 1 otherwise

    # Trinary value: gt_merge_score_trinary

    edge_gt = {}
    edge_labeled = {}
    edge_gt_trinary = {}

    start = time.time()
    for u, v in graph.edges(data=False):
        if u not in gt or v not in gt:
            # maybe edges going out of the ROI
            # TODO careful with this edge case
            logger.warning(
                f'nodes of edge {u} - {v} not contained in overlap ground truth')
            edge_label = 1
            labeled = 0
            gt_trinary = None
        else:
            if gt[u] == gt[v]:
                edge_label = 0
            else:
                edge_label = 1

            if gt[u] == config.background_id and gt[v] == config.background_id:
                labeled = 0
            else:
                labeled = 1

            # TODO either trinary- or two-binaries-solution
            if gt[u] == config.background_id and gt[v] == config.background_id:
                gt_trinary = None
            elif gt[u] == gt[v]:
                gt_trinary = 0
            else:
                gt_trinary = 1

        edge_gt[(u, v)] = edge_label
        edge_labeled[(u, v)] = labeled
        edge_gt_trinary[(u, v)] = gt_trinary

    logger.debug(f"Computed edge ground truth in {time.time() - start:.3f} s")
    assert len(edge_gt) == len(graph.edges(data=False))

    nx.set_edge_attributes(graph, values=edge_gt, name=config.new_edge_attr)
    nx.set_edge_attributes(graph, values=edge_labeled,
                           name=config.new_edge_masking)
    nx.set_edge_attributes(graph, values=edge_gt_trinary,
                           name=config.new_edge_attr_trinary)

    start = time.time()
    graph.update_node_attrs(roi=roi, attributes=[config.new_node_attr])
    logger.debug(f"Updated nodes in {time.time() - start:.3f} s")

    start = time.time()
    graph.update_edge_attrs(
        roi=roi,
        attributes=[
            config.new_edge_attr,
            config.new_edge_masking,
            config.new_edge_attr_trinary])
    logger.debug(f"Updated edges in {time.time() - start:.3f} s")


def save_to_lookup_table(gt):
    start = time.time()
    lut = np.array([list(gt.keys()), list(gt.values())], dtype=np.uint64)

    # stick to naming convention for re-using lsd experiments script
    filename = 'seg_%s_%d' % (config.edges_collection,
                              int(config.threshold_overlap * 100))
    if not os.path.isdir(
        os.path.join(
            config.fragments_zarr,
            config.lut_fragments_to_overlap_gt)):
        os.makedirs(os.path.join(config.fragments_zarr,
                                 config.lut_fragments_to_overlap_gt))
    out_file = os.path.join(config.fragments_zarr,
                            config.lut_fragments_to_overlap_gt, filename)

    np.savez_compressed(out_file, fragment_segment_lut=lut)
    logger.debug(
        f"Saved overlap relabelling to LUT in {time.time() - start:.3f} s")


if __name__ == '__main__':
    ground_truth = overlap()
    save_to_lookup_table(ground_truth)
    update_rag_db_with_gt(ground_truth)
