from pymongo import MongoClient
import daisy
import json
import logging
import lsd
# import malis
import numpy as np
import os
# import scipy
import sys
import waterz
from funlib.segment.arrays import replace_values
from funlib.evaluate import rand_voi

from .config import config

logging.basicConfig(level=logging.INFO)


def evaluate(
        experiment,
        setup,
        iteration,
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        db_host,
        rag_db_name,
        edges_collection,
        scores_db_name,
        thresholds_minmax,
        thresholds_step,
        num_workers,
        configuration,
        volume_size,
        lut_fragment_segment,
        relabel=False,
        erode=False,
        border_threshold=None):

    # open fragments
    logging.info("Reading fragments from %s" % fragments_file)
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = daisy.persistence.MongoDbGraphProvider(
        rag_db_name,
        host=db_host,
        mode='r',
        edges_collection=edges_collection,
        position_attribute=['center_z', 'center_y', 'center_x'])
    logging.info("RAG DB opened")

    total_roi = fragments.roi

    # slice
    logging.info("Reading fragments and RAG in %s", total_roi)
    fragments = fragments[total_roi]
    rag = rag_provider[total_roi]

    logging.info("Number of nodes in RAG: %d", len(rag.nodes()))
    logging.info("Number of edges in RAG: %d", len(rag.edges()))

    # read gt data
    gt = daisy.open_ds(gt_file, gt_dataset)
    common_roi = fragments.roi.intersect(gt.roi)

    read_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))
    write_roi = daisy.Roi((0, 0, 0), (2048, 2048, 2048))

    # evaluate only where we have both fragments and GT
    logging.info("Cropping fragments and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    if relabel:

        # relabel connected components
        logging.info("Relabelling connected components in GT...")
        gt.materialize()
        components = gt.data
        dtype = components.dtype
        simple_neighborhood = malis.mknhood3d()
        affs_from_components = malis.seg_to_affgraph(
            components,
            simple_neighborhood
        )
        components, _ = malis.connected_components_affgraph(
            affs_from_components,
            simple_neighborhood
        )
        # curate GT
        components[gt.data > np.uint64(-10)] = 0
        gt.data = components.astype(dtype)

    if erode:

        logging.info("Creating 2D border mask...")
        for z in range(gt.data.shape[0]):
            border_mask = create_border_mask_2d(
                gt.data[z],
                float(border_threshold)/gt.voxel_size[1])
            gt.data[z][border_mask] = 0

    logging.info("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    logging.info("Evaluating thresholds...")
    for threshold in thresholds:

        segment_ids = get_segmentation(
            fragments,
            fragments_file,
            lut_fragment_segment,
            edges_collection,
            threshold)

        evaluate_threshold(
            experiment,
            setup,
            iteration,
            db_host,
            scores_db_name,
            edges_collection,
            segment_ids,
            gt,
            threshold,
            configuration,
            volume_size)


def get_segmentation(
        fragments,
        fragments_file,
        lut_fragment_segment,
        edges_collection,
        threshold):

    logging.info(
        "Loading fragment - segment lookup table for threshold %s..." % threshold)
    fragment_segment_lut_dir = os.path.join(
        fragments_file,
        lut_fragment_segment)

    fragment_segment_lut_file = os.path.join(
        fragment_segment_lut_dir,
        'seg_%s_%d.npz' % (edges_collection, int(threshold*100)))

    fragment_segment_lut = np.load(
        fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    # fragments = fragments.to_ndarray(block.write_roi)

    logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(
        fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids


def evaluate_threshold(
        experiment,
        setup,
        iteration,
        db_host,
        scores_db_name,
        edges_collection,
        segment_ids,
        gt,
        threshold,
        configuration,
        volume_size):

    # open score DB
    client = MongoClient(db_host)
    database = client[scores_db_name]
    score_collection = database['scores']

    # get VOI and RAND
    logging.info("Calculating VOI scores for threshold %f...", threshold)

    logging.info(type(segment_ids))

    rand_voi_report = rand_voi(
        gt,
        segment_ids,
        return_cluster_scores=False)

    metrics = rand_voi_report.copy()

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    logging.info("Storing VOI values for threshold %f in DB" % threshold)

    metrics['threshold'] = threshold
    metrics['experiment'] = experiment
    metrics['setup'] = setup
    metrics['iteration'] = iteration
    metrics['network'] = configuration
    metrics['volume_size'] = volume_size
    metrics['merge_function'] = edges_collection.strip('edges_')

    logging.info(metrics)

    score_collection.replace_one(
        filter={
            'network': metrics['network'],
            'volume_size': metrics['volume_size'],
            'merge_function': metrics['merge_function'],
            'threshold': metrics['threshold']
        },
        replacement=metrics,
        upsert=True)


def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.

    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)

    padded = np.pad(image, 1, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(image == padded[:-2, 1:-1], image == padded[2:, 1:-1]),
        np.logical_and(image == padded[1:-1, :-2], image == padded[1:-1, 2:])
    )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
    )

    return distances <= max_dist


if __name__ == "__main__":

    evaluate(
        experiment=config.experiment,
        setup=config.setup,
        iteration=config.iteration,
        gt_file=config.groundtruth_zarr,
        gt_dataset=config.groundtruth_ds,
        fragments_file=config.fragments_zarr,
        fragments_dataset=config.fragments_ds,
        db_host=config.db_host,
        rag_db_name=config.db_name,
        edges_collection=config.edges_collection,
        scores_db_name=config.scores_db_name,
        thresholds_minmax=config.con_comp_thresholds_minmax,
        thresholds_step=config.con_comp_thresholds_step,
        num_workers=config.num_workers,
        configuration=config.configuration,
        volume_size=config.volume_size,
        lut_fragment_segment=config.lut_fragment_segment,
        relabel=False,
        erode=False,
        border_threshold=None
    )
