import configargparse
import configparser
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

p = configargparse.ArgParser(
    default_config_files=['configs/config_hemi_mtlsd_400k_roi_3.ini'],
    allow_abbrev=False
)

p.add('--config_file', is_config_file=True,
      help='file path to config that overwrites the default configs')

# [FILE PATHS]
p.add(
    '--fragments_zarr',
    type=str,
    help='absolute path to .zarr directory that contains the fragments')
p.add(
    '--fragments_ds',
    type=str,
    help='relative path to fragments in the .zarr file')
p.add(
    '--groundtruth_zarr',
    type=str,
    help='absolute path to .zarr directory that contains the ground truth')
p.add(
    '--groundtruth_ds',
    type=str,
    help='relative path to the ground truth in the .zarr file')

p.add(
    '--fragments_ds_emb',
    type=str,
    help='relative path to the fragments ds in fragments_zarr used to create node embeddings')
p.add(
    '--raw_ds_emb',
    type=str,
    help='relative path to the raw image data in groundtruth_zarr used to create node_embeddings')

p.add('--temp_path', type=str, help='where to temp. store the blockwise outputs')
p.add('--lut_fragments_to_overlap_gt', type=str,
      help='where to store the lookup table from overlapping')
p.add('--overlap_gt_ds', type=str,
      help='where to write the extracted segmentation volume')

p.add('--lut_fragment_segment', type=str,
      help='where to store the segments found by connected components')
p.add('--volume_segmentation', type=str,
      help='where to store the relabelled volume with the final segmentation')

# [DATABASES]
p.add('--db_host', type=str, help='connection to the RAG mongoDB client')
p.add('--db_name', type=str, help='database name')
p.add('--nodes_collection', type=str, help='nodes collection in mongoDB')
p.add('--edges_collection', type=str, help='edges collection in mongoDB')

p.add('--new_node_attr', type=str, help='fragment-wise best effort label')
p.add('--new_edge_attr', type=str, help='binary best effort merge score')
p.add(
    '--new_edge_masking',
    type=str,
    help="binary masking to avoid labelling two adjacent background fragments as 'merge'")
p.add('--new_edge_attr_trinary', type=str,
      help='trinary value: merge, do not merge, unknown')

p.add('--scores_db_name', type=str, help='database for VOI and other metrics')

# [DATA]
p.add(
    '--background_id',
    type=int,
    help='id for background voxels in fragment data')
p.add('--roi_offset', type=int, action='append',
      help='3D ROI offset in nanometers')
p.add('--roi_shape', type=int,
      action='append', help='3D ROI shape in nanometers')
p.add('--block_size', type=int, action='append',
      help='block size used for processing fragments')
p.add('--padding', type=int, action='append',
      help='padding used for fragment creation. Not used at the moment')
p.add('--voxel_size', type=int, action='append',
      help='voxel size in nanometers')
p.add('--voxel_size_emb', type=int, action='append',
      help='voxel size in nanometers used to create node embeddings')

# [DATA PROCESSING]
p.add('--num_workers', type=int, help='number of daisy subprocesses')
p.add(
    '--threshold_overlap',
    type=float,
    help='percentage of overlap required to consider a fragment non-background')
p.add('--con_comp_thresholds_minmax', type=float, action='append',
      help='lower and upper limit for the different runs of connected components')
p.add('--con_comp_thresholds_step', type=float,
      help='step size for threshold of connected components runs')
p.add('--con_comp_score', type=str, help='edge weights for connected components')
p.add('--lut_threshold', type=float, help='lut to use for extracting a volume')

# [META]
p.add('--experiment', type=str)
p.add('--setup', type=str)
p.add('--iteration', type=int)
p.add('--configuration', type=str)
p.add('--volume_size', type=str)

# [MISCELLANEOUS]
p.add(
    '--logging_level',
    type=str,
    choices=[
        'CRITICAL',
        'ERROR',
        'WARNING',
        'INFO',
        'DEBUG',
        'NOTSET'],
    help='basic logging level')

config, remaining_argv = p.parse_known_args()
sys.argv = [sys.argv[0], *remaining_argv]
logger.info(f"\n{p.format_values()}")

pw_parser = configparser.ConfigParser()
pw_parser.read(config.db_host)
config.db_host = pw_parser['DEFAULT']['db_host']
