import configargparse
import configparser
import logging

p = configargparse.ArgParser(default_config_files=['./config.ini'])
p.add('--config_file', is_config_file=True,
      help='file path to config that overwrites the default configs')

# [FILE PATHS]
p.add('--temp_path', type=str, help='where to temp. store the blockwise outputs')
p.add('--fragments_zarr', type=str, help='absolute path to .zarr directory that contains the fragments')
p.add('--fragments_ds', type=str, help='relative path to fragments in the .zarr file')
p.add('--groundtruth_zarr', type=str, help='absolute path to .zarr directory that contains the ground truth')
p.add('--groundtruth_ds', type=str, help='relative path to the ground truth in the .zarr file')

# [DATABASE]
p.add('--db_host', type=str, help='connection to the RAG mongoDB client')
p.add('--db_name', type=str, help='database name')
p.add('--nodes_collection', type=str, help='nodes collection in mongoDB')
p.add('--edges_collection', type=str, help='edges collection in mongoDB')

# [DATA]
p.add('--background_id', type=int, help='id for background voxels in fragment data')
p.add('--roi_offset', type=int, nargs='+', help='3D ROI offset in nanometers')
p.add('--roi_shape', type=int, nargs='+', help='3D ROI shape in nanometers')

# [DATA PROCESSING]
p.add('--threshold_overlap', type=float, help='percentage of overlap required to consider a fragment non-background')
p.add('--num_workers', type=int, help='number of daisy subprocesses')
p.add('--block_size', type=int, nargs='+', help='block size used for processing fragments')

# [MISCELLANEOUS]
p.add('--logging_level', type=str, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
      help='basic logging level')


config = p.parse_args()
logging.info(f"\n{p.format_values()}")

pw_parser = configparser.ConfigParser()
pw_parser.read(config.db_host)
config.db_host = pw_parser['DEFAULT']['db_host']

