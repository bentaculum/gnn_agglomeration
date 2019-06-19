import configargparse
import daisy
import numpy as np
import time
import logging
from collections import Counter
import pickle
import sys
import json
import shutil
import os

logging.basicConfig(level=logging.INFO)
# logging.getLogger('daisy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = configargparse.ArgParser(default_config_files=['./config.ini'])
    parser.add('--config_file', required=False, is_config_file=True,
               help='file path to config that overwrites the default configs')

    parser.add('--fragments_zarr', type=str, help='absolute path to .zarr directory that contains the fragments')
    parser.add('--fragments_ds', type=str, help='relative path to fragments in the .zarr file')
    parser.add('--groundtruth_zarr', type=str, help='absolute path to .zarr directory that contains the ground truth')
    parser.add('--groundtruth_ds', type=str, help='relative path to the ground truth in the .zarr file')

    options = parser.parse_args()

    logger.info(f"\n{parser.format_values()}")

    return options


if __name__ == '__main__':
    logger.info('Parsing arguments ...')
    config = parse_args()
