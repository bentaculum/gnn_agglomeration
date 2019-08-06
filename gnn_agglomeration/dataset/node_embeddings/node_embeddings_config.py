import configargparse
import configparser
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


p = configargparse.ArgParser(
    default_config_files=['config.ini'])

p.add('--config_file', is_config_file=True,
      help='file path to config that overwrites the default configs')

p.add('--num_workers', type=int, help='number of workers for torch DataLoader')
p.add('--samples', type=int, help='number of rag nodes to use for building dataset')
p.add('--patch_size', type=int, nargs=3, help='3D size to use for creating samples in nanometers')
p.add('--raw_channel', type=str2bool, help='if set true, create a channel with raw volumetric patches')
p.add('--mask_channel', type=str2bool, help='if set true, create a channel with binary mask of the fragment in a patch')

config = p.parse_args()
logging.info(f"\n{p.format_values()}")

# TODO adapt
pw_parser = configparser.ConfigParser()
pw_parser.read(config.db_host)
config.db_host = pw_parser['DEFAULT']['db_host']
