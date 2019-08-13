import configargparse
import logging
import sys

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


def list_of_ints(v):
    return list(map(int, v.replace('[', '').replace(']', '').split(' ')))


p = configargparse.ArgParser(
    default_config_files=['node_embeddings/config_siamese.ini'])

p.add('--config_file', is_config_file=True,
      help='file path to config that overwrites the default configs')

p.add('--runs_dir', type=str)
p.add('--summary', type=str2bool)
p.add('--summary_interval', type=int)
p.add('--checkpoint_interval', type=int)

# TODO no redundant help messages
p.add('--num_workers_dataloader', type=int, help='number of workers for torch DataLoader')
p.add('--training_samples', type=int, help='number of rag edges to used for training')
p.add('--patch_size', type=int, nargs=3, help='3D size to use for creating samples in nanometers')
p.add('--raw_channel', type=str2bool, help='if set true, create a channel with raw volumetric patches')
p.add('--mask_channel', type=str2bool, help='if set true, create a channel with binary mask of the fragment in a patch')
p.add('--batch_size', type=int)

p.add('--fmaps', type=int, help='number of channels, to be doubled per layer')
p.add('--fmaps_max', type=int)
p.add('--output_features', type=int, help='dimensionality of embeddings before loss')
p.add('--downsample_factors', type=list_of_ints, nargs='+', help='tuple of 3D downsample factors for each pooling layer')

p.add('--adam_lr', type=float, help='learning rate for adam optimizer')
p.add('--adam_weight_decay', type=float, help='weight decay for adam optimizer')

p.add('--load_model', type=str, help='latest | path/to/tarfile')
p.add('--load_model_version', type=str, help='latest | checkpoint name, including extension')
p.add('--inference_samples', type=str, help='`all` | threshold number')

config, remaining_argv = p.parse_known_args()
sys.argv = [sys.argv[0], *remaining_argv]
logger.info(f"\n{p.format_values()}")
