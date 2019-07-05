import torch
import argparse
import configargparse
import os
import datetime
import logging
import json
import pytz
from operator import attrgetter


class SortingHelpFormatter(configargparse.HelpFormatter):
    """
    sort arguments when listed on the command line via --help option
    """
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


class Config:
    def __init__(self):
        self.parser = configargparse.ArgumentParser(
            default_config_file=['./config.ini'],
            formatter_class=SortingHelpFormatter)

        self.parser.add_argument('--config_file', is_config_file=True,
            help='file path to config that overwrites the default configs')

        self.parser.add_argument(
            '--nodes',
            type=positive_int,
            help='Number of nodes in the graph')

        self.parser.add_argument(
            '--self_loops',
            type=str2bool,
            help='whether the random graph should have reflexive edges')

        self.parser.add_argument(
            '--euclidian_dimensionality',
            type=positive_int,
            help='Dimension of the Euclidian space, used in data.pos')

        self.parser.add_argument(
            '--feature_dimensionality',
            type=positive_int,
            help='Dimension of the feature space, used in data.x')

        self.parser.add_argument(
            '--pseudo_dimensionality',
            type=positive_int,
            help='Dimension of the pseudo coordinates, according to their type')

        self.parser.add_argument(
            '--kernel_size',
            type=positive_int,
            help='Attention: # of heads, Splines: # of control points, GMM: # of mixture components')

        self.parser.add_argument(
            '--data_transform',
            type=str,
            choices=['Cartesian', 'LocalCartesian', 'Distance', 'Polar'],
            help='define the edge attributes (pseudo coordinates)')

        self.parser.add_argument(
            '--theta_max',
            type=float,
            help='nodes with lower euclidian distance will be connected')

        self.parser.add_argument(
            '--theta',
            type=float,
            help='euclidian neighborhood distance')

        self.parser.add_argument(
            '--dataset_type',
            type=str,
            choices=['DiameterDataset',
                     'CountNeighborsDataset', 'IterativeDataset', 'HemibrainDataset'],
            help='choose from different types of local datasets')

        self.parser.add_argument(
            '--dataset_path',
            type=str,
            help='the directory to read the Dataset from')

        self.parser.add_argument(
            '--run_path',
            type=str,
            help='directory to save temporary outputs')

        self.parser.add_argument(
            '--summary_dir',
            type=str,
            help='relative directory to save temporary summary')

        self.parser.add_argument(
            '--write_summary',
            type=str2bool,
            help='option to use tensorboardx to monitor the training')

        self.parser.add_argument(
            '--log_only_gradients',
            type=str2bool,
            help='whether to write gradients to tensorboard')

        self.parser.add_argument(
            '--log_histograms',
            type=str2bool,
            help='whether to perform the costly plotting of histograms')

        self.parser.add_argument(
            '--log_per_epoch_only',
            type=str2bool,
            help='minimal logging, only the loss and the metric')

        self.parser.add_argument(
            '--log_namespaces',
            type=str,
            nargs='+',
            help='If you want to log only specific namespaces (e.g. layers), specify them here')

        self.parser.add_argument(
            '--model_dir',
            type=str,
            help='relative directory to save temporary model, for both training and inference')

        self.parser.add_argument(
            '--plot_error_locations',
            action='store_true',
            help='if passed, the plot errors by location will be created')

        self.parser.add_argument(
            '--plot_graphs_testset',
            type=int,
            help='how many graphs to plot from the test set for visual inspection. -1 means all')

        self.parser.add_argument(
            '--validation_split',
            type=unit_float,
            help='define size of validation set, 0 <= ... <= 1')

        self.parser.add_argument(
            '--test_split',
            type=unit_float,
            help='define size of test set, 0 <= ... <= 1')

        self.parser.add_argument(
            '--model',
            type=str,
            help='OurConvModel | GcnModel | GmmConvModel | etc.')

        self.parser.add_argument(
            '--model_type',
            type=str,
            choices=['ClassificationProblem', 'RegressionProblem'],
            help='ClassificationProblem | RegressionProblem')

        self.parser.add_argument(
            '--training_epochs',
            type=positive_int,
            help='number of training epochs')

        self.parser.add_argument(
            '--samples',
            type=positive_int,
            help='Number of random graphs to create, if a new dataset is created')

        self.parser.add_argument(
            '--standardize_targets',
            type=str2bool,
            help='targets to mean 0, std 1, if Regression is performed')

        self.parser.add_argument(
            '--non_linearity',
            type=str,
            help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh')

        self.parser.add_argument(
            '--batch_size_train',
            type=positive_int,
            help='batch size for training')

        self.parser.add_argument(
            '--batch_size_eval',
            type=positive_int,
            help='batch size for evaluation')

        self.parser.add_argument(
            '--hidden_layers',
            type=nonnegative_int,
            help='number of hidden layers')

        self.parser.add_argument(
            '--hidden_units',
            type=positive_int,
            nargs='+',
            help='number of units per hidden layer in the GNN')

        self.parser.add_argument(
            '--use_bias',
            type=str2bool,
            help='whether to use an additive bias')

        self.parser.add_argument(
            '--fc_bias',
            type=str2bool,
            help='whether to use a bias term for the final fully connected layer')

        self.parser.add_argument(
            '--fc_layers',
            type=positive_int,
            help='number of fully connected layers in the end, at least 1')

        self.parser.add_argument(
            '--fc_layer_dims',
            type=positive_int,
            nargs='*',
            help='list of hidden layer dims for fc layers in the end')

        self.parser.add_argument(
            '--fc_dropout_probs',
            type=unit_float,
            nargs='*',
            help='fc network: dropout probs')

        self.parser.add_argument(
            '--fc_batch_norm',
            type=str2bool,
            help='whether to use Batch Normalization in the final fully connected layer')

        self.parser.add_argument(
            '--dropout_type',
            type=str,
            choices=['dropout', 'dropout2d'],
            help='dropout | dropout2d')

        self.parser.add_argument(
            '--dropout_probs',
            type=unit_float,
            nargs='+',
            help='dropout probabilites during training for the input layer and all the hidden layers')

        self.parser.add_argument(
            '--adam_lr',
            type=unit_float,
            help='Learning rate for ADAM optimizer')

        self.parser.add_argument(
            '--adam_weight_decay',
            type=unit_float,
            help='Weight decay for ADAM optimizer')

        self.parser.add_argument(
            '--batch_norm',
            type=str2bool,
            help='whether to use Batch Normalization')

        self.parser.add_argument(
            '--clip_grad',
            type=str2bool,
            help='whether to use gradient clipping')

        self.parser.add_argument(
            '--clip_value',
            type=float,
            help='gradients are clipped at this value')

        self.parser.add_argument(
            '--clip_method',
            type=str,
            choices=['value', 'inf', '1', '2'],
            help='gradient clipping per value or L_-norm')

        self.parser.add_argument(
            '--att_heads_concat',
            type=str2bool,
            help='whether to concat or average the outputs of the different attention heads')

        self.parser.add_argument(
            '--att_final_dropout',
            type=unit_float,
            help='Dropout probability for the final attention vector')

        self.parser.add_argument(
            '--att_layers',
            type=positive_int,
            help='Attention NN: number of layers')

        self.parser.add_argument(
            '--att_layer_dims',
            type=positive_int,
            nargs='+',
            help='Attention NN: list of layer dimensions')

        self.parser.add_argument(
            '--att_non_linearity',
            type=str,
            help='Attention NN: torch.nn.functional non linearity to use e.g. relu')

        self.parser.add_argument(
            '--att_batch_norm',
            type=str2bool,
            help='Attention NN: whether to use batch normalization')

        self.parser.add_argument(
            '--att_dropout_probs',
            type=unit_float,
            nargs='+',
            help='Attention NN: dropout probabilites during training for the input layer and all the hidden layers')

        self.parser.add_argument(
            '--att_bias',
            type=str2bool,
            help='Attention NN: option to use bias')

        self.parser.add_argument(
            '--att_normalize',
            type=str2bool,
            help='whether to use a softmax over each neighborhood')

        self.parser.add_argument(
            '--att_nodenet_layers',
            type=positive_int,
            help='number of fc layers before the attention mechanism')

        self.parser.add_argument(
            '--att_nodenet_hidden_dims',
            type=positive_int,
            nargs='*',
            help='hidden dimensionality of nodenet')

        self.parser.add_argument(
            '--load_model',
            type=str,
            help="Load model from file. 'latest' | relative/path/to/tarfile")

        self.parser.add_argument(
            '--telegram',
            type=str2bool,
            help='whether to have a Sacred Telegram Observer')

        self.parser.add_argument(
            '--confusion_matrix_path',
            type=str,
            help='file name of confusion matrix')

        self.parser.add_argument(
            '--temp',
            type=str2bool,
            help='If true, save results to temp folder. If false, create timestamped directory.')

        self.parser.add_argument(
            '--checkpoint_interval',
            type=positive_int,
            help='how often to save a checkpoint of the model that can be used for restarting')

        self.parser.add_argument(
            '--machine',
            type=str,
            choices=[
                'localhost',
                'slowpoke1'],
            help='machine-dependent parameters to be imported, e.g. for connecting to the MongoDB')

        self.parser.add_argument(
            '--msts',
            type=positive_int,
            help='How many different classes in an instance of DiameterGraph')

        self.parser.add_argument(
            '--class_noise',
            type=unit_float,
            help='how much probability mass is spread between the wrong target classes')

        self.parser.add_argument(
            '--class_label_feature',
            type=str2bool,
            help='whether to use the noisy class labels as a feature')

        self.parser.add_argument(
            '--affinity_dist_alpha',
            type=float,
            help='alpha value for the beta dist that generates noisy edge features')

        self.parser.add_argument(
            '--affinity_dist_beta',
            type=float,
            help='beta value for the beta dist that generates noisy edge features')

        self.parser.add_argument(
            '--affinities',
            type=str,
            choices=['all_affinities', 'only_gt_affinities',
                     'only_gt_dir_affinities'],
            help='what kind of affinities on the edges')

        self.parser.add_argument(
            '--curvature_degree_limit',
            type=positive_int,
            help='Restrict the deviation from the continuing line by this amount of degrees, in both dirs')

        self.parser.add_argument(
            '--edge_labels',
            type=str2bool,
            help='whether to use edge labels')

        self.parser.add_argument(
            '--fc_use_edge',
            type=str2bool,
            help='whether to use the edge information for the final fc layer')

        self.parser.add_argument(
            '--use_latest_config',
            action='store_true',
            help='use custom .json loader to re-use the latest config file'
        )

    # TODO move this to separate config files
    def localhost(self):
        return {
            'mongo_url': 'localhost:27017',
            'mongo_db': 'sacred',
        }

    def slowpoke1(self):
        return {
            'mongo_url': 'slowpoke1.int.janelia.org:27017',
            'mongo_db': 'sacred',
        }

    def overwrite_defaults(self, config_filepath):
        with open(config_filepath) as json_file:
            self.default.update(json.load(json_file))

    def update_defaults_with_cmd_args(self, config):
        # overwrite with possible new config variables, and log a warning
        for k, v in config.items():
            if v is not None:
                logging.warning(
                    '{} = {} from defaults is overwritten with "{}"'.format(
                        k, self.default[k], v))
                self.default[k] = v
        return self.default

    def parse_args(self):
        config_cmd, remaining_args = self.parser.parse_known_args()
        config_cmd = vars(config_cmd)

        # TODO could also be dependent on the position of main.py
        # detect root path, one level up from the config file
        self.default['root_dir'] = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))

        # load old config and set as default for continuation of training
        if config_cmd['load_model']:
            if config_cmd['load_model'] == 'latest':
                # find latest model in the runs path
                all_runs_dir = os.path.join(
                    self.default['root_dir'], self.default['run_path'])

                # TODO filter for correct format of directory name, instead of
                # '2019'
                runs = sorted([name for name in os.listdir(
                    all_runs_dir) if name.startswith('2019')])

                rel_run_path = runs[-1]
                config_cmd['load_model'] = runs[-1]
            else:
                rel_run_path = config_cmd['load_model']

            config_filepath = os.path.join(
                self.default['root_dir'],
                self.default['run_path'],
                rel_run_path,
                'config.json')
            self.overwrite_defaults(config_filepath)
            config = self.update_defaults_with_cmd_args(config_cmd)

        # for new model, go to a new directory
        else:
            # allow for loading a config from file, set values as default
            if config_cmd['config_from_file']:

                # find latest model in the runs path
                if config_cmd['config_from_file'] == 'latest':
                    all_runs_dir = os.path.join(
                        self.default['root_dir'], self.default['run_path'])
                    # TODO filter for correct format of directory name, instead
                    # of '2019'
                    runs = sorted([name for name in os.listdir(
                        all_runs_dir) if name.startswith('2019')])
                    config_filepath = os.path.join(
                        self.default['root_dir'],
                        self.default['run_path'],
                        runs[-1], 'config.json')
                    config_cmd['config_from_file'] = config_filepath
                    self.overwrite_defaults(config_filepath)
                else:
                    self.overwrite_defaults(config_cmd['config_from_file'])

            config = self.update_defaults_with_cmd_args(config_cmd)

            if config['temp']:
                # save to a temporary directory that will be overwritten
                rel_run_path = 'temp'
            else:
                # create a custom directory for each run
                rel_run_path = datetime.datetime.now(
                    pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

        # set the absolute paths in the config file
        config['run_abs_path'] = os.path.join(
            config['root_dir'], config['run_path'], rel_run_path)
        config['dataset_abs_path'] = os.path.join(
            config['root_dir'], config['dataset_path'])
        config.update(getattr(self, config['machine'])())

        return config, remaining_args


def unit_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
