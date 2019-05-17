import torch
import argparse
import os
import datetime
import logging
import json
import pytz


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.default = {}

        self.parser.add_argument(
            '--config_from_file',
            type=str,
            help="absolute path to config file to be used (json format)")
        self.default['config_from_file'] = None

        self.parser.add_argument(
            '--nodes',
            type=positive_int,
            help='Number of nodes in the graph')
        self.default['nodes'] = 100

        self.parser.add_argument(
            '--self_loops',
            type=str2bool,
            help='whether the random graph should have reflexive edges')
        self.default['self_loops'] = True

        self.parser.add_argument(
            '--euclidian_dimensionality',
            type=positive_int,
            help='Dimension of the Euclidian space, used in data.pos')
        self.default['euclidian_dimensionality'] = 2

        self.parser.add_argument(
            '--feature_dimensionality',
            type=positive_int,
            help='Dimension of the feature space, used in data.x')
        self.default['feature_dimensionality'] = 2

        self.parser.add_argument(
            '--pseudo_dimensionality',
            type=positive_int,
            help='Dimension of the pseudo coordinates, according to their type')
        self.default['pseudo_dimensionality'] = 2

        self.parser.add_argument(
            '--kernel_size',
            type=positive_int,
            help='Attention: # of heads, Splines: # of control points, GMM: # of mixture components')
        self.default['kernel_size'] = 2

        self.parser.add_argument(
            '--data_transform',
            type=str,
            choices=['Cartesian', 'LocalCartesian', 'Distance', 'Polar'],
            help='define the edge attributes (pseudo coordinates)')
        self.default['data_transform'] = 'Cartesian'

        self.parser.add_argument(
            '--theta_max',
            type=float,
            help='nodes with lower euclidian distance will be connected')
        self.default['theta_max'] = 0.3

        self.parser.add_argument(
            '--theta',
            type=float,
            help='euclidian neighborhood distance')
        self.default['theta'] = 0.1

        self.parser.add_argument(
            '--dataset_path',
            type=str,
            help='the directory to read the Dataset from')
        self.default['dataset_path'] = 'data/example_latest'

        self.parser.add_argument(
            '--run_path',
            type=str,
            help='directory to save temporary outputs')
        self.default['run_path'] = 'runs'

        self.parser.add_argument(
            '--summary_dir',
            type=str,
            help='relative directory to save temporary summary')
        self.default['summary_dir'] = 'summary'

        self.parser.add_argument(
            '--write_summary',
            type=str2bool,
            help='option to use tensorboardx to monitor the training')
        self.default['write_summary'] = True

        self.parser.add_argument(
            '--log_only_gradients',
            type=str2bool,
            help='whether to write gradients to tensorboard')
        self.default['log_only_gradients'] = False

        self.parser.add_argument(
            '--log_histograms',
            type=str2bool,
            help='whether to perform the costly plotting of histograms' )
        self.default['log_histograms'] = False

        self.parser.add_argument(
            '--log_per_epoch_only',
            type=str2bool,
            help='minimal logging, only the loss and the metric' )
        self.default['log_per_epoch_only'] = False

        self.parser.add_argument(
            '--log_namespaces',
            type=str,
            nargs='+',
            help='If you want to log only specific namespaces (e.g. layers), specify them here' )
        self.default['log_namespaces'] = []

        self.parser.add_argument(
            '--model_dir',
            type=str,
            help='relative directory to save temporary model, for both training and inference')
        self.default['model_dir'] = 'model'

        self.parser.add_argument(
            '--plot_error_locations',
            action='store_true',
            help='if passed, the plot errors by location will be created')
        self.default['plot_error_locations'] = False

        self.parser.add_argument(
            '--plot_graphs_testset',
            type=str2bool,
            help='Whether to plot the graphs from the test set for visual inspection')
        self.default['plot_graphs_testset'] = False

        self.parser.add_argument(
            '--validation_split',
            type=unit_float,
            help='define size of validation set, 0 <= ... <= 1')
        self.default['validation_split'] = 0.1

        self.parser.add_argument(
            '--test_split',
            type=unit_float,
            help='define size of test set, 0 <= ... <= 1')
        self.default['test_split'] = 0.1

        self.parser.add_argument(
            '--model',
            type=str,
            help='OurConvModel | GcnModel | GmmConvModel | etc.')
        self.default['model'] = 'OurConvModel'

        self.parser.add_argument(
            '--model_type',
            type=str,
            choices=['ClassificationProblem', 'RegressionProblem'],
            help='ClassificationProblem | RegressionProblem')
        self.default['model_type'] = 'ClassificationProblem'

        self.parser.add_argument(
            '--training_epochs',
            type=positive_int,
            help='number of training epochs')
        self.default['training_epochs'] = 100

        self.parser.add_argument(
            '--samples',
            type=positive_int,
            help='Number of random graphs to create, if a new dataset is created')
        self.default['samples'] = 100

        self.parser.add_argument(
            '--standardize_targets',
            type=str2bool,
            help='targets to mean 0, std 1')
        self.default['standardize_targets'] = True

        self.parser.add_argument(
            '--non_linearity',
            type=str,
            help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh')
        self.default['non_linearity'] = 'leaky_relu'

        self.parser.add_argument(
            '--batch_size_train',
            type=positive_int,
            help='batch size for training')
        self.default['batch_size_train'] = 8

        self.parser.add_argument(
            '--batch_size_eval',
            type=positive_int,
            help='batch size for evaluation')
        self.default['batch_size_eval'] = 8

        self.parser.add_argument(
            '--hidden_layers',
            type=nonnegative_int,
            help='number of hidden layers')
        self.default['hidden_layers'] = 0

        self.parser.add_argument(
            '--hidden_units',
            type=positive_int,
            nargs='+',
            help='number of units per hidden layer in the GNN')
        self.default['hidden_units'] = [1]

        self.parser.add_argument(
            '--use_bias',
            type=str2bool,
            help='whether to use an additive bias')
        self.default['use_bias'] = True

        self.parser.add_argument(
            '--fc_bias',
            type=str2bool,
            help='whether to use a bias term for the final fully connected layer'
        )
        self.default['fc_bias'] = False

        self.parser.add_argument(
            '--dropout_type',
            type=str,
            choices=['dropout', 'dropout2d'],
            help='dropout | dropout2d')
        self.default['dropout_type'] = 'dropout'

        self.parser.add_argument(
            '--dropout_probs',
            type=unit_float,
            nargs='+',
            help='dropout probabilites during training for the input layer and all the hidden layers')
        self.default['dropout_probs'] = [0.0]

        self.parser.add_argument(
            '--adam_lr',
            type=unit_float,
            help='Learning rate for ADAM optimizer')
        self.default['adam_lr'] = 0.005

        self.parser.add_argument(
            '--adam_weight_decay',
            type=unit_float,
            help='Weight decay for ADAM optimizer')
        self.default['adam_weight_decay'] = 0.002

        self.parser.add_argument(
            '--batch_norm',
            type=str2bool,
            help='whether to use Batch Normalization')
        self.default['batch_norm'] = True

        self.parser.add_argument(
            '--clip_grad',
            type=str2bool,
            help='whether to use gradient clipping')
        self.default['clip_grad'] = True

        self.parser.add_argument(
            '--clip_value',
            type=float,
            help='gradients are clipped at this value')
        self.default['clip_value'] = 0.1

        self.parser.add_argument(
            '--clip_method',
            type=str,
            choices=['value', 'inf', '1', '2'],
            help='gradient clipping per value or L_-norm')
        self.default['clip_method'] = 'value'

        self.parser.add_argument(
            '--att_heads_concat',
            type=str2bool,
            help='whether to concat or average the outputs of the different attention heads' )
        self.default['att_heads_concat'] = True

        self.parser.add_argument(
            '--att_final_dropout',
            type=unit_float,
            help='Dropout probability for the final attention vector')
        self.default['att_final_dropout'] = 0.0

        self.parser.add_argument(
            '--att_layers',
            type=positive_int,
            help='Attention NN: number of layers' )
        self.default['att_layers'] = 1

        self.parser.add_argument(
            '--att_layer_dims',
            type=positive_int,
            nargs='+',
            help='Attention NN: list of layer dimensions' )
        self.default['att_layer_dims'] = [1]

        self.parser.add_argument(
            '--att_non_linearity',
            type=str,
            help='Attention NN: torch.nn.functional non linearity to use e.g. relu')
        self.default['att_non_linearity'] = 'leaky_relu'

        self.parser.add_argument(
            '--att_batch_norm',
            type=str2bool,
            help='Attention NN: whether to use batch normalization')
        self.default['att_batch_norm'] = True

        self.parser.add_argument(
            '--att_dropout_probs',
            type=unit_float,
            nargs='+',
            help='Attention NN: dropout probabilites during training for the input layer and all the hidden layers')
        self.default['att_dropout_probs'] = [0.0]

        self.parser.add_argument(
            '--att_bias',
            type=str2bool,
            help='Attention NN: option to use bias')
        self.default['att_bias'] = True

        self.parser.add_argument(
            '--load_model',
            type=str,
            help="Load model from file. 'latest' | relative/path/to/tarfile")
        self.default['load_model'] = None

        self.parser.add_argument(
            '--telegram',
            type=str2bool,
            help='whether to have a Sacred Telegram Observer')
        self.default['telegram'] = True

        self.parser.add_argument(
            '--confusion_matrix_path',
            type=str,
            help='file name of confusion matrix')
        self.default['confusion_matrix_path'] = 'confusion_matrix_test.png'

        self.parser.add_argument(
            '--temp',
            type=str2bool,
            help='If true, save results to temp folder. If false, create timestamped directory.' )
        self.default['temp'] = False

        self.parser.add_argument(
            '--checkpoint_interval',
            type=positive_int,
            help='how often to save a checkpoint of the model that can be used for restarting' )
        self.default['checkpoint_interval'] = 10

        self.parser.add_argument(
            '--machine',
            type=str,
            choices=['localhost', 'slowpoke1'],
            help='machine-dependent parameters to be imported, e.g. for connecting to the MongoDB')
        self.default['machine'] = 'localhost'

        self.parser.add_argument(
            '--msts',
            type=positive_int,
            help='How many different classes in an instance of DiameterGraph')
        self.default['msts'] = 2

        self.parser.add_argument(
            '--class_noise',
            type=unit_float,
            help='how much probability mass is spread between the wrong target classes')
        self.default['class_noise'] = 0.2

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
        if config_cmd['load_model'] is not None:
            if config_cmd['load_model'] == 'latest':
                # find latest model in the runs path
                all_runs_dir = os.path.join(self.default['root_dir'], self.default['run_path'])

                # TODO filter for correct format of directory name, instead of '2019'
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
                    all_runs_dir = os.path.join(self.default['root_dir'], self.default['run_path'])
                    # TODO filter for correct format of directory name, instead of '2019'
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
