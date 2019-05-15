import torch
import argparse
import os
import datetime
import logging
import json
import pytz


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument(
            '--nodes',
            type=positive_int,
            default=100,
            help='Number of nodes in the graph')
        self.parser.add_argument(
            '--self_loops',
            type=str2bool,
            default=True,
            help='whether the random graph should have reflexive edges'
        )
        self.parser.add_argument(
            '--euclidian_dimensionality',
            type=positive_int,
            default=2,
            help='Dimension of the Euclidian space, used in data.pos')
        self.parser.add_argument(
            '--feature_dimensionality',
            type=positive_int,
            default=2,
            help='Dimension of the feature space, used in data.x')
        self.parser.add_argument(
            '--pseudo_dimensionality',
            type=positive_int,
            default=2,
            help='Dimension of the pseudo coordinates, according to their type')
        self.parser.add_argument(
            '--kernel_size',
            type=positive_int,
            default=2,
            help='Attention: # of heads, Splines: # of control points, GMM: # of mixture components')
        self.parser.add_argument(
            '--data_transform',
            type=str,
            default='Cartesian',
            choices=['Cartesian', 'LocalCartesian', 'Distance', 'Polar'],
            help='define the edge attributes (pseudo coordinates)')

        self.parser.add_argument(
            '--theta_max',
            type=float,
            default=0.3,
            help='nodes with lower euclidian distance will be connected')
        self.parser.add_argument(
            '--theta',
            type=float,
            default=0.1,
            help='euclidian neighborhood distance')

        self.parser.add_argument(
            '--dataset_path',
            type=str,
            default='data/example_latest',
            help='the directory to read the Dataset from')
        self.parser.add_argument(
            '--run_path',
            type=str,
            default='runs',
            help='directory to save temporary outputs')
        self.parser.add_argument(
            '--summary_dir',
            type=str,
            default='summary',
            help='relative directory to save temporary summary')

        self.parser.add_argument(
            '--no_summary',
            action='store_true',
            default=False,
            help='if passed, tensorboardx will not be used to monitor the training')
        self.parser.add_argument(
            '--log_only_gradients',
            type=str2bool,
            default=False,
            help='whether to write gradients to tensorboard')
        self.parser.add_argument(
            '--log_histograms',
            type=str2bool,
            default=True,
            help='whether to perform the costly plotting of histograms'
        )
        self.parser.add_argument(
            '--log_per_epoch_only',
            type=str2bool,
            default=False,
            help='minimal logging, only the loss and the metric'
        )
        self.parser.add_argument(
            '--log_namespaces',
            type=str,
            nargs='+',
            default=[],
            help='If you want to log only specific namespaces (e.g. layers), specify them here'
        )

        self.parser.add_argument(
            '--model_dir',
            type=str,
            default='model',
            help='relative directory to save temporary model, for both training and inference')
        self.parser.add_argument(
            '--plot_error_locations',
            action='store_true',
            default=False,
            help='if passed, the plot errors by location will be created')
        self.parser.add_argument(
            '--plot_graphs_testset',
            type=str2bool,
            default=False,
            help='Whether to plot the graphs from the test set for visual inspection')

        self.parser.add_argument(
            '--validation_split',
            type=unit_float,
            default=0.1,
            help='define size of validation set, 0 <= ... <= 1')
        self.parser.add_argument(
            '--test_split',
            type=unit_float,
            default=0.1,
            help='define size of test set, 0 <= ... <= 1')

        self.parser.add_argument(
            '--model',
            type=str,
            default='OurConvModel',
            help='OurConvModel | GcnModel | GmmConvModel | etc.')
        self.parser.add_argument(
            '--model_type',
            type=str,
            default='ClassificationProblem',
            choices=['ClassificationProblem', 'RegressionProblem'],
            help='ClassificationProblem | RegressionProblem')
        self.parser.add_argument(
            '--training_epochs',
            type=positive_int,
            default=100,
            help='number of training epochs')
        self.parser.add_argument(
            '--samples',
            type=positive_int,
            default=100,
            help='Number of random graphs to create, if a new dataset is created')
        self.parser.add_argument(
            '--non_linearity',
            type=str,
            default='relu',
            help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh')
        self.parser.add_argument('--batch_size_train', type=positive_int,
                                 default=8, help='batch size for training')
        self.parser.add_argument('--batch_size_eval', type=positive_int,
                                 default=8, help='batch size for evaluation')

        self.parser.add_argument(
            '--hidden_layers',
            type=nonnegative_int,
            default=0,
            help='number of hidden layers')
        self.parser.add_argument(
            '--hidden_units',
            type=positive_int,
            nargs='+',
            default=[1],
            help='number of units per hidden layer in the GNN')
        self.parser.add_argument(
            '--use_bias',
            type=str2bool,
            default=False,
            help='whether to use an additive bias')
        self.parser.add_argument(
            '--dropout_type',
            type=str,
            default='dropout',
            choices=['dropout', 'dropout2d'],
            help='dropout | dropout2d')
        self.parser.add_argument(
            '--dropout_probs',
            type=unit_float,
            nargs='+',
            default=[0.0],
            help='dropout probabilites during training for the input layer and all the hidden layers')
        self.parser.add_argument(
            '--adam_lr',
            type=unit_float,
            default=0.005,
            help='Learning rate for ADAM optimizer')
        self.parser.add_argument(
            '--adam_weight_decay',
            type=unit_float,
            default=0.0005,
            help='Weight decay for ADAM optimizer')
        self.parser.add_argument(
            '--batch_norm',
            type=str2bool,
            default=True,
            help='whether to use Batch Normalization'
        )

        self.parser.add_argument(
            '--att_heads_concat',
            type=str2bool,
            default=True,
            help='whether to concat or average the outputs of the different attention heads'
        )
        self.parser.add_argument(
            '--att_final_dropout',
            type=unit_float,
            default=0.0,
            help='Dropout probability for the final attention vector')

        self.parser.add_argument(
            '--att_layers',
            type=positive_int,
            default=1,
            help='Attention NN: number of layers'
        )
        self.parser.add_argument(
            '--att_layer_dims',
            type=positive_int,
            nargs='+',
            default=[1],
            help='Attention NN: list of layer dimensions'
        )
        self.parser.add_argument(
            '--att_non_linearity',
            type=str,
            default='relu',
            help='Attention NN: torch.nn.functional non linearity to use e.g. relu')
        self.parser.add_argument(
            '--att_batch_norm',
            type=str2bool,
            default=True,
            help='Attention NN: whether to use batch normalization'
        )
        self.parser.add_argument(
            '--att_dropout_probs',
            type=unit_float,
            nargs='+',
            default=[0.0],
            help='dropout probabilites during training for the input layer and all the hidden layers')

        self.parser.add_argument(
            '--load_model',
            type=str,
            default=None,
            help="Load model from file. 'latest' | relative/path/to/tarfile")

        self.parser.add_argument(
            '--telegram',
            type=str2bool,
            default=True,
            help='whether to have a Sacred Telegram Observer'
        )

        self.parser.add_argument(
            '--confusion_matrix_path',
            type=str,
            default='confusion_matrix_test.png',
            help='file name of confusion matrix'
        )

        self.parser.add_argument(
            '--temp',
            type=str2bool,
            default=False,
            help='If true, save results to temp folder. If false, create timestamped directory.'
        )
        self.parser.add_argument(
            '--checkpoint_interval',
            type=positive_int,
            default=10,
            help='how often to save a checkpoint of the model that can be used for restarting'
        )
        self.parser.add_argument(
            '--machine',
            type=str,
            choices=['localhost', 'slowpoke1'],
            default='localhost',
            help='machine-dependent parameters to be imported, e.g. for connecting to the MongoDB'

        )

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

    def parse_args(self):
        config, remaining_args = self.parser.parse_known_args()

        # detect root path, one level up from the config file
        config.root_dir = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )

        # LOAD OLD CONFIG
        if config.load_model is not None:
            if config.load_model == 'latest':
                # find latest model in the runs path
                all_runs_dir = os.path.join(config.root_dir, config.run_path)
                # TODO filter for correct format of directory name, instead of
                # '2019'
                runs = sorted([name for name in os.listdir(
                    all_runs_dir) if name.startswith('2019')])
                rel_run_path = runs[-1]
            else:
                rel_run_path = config.load_model

            with open(os.path.join(config.root_dir, config.run_path, rel_run_path, 'config.json')) as json_file:
                old_config = json.load(json_file)
            new_config = vars(config)

            # overwrite with possible new config variables, and log a warning
            # Unfortunately this will use the config default values if no arguments are passed.
            # So the command line args have to be fully set again for loading a
            # model
            for k, v in new_config.items():
                if v != old_config[k] and v is not None:
                    logging.warning(
                        '{} = {} from loaded config is overwritten with "{}"'.format(
                            k, old_config[k], v))
                    old_config[k] = v

            config = old_config

        # for new model, go to a new directory
        else:
            if config.temp:
                # save to a temporary directory that will be overwritten
                rel_run_path = 'temp'
            else:
                # create a custom directory for each run
                rel_run_path = datetime.datetime.now(
                    pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

            config = vars(config)

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
