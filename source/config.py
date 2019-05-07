import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument(
            '--nodes',
            type=int,
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
            type=int,
            default=2,
            help='Dimension of the Euclidian space, used in data.pos')
        self.parser.add_argument(
            '--feature_dimensionality',
            type=int,
            default=2,
            help='Dimension of the feature space, used in data.x')
        self.parser.add_argument(
            '--pseudo_dimensionality',
            type=int,
            default=2,
            help='Dimension of the pseudo coordinates, according to their type')
        self.parser.add_argument('--kernel_size', type=int,
                                 default=1, help='kernel size for SplineConv')
        self.parser.add_argument(
            '--data_transform',
            type=str,
            default='Cartesian',
            help='define the edge attributes (pseudo coordinates) of the graphs e.g. Cartesian | Distance | LocalCartesian | Polar')

        self.parser.add_argument(
            '--theta_max',
            type=float,
            default=0.2,
            help='nodes with lower euclidian distance will be connected')
        self.parser.add_argument(
            '--theta',
            type=float,
            default=0.1,
            help='euclidian neighborhood distance')

        self.parser.add_argument(
            '--dataset_path',
            type=str,
            default='../data/example10',
            help='the directory to read the Dataset from')
        self.parser.add_argument(
            '--temp_dir',
            type=str,
            default='../temp',
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
            '--validation_split',
            type=float,
            default=0.1,
            help='define size of validation set, 0 <= ... <= 1')
        self.parser.add_argument(
            '--test_split',
            type=float,
            default=0.1,
            help='define size of test set, 0 <= ... <= 1')

        self.parser.add_argument(
            '--model',
            type=str,
            default='OurConvModel',
            help='GcnModel | GmmConvModel | etc.')
        self.parser.add_argument(
            '--model_type',
            type=str,
            default='ClassificationProblem',
            help='ClassificationProblem | RegressionProblem')
        self.parser.add_argument(
            '--training_epochs',
            type=int,
            default=100,
            help='number of training epochs')
        self.parser.add_argument(
            '--hidden_layers',
            type=int,
            default=0,
            help='number of hidden layers')
        self.parser.add_argument(
            '--hidden_units',
            type=int,
            default=16,
            help='number of units per hidden layer in the GNN')
        self.parser.add_argument(
            '--use_bias',
            type=str2bool,
            default=False,
            help='whether to use an additive bias')
        self.parser.add_argument(
            '--samples',
            type=int,
            default=100,
            help='Number of random graphs to create, if a new dataset is created')
        self.parser.add_argument(
            '--non_linearity',
            type=str,
            default='relu',
            help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh')
        self.parser.add_argument('--batch_size_train', type=int,
                                 default=1, help='batch size for training')
        self.parser.add_argument('--batch_size_eval', type=int,
                                 default=1, help='batch size for evaluation')
        self.parser.add_argument('--dropout_type', type=str,
                                 default='dropout', help='dropout | dropout2d')
        self.parser.add_argument(
            '--dropout_prob',
            type=float,
            default=0.5,
            help='dropout probability during training')
        self.parser.add_argument(
            '--adam_lr',
            type=float,
            default=0.005,
            help='Learning rate for ADAM optimizer')
        self.parser.add_argument(
            '--adam_weight_decay',
            type=float,
            default=0.0005,
            help='Weight decay for ADAM optimizer')

        self.parser.add_argument(
            '--att_dropout',
            type=float,
            default=0.0,
            help='Dropout probability for the final attention vector')

        self.parser.add_argument(
            '--att_layers',
            type=int,
            default=1,
            help='Attention NN: number of layers'
        )
        self.parser.add_argument(
            '--att_layer_dims',
            type=int,
            nargs='+',
            default=[1],
            help='Attention NN: list of layer dimensions'
        )
        self.parser.add_argument(
            '--att_non_linearity',
            type=str,
            default='relu',
            help='Attention NN: torch.nn.functional non linearity to use e.g. relu'
        )

        self.parser.add_argument(
            '--load_model',
            type=str,
            default=None,
            help="Load model from file. 'latest' | relative/path/to/tarfile")

    def parse_args(self):
        return self.parser.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
