import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--nodes', type=int,
                                 default=100, help='Number of nodes in the graph')
        self.parser.add_argument('--euclidian_dimensionality', type=int,
                                 default=2, help='Dimension of the Euclidian space, used in data.pos')
        self.parser.add_argument('--feature_dimensionality',type=int,
                                 default=1, help='Dimension of the feature space, used in data.x')
        self.parser.add_argument('--pseudo_dimensionality', type=int,
                                 default=2, help='Dimension of the pseudo coordinates for GmmConv')
        self.parser.add_argument('--data_transform', type=str,
                                 default='Polar', help='define the edge attributes of the graphs e.g. Cartesian | Distance | LocalCartesian | Polar')

        self.parser.add_argument('--theta_max', type=float,
                                 default=0.2, help='nodes with lower euclidian distance will be connected')
        self.parser.add_argument('--theta', type=float,
                                 default=0.1, help='euclidian neighborhood distance')

        self.parser.add_argument('--dataset_path', type=str,
                                 default='../data/example1', help='the directory to read the Dataset from')
        self.parser.add_argument('--temp_dir', type=str,
                                 default='../temp', help='directory to save temporary outputs')
        self.parser.add_argument('--summary_dir', type=str,
                                 default='summary', help='relative directory to save temporary summary')
        self.parser.add_argument('--no_summary', action='store_true',
                                 default=False, help='if passed, tensorboardx will not be used to monitor the training')
        self.parser.add_argument('--validation_split', type=float,
                                 default=0.1, help='define size of validation set, 0 <= ... <= 1')
        self.parser.add_argument('--test_split', type=float,
                                 default=0.1, help='define size of test set, 0 <= ... <= 1')


        self.parser.add_argument('--model', type=str,
                                 default='GmmConvClassification', help='GcnRegression | GcnClassification | GmmConvClassification')
        self.parser.add_argument('--training_epochs', type=int,
                                 default=1000, help='number of training epochs')
        self.parser.add_argument('--hidden_layers', type=int,
                                 default=3, help='number of hidden layers in the n layer MoNet')
        self.parser.add_argument('--hidden_units', type=int,
                                 default=64, help='number of hidden units in the GNN')
        self.parser.add_argument('--samples', type=int,
                                 default=10, help='Number of random graphs to create, if a new dataset is created')
        self.parser.add_argument('--hidden_activation', type=str,
                                 default='sigmoid', help='Activation function from torch.nn.functional, used for hidden layers, e.g. relu | sigmoid | tanh')
        self.parser.add_argument('--batch_size_train', type=int,
                                 default='1', help='batch size for training')
        self.parser.add_argument('--batch_size_eval', type=int,
                                 default='1', help='batch size for evaluation')


    def parse_args(self):
        return self.parser.parse_args()
