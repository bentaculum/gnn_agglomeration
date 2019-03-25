import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--nodes', type=int,
                            default=100, help='Number of nodes in the graph')
        self.parser.add_argument('--dimensionality', type=int,
                                 default=2, help='Dimension of the Euclidian space')
        self.parser.add_argument('--theta_max', type=float,
                    default=0.2, help='nodes with lower euclidian distance will be connected')
        self.parser.add_argument('--theta', type=float,
                            default=0.1, help='euclidian neighborhood distance')
        self.parser.add_argument('--dataset_path', type=str,
                            default='../data/example1', help='the directory to read the Dataset from')
        self.parser.add_argument('--training_epochs', type=int,
                                 default=200, help='number of training epochs')
        self.parser.add_argument('--hidden_units', type=int,
                                 default=64, help='number of hidden units in the GNN')
        self.parser.add_argument('--temp_dir', type=str,
                                 default='../temp', help='directory to save temporary outputs')
        self.parser.add_argument('--model', type=str,
                                 default='GmmConvClassification', help='GcnRegression | GcnClassification | GmmConvClassification')
        self.parser.add_argument('--samples', type=int,
                                 default=10, help='Number of random graphs to create, if a new dataset is created')

    def parse_args(self):
        return self.parser.parse_args()
