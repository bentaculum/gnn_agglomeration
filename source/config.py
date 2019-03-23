import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--nodes', type=int,
                            default=100, help='Number of nodes in the graph')
        self.parser.add_argument('--theta_max', type=float,
                    default=0.2, help='nodes with lower euclidian distance will be connected')
        self.parser.add_argument('--theta', type=float,
                            default=0.1, help='euclidian neighborhood distance')
        self.parser.add_argument('--dataset_path', type=str,
                            default=None, help='the directory to read the Dataset from')

    def parse_args(self):
        return self.parser.parse_args()
