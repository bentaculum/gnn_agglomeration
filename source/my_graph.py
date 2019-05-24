from abc import ABC, abstractmethod

from torch_geometric.data import Data


class MyGraph(Data, ABC):

    # can't overwrite __init__ using different args than base class

    @abstractmethod
    def create_random_graph(self, config):
        pass

    # TODO
    # @abstractmethod
    # def plot(self):
    #     pass

    @abstractmethod
    def plot_predictions(self, config, pred, graph_nr, run, acc, logger):
        pass
