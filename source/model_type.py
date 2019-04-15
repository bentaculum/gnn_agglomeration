import torch
from abc import ABC, abstractmethod
import os
import tensorboardX

class ModelType(torch.nn.Module, ABC):
    def __init__(self, config):
        super(ModelType, self).__init__()

        self.config = config

    @abstractmethod
    def loss(self, inputs, targets):
        pass

    @abstractmethod
    def out_to_predictions(self, out):
        pass

    @abstractmethod
    def predictions_to_list(self, predictions):
        pass

    @abstractmethod
    def metric(self, predictions, targets):
        pass
