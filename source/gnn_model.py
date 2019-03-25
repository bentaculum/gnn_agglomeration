import torch
from abc import ABC, abstractmethod


class GnnModel(torch.nn.Module, ABC):
    def __init__(self, config):
        super(GnnModel, self).__init__()
        self.config = config
        self.layers()
        self.optimizer()

    @abstractmethod
    def layers(self):
        pass

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def loss(self, inputs, targets):
        pass

    def optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    @abstractmethod
    def evaluate_metric(self, data):
        pass

    @abstractmethod
    def evaluate_as_list(self, data):
        pass

    def print_current_loss(self, epoch):
        print('epoch {} {}: {} '.format(epoch, self.loss_name, self.current_loss))

    def evaluate(self, data, sample_no):
        self.eval()
        out = self.forward(data)
        _ = self.loss(out, data.y)
        print("test loss sample {}: {}".format(sample_no, self.current_loss))


