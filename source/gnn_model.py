import torch
from abc import ABC, abstractmethod
import os
import tensorboardX

class GnnModel(torch.nn.Module, ABC):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0):

        super(GnnModel, self).__init__()

        self.config = config
        self.layers()
        self.optimizer()

        self.epoch = epoch
        self.train_writer = train_writer
        self.val_writer = val_writer
        self.current_writer = None

        self.train_batch_iteration = train_batch_iteration
        self.val_batch_iteration = val_batch_iteration

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
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.adam_lr,
            weight_decay=self.config.adam_weight_decay)

    @abstractmethod
    def out_to_predictions(self, out):
        pass

    @abstractmethod
    def metric(self, predictions, targets):
        pass

    def evaluate_metric(self, data):
        out = self.forward(data)
        pred = self.out_to_predictions(out)
        return self.metric(pred, data.y)

    def out_to_metric(self, out, targets):
        pred = self.out_to_predictions(out)
        return self.metric(pred, targets)

    @abstractmethod
    def predictions_to_list(self, predictions):
        pass

    def print_current_loss(self, epoch, batch_i):
        print('epoch {}, batch {}, {}: {} '.format(epoch, batch_i, self.loss_name, self.current_loss))

    def evaluate(self, data):
        out = self.forward(data)
        _ = self.loss(out, data.y)
        return self.current_loss

    def train(self, mode=True):
        ret = super(GnnModel, self).train(mode=mode)
        self.current_writer = self.train_writer
        return ret

    def eval(self):
        ret = super(GnnModel, self).eval()
        self.current_writer = self.val_writer
        return ret

    def write_to_variable_summary(self, var, namespace, var_name):
        """Write summary statistics for a Tensor (for tensorboardX visualization)"""
        if self.config.no_summary:
            return

        if self.current_writer is None:
            # after the training loop, no more statistics should be recorded
            return

        if self.training is True:
            iteration = self.train_batch_iteration
        else:
            iteration = self.val_batch_iteration

        mean = torch.mean(var.data)
        self.current_writer.add_scalar(os.path.join(namespace, var_name, 'mean'), mean, iteration)
        stddev = torch.std(var.data)
        self.current_writer.add_scalar(os.path.join(namespace, var_name, 'stddev'), stddev, iteration)
        # self.current_writer.add_scalar(os.path.join(namespace, var_name, 'max'), torch.max(var), iteration)
        # self.current_writer.add_scalar(os.path.join(namespace, var_name, 'min'), torch.min(var), iteration)
        self.current_writer.add_histogram(os.path.join(namespace, var_name), var.data, iteration)

        #plot gradients of weights
        grad = var.grad
        if grad is not None:
            self.current_writer.add_histogram(os.path.join(namespace, var_name, 'gradients'), grad, iteration)

    def save(self, name):
        """
        Should only be called after the end of a training+validation epoch
        """

        torch.save({
            'epoch': self.epoch,
            'train_batch_iteration': self.train_batch_iteration,
            'val_batch_iteration': self.val_batch_iteration,
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.temp_dir, self.config.model_dir, name))
