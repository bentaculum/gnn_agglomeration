import torch
from abc import ABC, abstractmethod
import os
import json

from .model_type import *


class GnnModel(torch.nn.Module, ABC):
    def __init__(self,
                 config,
                 train_writer,
                 val_writer,
                 epoch=0,
                 train_batch_iteration=0,
                 val_batch_iteration=0,
                 model_type='ClassificationProblem'):

        super(GnnModel, self).__init__()

        self.config = config

        try:
            self.model_type = globals()[model_type](config=self.config)
        except Exception as e:
            print(e)
            raise NotImplementedError(
                'The model type you have specified is not implemented')

        self.layers()
        self.init_optimizer()

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

    def loss(self, inputs, targets, mask):
        self.current_loss = self.model_type.loss(
            inputs=inputs, targets=targets, mask=mask)
        self.write_to_variable_summary(
            self.current_loss, 'out_layer', self.model_type.loss_name)
        return self.current_loss

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.adam_lr,
            weight_decay=self.config.adam_weight_decay)

    def out_to_predictions(self, out):
        return self.model_type.out_to_predictions(out=out)

    def out_to_one_dim(self, out):
        return self.model_type.out_to_predictions(out=out)

    def metric(self, predictions, targets):
        return self.model_type.metric(predictions=predictions, targets=targets)

    def evaluate_metric(self, data):
        out = self.forward(data)
        pred = self.out_to_predictions(out)
        return self.metric(pred, data.y)

    def out_to_metric(self, out, targets):
        pred = self.out_to_predictions(out)
        return self.metric(pred, targets)

    def predictions_to_list(self, predictions):
        return self.model_type.predictions_to_list(predictions=predictions)

    def plot_targets_vs_predictions(self, targets, predictions):
        self.model_type.plot_targets_vs_predictions(
            targets=targets, predictions=predictions)

    def print_current_loss(self, epoch, batch_i, logger):
        logger.debug('epoch {}, batch {}, {}: {} '.format(
            epoch, batch_i, self.model_type.loss_name, self.current_loss))

    def evaluate(self, data):
        out = self.forward(data)
        _ = self.loss(out, data.y, data.mask)
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

        if not self.config.write_summary or self.config.log_per_epoch_only:
            return

        # optional filter on namespaces
        if len(self.config.log_namespaces) > 0:
            if namespace not in self.config.log_namespaces:
                return

        # after the training loop, no more statistics should be recorded
        if self.current_writer is None:
            return

        if self.training is True:
            iteration = self.train_batch_iteration
        else:
            iteration = self.val_batch_iteration

        # plot gradients of weights
        grad = var.grad
        if grad is not None:
            grad_mean = torch.mean(grad)
            self.current_writer.add_scalar(os.path.join(
                namespace, var_name, 'gradients_mean'), grad_mean, iteration)
            grad_stddev = torch.std(grad)
            self.current_writer.add_scalar(
                os.path.join(
                    namespace,
                    var_name,
                    'gradients_stddev'),
                grad_stddev,
                iteration)

            if self.config.log_histograms:
                self.current_writer.add_histogram(os.path.join(
                    namespace, var_name, 'gradients'), grad, iteration)

        if self.config.log_only_gradients:
            return

        mean = torch.mean(var.data)
        self.current_writer.add_scalar(os.path.join(
            namespace, var_name, 'mean'), mean, iteration)
        stddev = torch.std(var.data)
        self.current_writer.add_scalar(os.path.join(
            namespace, var_name, 'stddev'), stddev, iteration)
        if self.config.log_histograms:
            self.current_writer.add_histogram(os.path.join(
                namespace, var_name), var.data, iteration)

    def save(self, name):
        """
        Should only be called after the end of a training+validation epoch
        """
        # delete older models
        load_model_dir = os.path.join(
            self.config.root_dir,
            self.config.run_abs_path,
            self.config.model_dir)
        checkpoint_versions = [name for name in os.listdir(load_model_dir) if (
            name.endswith('.tar') and name.startswith('epoch'))]
        if len(checkpoint_versions) >= 3:
            checkpoint_versions.sort()
            os.remove(os.path.join(load_model_dir, checkpoint_versions[0]))

        # save the new one
        torch.save({
            'epoch': self.epoch,
            'train_batch_iteration': self.train_batch_iteration,
            'val_batch_iteration': self.val_batch_iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.run_abs_path, self.config.model_dir, name + '.tar'))

        with open(os.path.join(self.config.run_abs_path, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)
