import torch
import os
import shutil
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

import sacred
from sacred import Experiment
import sys
from sacred.observers import MongoObserver, TelegramObserver
from sacred.stflow import LogFileWriter
import atexit
import tarfile
import argparse

from config import Config
from gcn_model import GcnModel
from gmm_conv_model import GmmConvModel
from spline_conv_model import SplineConvModel
from minimal_spline_conv_model import MinimalSplineConvModel
from gat_conv_model import GatConvModel
from our_conv_model import OurConvModel
from regression_problem import RegressionProblem
from result_plotting import ResultPlotting

from random_graph_dataset import RandomGraphDataset
from my_graph import MyGraph


ex = Experiment()


@ex.main
@ex.capture
@LogFileWriter(ex)
def main(_config, _run, _log):
    config = argparse.Namespace(**_config)
    _log.info('Logging to {}'.format(config.run_abs_path))

    # -----------------------------------------------
    # ---------------- CREATE SETUP -----------------
    # -----------------------------------------------

    # make necessary directory structure
    if not os.path.isdir(config.run_abs_path):
        os.makedirs(config.run_abs_path)

    # clear old stuff from the run dir, if it's not a restart
    if config.load_model is None:
        summary_dir = os.path.join(config.run_abs_path, config.summary_dir)
        if os.path.isdir(summary_dir):
            shutil.rmtree(summary_dir)
        model_dir = os.path.join(config.run_abs_path, config.model_dir)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)

        # make dir structure in temp dir
        os.makedirs(summary_dir)
        os.makedirs(model_dir)

    # Pass the path of tensorboardX summaries to sacred
    if not config.write_summary:
        _run.info["tensorflow"] = dict()
        _run.info["tensorflow"]["logdirs"] = [os.path.join(
            config.run_abs_path, config.summary_dir)]

    # set up the summary writer for tensorboardX
    train_writer = SummaryWriter(os.path.join(
        config.run_abs_path, 'summary', 'training'))
    val_writer = SummaryWriter(os.path.join(
        config.run_abs_path, 'summary', 'validation'))

    # create and load dataset
    dataset = RandomGraphDataset(root=config.dataset_abs_path, config=config)
    config.max_neighbors = dataset.max_neighbors()
    if config.standardize_targets:
        config.targets_mean, config.targets_std = dataset.targets_mean_std()
    # TODO if model is loaded, use the same train val test split.
    # shuffle can return the permutation of the dataset, which can then be used to permute the same way
    # dataset, perm = dataset.shuffle(return_perm=True)
    # when loading a model:
    # dataset = dataset.__indexing__(permutation)

    # TODO if model is loaded, use the same train val test split
    # split into train and test
    split_train_idx = int(
        config.samples * (1 - config.test_split - config.validation_split))
    split_validation_idx = int(config.samples * (1 - config.test_split))

    train_dataset = dataset[:split_train_idx]
    validation_dataset = dataset[split_train_idx:split_validation_idx]
    test_dataset = dataset[split_validation_idx:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader_train = DataLoader(
        train_dataset, batch_size=config.batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(
        validation_dataset, batch_size=config.batch_size_eval, shuffle=False)

    try:
        if config.load_model is None:
            model = globals()[config.model](
                config=config,
                train_writer=train_writer,
                val_writer=val_writer,
                model_type=config.model_type
            )
            model = model.to(device)
        else:
            # TODO allow to load previous models
            # find latest state of model
            load_model_dir = os.path.join(
                config.root_dir, config.run_abs_path, config.model_dir)
            checkpoint_versions = [name for name in os.listdir(
                load_model_dir) if name.endswith('.tar')]
            if 'final.tar' in checkpoint_versions:
                checkpoint_to_load = 'final.tar'
            else:
                checkpoint_versions = [
                    x for x in checkpoint_versions if x.startswith('epoch')].sort()
                checkpoint_to_load = checkpoint_versions[-1]

            _log.info('Loading checkpoint {} ...'.format(
                os.path.join(load_model_dir, checkpoint_to_load)))
            checkpoint = torch.load(os.path.join(
                load_model_dir, checkpoint_to_load))

            # restore the checkpoint
            model = globals()[config.model](
                config=config,
                train_writer=train_writer,
                val_writer=val_writer,
                epoch=checkpoint['epoch'],
                train_batch_iteration=checkpoint['train_batch_iteration'],
                val_batch_iteration=checkpoint['val_batch_iteration'],
                model_type=config.model_type
            )
            # model.to(device) has to be executed before loading the state dicts
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    except KeyError as e:
        print(e)
        raise NotImplementedError(
            'The model you have specified is not implemented')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _run.log_scalar('nr_params', total_params, config.training_epochs)

    @atexit.register
    def atexit_tasks():

        # -----------------------------------------------
        # ---------------- EVALUATION ROUTINE -----------
        # -----------------------------------------------

        # save the tensorboardx summary files
        summary_dir_exit = os.path.join(
            config.run_abs_path, config.summary_dir)
        summary_compressed = summary_dir_exit + '.tar.gz'
        # remove old tar file
        if os.path.isfile(summary_compressed):
            os.remove(summary_compressed)

        with tarfile.open(summary_compressed, mode='w:gz') as archive:
            archive.add(summary_dir_exit, arcname='summary', recursive=True)
        _run.add_artifact(filename=summary_compressed, name='summary.tar.gz')

        model.eval()
        model.current_writer = None

        # train loss
        final_loss_train = 0.0
        final_metric_train = 0.0
        for data in data_loader_train:
            data = data.to(device)
            out = model(data)
            final_loss_train += model.loss(out, data.y).item() * data.num_graphs
            final_metric_train += model.out_to_metric(
                out, data.y) * data.num_graphs
        final_loss_train /= train_dataset.__len__()
        final_metric_train /= train_dataset.__len__()

        _run.log_scalar('loss_train_final', final_loss_train, config.training_epochs)
        _run.log_scalar('accuracy_train_final', final_metric_train, config.training_epochs)

        # test loss
        data_loader_test = DataLoader(
            test_dataset, batch_size=config.batch_size_eval, shuffle=False)
        test_loss = 0.0
        test_metric = 0.0
        test_predictions = []
        test_targets = []

        for data in data_loader_test:
            data = data.to(device)
            out = model(data)
            test_loss += model.loss(out, data.y).item() * data.num_graphs
            test_metric += model.out_to_metric(out, data.y) * data.num_graphs
            pred = model.out_to_predictions(out)
            test_predictions.extend(model.predictions_to_list(pred))
            test_targets.extend(data.y.tolist())
        test_loss /= test_dataset.__len__()
        test_metric /= test_dataset.__len__()

        _run.log_scalar('loss_test', test_loss, config.training_epochs)
        _run.log_scalar('accuracy_test', test_metric, config.training_epochs)

        # final print routine
        print('')
        print('Maximum # of neighbors within distance {} in dataset: {}'.format(
            config.theta, config.max_neighbors))
        print('# of neighbors, distribution:')
        dic = dataset.neighbors_distribution()
        for key, value in sorted(dic.items(), key=lambda x: x[0]):
            print("{} : {}".format(key, value))
        print('')
        print('Total number of parameters: {}'.format(total_params))
        print('Mean train loss ({0} samples): {1:.3f}'.format(
            train_dataset.__len__(),
            final_loss_train))
        print('Mean accuracy on train set: {0:.3f}'.format(
            final_metric_train))
        print('Mean test loss ({0} samples): {1:.3f}'.format(
            test_dataset.__len__(),
            test_loss))
        print('Mean accuracy on test set: {0:.3f}'.format(
            test_metric))
        print('')

        # plot targets vs predictions. default is a confusion matrix
        model.plot_targets_vs_predictions(
            targets=test_targets, predictions=test_predictions)
        _run.add_artifact(
            filename=os.path.join(
                config.run_abs_path,
                config.confusion_matrix_path),
            name=config.confusion_matrix_path)

        # if Regression, plot targets vs. continuous outputs
        # if isinstance(model.model_type, RegressionProblem):
        #     test_outputs = []
        #     for data in data_loader_test:
        #         data = data.to(device)
        #         out = torch.squeeze(model(data)).tolist()
        #         test_outputs.extend(out)
        #     model.model_type.plot_targets_vs_outputs(
        #         targets=test_targets, outputs=test_outputs)

        # plot errors by location
        # plotter = ResultPlotting(config=config)
        # plotter.plot_errors_by_location(
        # data=test_dataset, predictions=test_predictions, targets=test_targets)

        # plot the graphs in the test dataset for visual inspection
        if config.plot_graphs_testset:
            for i, g in enumerate(test_dataset):
                graph = MyGraph(config, g)
                graph.plot_predictions(
                    pred=model.predictions_to_list(model.out_to_predictions(model(g))),
                    graph_nr=i)

        return '\ntrain acc: {0:.3f}\ntest acc: {1:.3f}'.format(
            final_metric_train, test_metric)

    # -----------------------------------------------
    # ---------------- TRAINING LOOP ----------------
    # -----------------------------------------------

    for epoch in range(model.epoch, config.training_epochs):
        # put model in training mode (e.g. use dropout)
        model.train()
        epoch_loss = 0.0
        epoch_metric_train = 0.0
        for batch_i, data in enumerate(data_loader_train):
            data = data.to(device)
            # call the forward method
            out = model(data)

            loss = model.loss(out, data.y)
            model.print_current_loss(epoch, batch_i)
            epoch_loss += loss.item() * data.num_graphs
            # TODO introduce weighting per node
            epoch_metric_train += model.out_to_metric(
                out, data.y) * data.num_graphs

            # clear the gradient variables of the model
            model.optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=config.clip_value)

            model.optimizer.step()
            model.train_batch_iteration += 1

        epoch_loss /= train_dataset.__len__()
        epoch_metric_train /= train_dataset.__len__()
        if not config.write_summary:
            train_writer.add_scalar('_per_epoch/loss', epoch_loss, epoch)
            train_writer.add_scalar(
                '_per_epoch/metric', epoch_metric_train, epoch)
        _run.log_scalar('loss_train', epoch_loss, epoch)
        _run.log_scalar('accuracy_train', epoch_metric_train, epoch)

        # validation
        model.eval()
        validation_loss = 0.0
        epoch_metric_val = 0.0
        for batch_i, data in enumerate(data_loader_validation):
            data = data.to(device)
            out = model(data)
            loss = model.loss(out, data.y)
            model.print_current_loss(epoch, 'validation {}'.format(batch_i))
            validation_loss += loss.item() * data.num_graphs
            epoch_metric_val += model.out_to_metric(
                out, data.y) * data.num_graphs
            model.val_batch_iteration += 1

        # The numbering of train and val does not correspond 1-to-1!
        # Here we skip some numbers for maintaining loose correspondence
        model.val_batch_iteration = model.train_batch_iteration

        validation_loss /= validation_dataset.__len__()
        epoch_metric_val /= validation_dataset.__len__()
        if not config.write_summary:
            val_writer.add_scalar('_per_epoch/loss', validation_loss, epoch)
            val_writer.add_scalar('_per_epoch/metric', epoch_metric_val, epoch)

        _run.log_scalar('loss_val', validation_loss, epoch)
        _run.log_scalar('accuracy_val', epoch_metric_val, epoch)

        model.epoch += 1

        # save intermediate models
        if model.epoch % config.checkpoint_interval == 0:
            model.save('epoch_{}'.format(model.epoch))

    # save the final model
    final_model_name = 'final'
    model.save(final_model_name)
    _run.add_artifact(
        filename=os.path.join(
            config.run_abs_path,
            config.model_dir,
            final_model_name + '.tar'),
        name=final_model_name)

    ###########################

    return atexit_tasks()


if __name__ == '__main__':
    config_dict, remaining_args = Config().parse_args()
    ex.add_config(config_dict)

    # sacred_default_flags = ['--enforce_clean', '-l', 'NOTSET']
    sacred_default_flags = []
    # remove all argparse arguments from sys.argv
    argv = [sys.argv[0], *sacred_default_flags, *remaining_args]

    ex.observers.append(
        MongoObserver.create(
            url=config_dict['mongo_url'],
            db_name=config_dict['mongo_db']
        )
    )

    if config_dict['telegram']:
        telegram_obs = TelegramObserver.from_config(
            os.path.join(config_dict['root_dir'], 'telegram.json'))
        ex.observers.append(telegram_obs)

    ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

    r = ex.run_commandline(argv)
    os._exit(0)