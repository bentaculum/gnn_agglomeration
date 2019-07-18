import torch
import os
import shutil
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

import sacred
from sacred.observers import MongoObserver, TelegramObserver
from sacred.stflow import LogFileWriter
import sys
import atexit
import tarfile
import argparse
import json
import time

from gnn_agglomeration.experiment import ex
from gnn_agglomeration.config import Config

# TODO parametrize imports to avoid overhead here
# TODO import the entire gnn_agglomeration package
from gnn_agglomeration.gcn_model import GcnModel
from gnn_agglomeration.gmm_conv_model import GmmConvModel
from gnn_agglomeration.spline_conv_model import SplineConvModel
from gnn_agglomeration.minimal_spline_conv_model import MinimalSplineConvModel
from gnn_agglomeration.gat_conv_model import GatConvModel
from gnn_agglomeration.our_conv_model import OurConvModel
from gnn_agglomeration.regression_problem import RegressionProblem
from gnn_agglomeration.result_plotting import ResultPlotting

from gnn_agglomeration.pyg_datasets.diameter_dataset import DiameterDataset
from gnn_agglomeration.pyg_datasets.count_neighbors_dataset import CountNeighborsDataset
from gnn_agglomeration.pyg_datasets.iterative_dataset import IterativeDataset
from gnn_agglomeration.pyg_datasets.hemibrain_dataset_random import HemibrainDatasetRandom
from gnn_agglomeration.pyg_datasets.hemibrain_dataset_blockwise import HemibrainDatasetBlockwise


@ex.main
@ex.capture
@LogFileWriter(ex)
def main(_config, _run, _log):
    # Check for a comment, if none is given raise error
    if _run.meta_info['options']['--comment'] is None:
        raise ValueError('You need to specify a comment with -c, --comment')

    config = argparse.Namespace(**_config)
    _log.info('Logging to {}'.format(config.run_abs_path))

    # -----------------------------------------------
    # ---------------- CREATE SETUP -----------------
    # -----------------------------------------------

    # make necessary directory structure
    if not os.path.isdir(config.run_abs_path):
        os.makedirs(config.run_abs_path)

    # clear old stuff from the run dir, if it's not a restart
    if not config.load_model:
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
    if config.write_summary:
        _run.info["tensorflow"] = dict()
        _run.info["tensorflow"]["logdirs"] = [os.path.join(
            config.run_abs_path, config.summary_dir)]

    # set up the summary writer for tensorboardX
    train_writer = SummaryWriter(os.path.join(
        config.run_abs_path, 'summary', 'training'))
    val_writer = SummaryWriter(os.path.join(
        config.run_abs_path, 'summary', 'validation'))

    # create and load datasets
    if config.dataset_type == 'HemibrainDataset':
        train_dataset = HemibrainDatasetRandom(
            root=config.dataset_abs_path_train,
            config=config,
            roi_offset=config.train_roi_offset,
            roi_shape=config.train_roi_shape,
            length=config.samples,
            save_processed=config.save_processed_train
        )

        validation_dataset = HemibrainDatasetBlockwise(
            root=config.dataset_abs_path_val,
            config=config,
            roi_offset=config.val_roi_offset,
            roi_shape=config.val_roi_shape,
            save_processed=config.save_processed_val
        )

        test_dataset = HemibrainDatasetBlockwise(
            root=config.dataset_abs_path_test,
            config=config,
            roi_offset=config.test_roi_offset,
            roi_shape=config.test_roi_shape,
            save_processed=config.save_processed_test
        )

    else:
        dataset = globals()[config.dataset_type](
            root=config.dataset_abs_path_train, config=config)
        # split into train and test
        split_train_idx = int(
            config.samples * (1 - config.test_split - config.validation_split))
        split_validation_idx = int(config.samples * (1 - config.test_split))

        train_dataset = dataset[:split_train_idx]
        validation_dataset = dataset[split_train_idx:split_validation_idx]
        test_dataset = dataset[split_validation_idx:]

        # TODO if model is loaded, use the same train val test split.
        # shuffle can return the permutation of the dataset, which can then be used to permute the same way
        # dataset, perm = dataset.shuffle(return_perm=True)
        # when loading a model:
        # dataset = dataset.__indexing__(permutation)

    train_dataset.update_config(config)
    assert train_dataset.__getitem__(0).edge_attr.size(
        1) == config.pseudo_dimensionality

    if config.standardize_targets and config.model_type == 'RegressionProblem':
        config.targets_mean, config.targets_std = train_dataset.targets_mean_std()

    _log.info('Datasets are ready')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.dataloader_pin_memory)
    data_loader_validation = DataLoader(
        validation_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=config.num_workers)

    if not config.load_model:
        model = globals()[config.model](
            config=config,
            train_writer=train_writer,
            val_writer=val_writer,
            model_type=config.model_type
        )
        model = model.to(device)
    else:
        _log.info('Loading model {} ...'.format(config.load_model))
        # TODO allow to load previous models
        # find latest state of model
        load_model_dir = os.path.join(
            config.root_dir, config.run_abs_path, config.model_dir)
        checkpoint_versions = [name for name in os.listdir(
            load_model_dir) if name.endswith('.tar')]
        if 'final.tar' in checkpoint_versions:
            checkpoint_to_load = 'final.tar'
        else:
            checkpoint_versions = sorted([
                x for x in checkpoint_versions if x.startswith('epoch')])
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
        # model.to(device) has to be executed before loading the state
        # dicts
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    _run.log_scalar('nr_params', total_params, config.training_epochs)
    _log.info('Model is ready')

    # save config to file and store in DB
    config_filepath = os.path.join(config.run_abs_path, 'config.json')
    with open(config_filepath, 'w') as f:
        json.dump(vars(config), f)
    _run.add_artifact(filename=config_filepath)

    def atexit_tasks(model):

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
        final_nr_nodes_train = 0
        for data_ft in data_loader_train:
            data_ft = data_ft.to(device)
            out_ft = model(data_ft)
            final_loss_train += model.loss(out_ft,
                                           data_ft.y,
                                           data_ft.mask).item() * data_ft.num_nodes
            final_metric_train += model.out_to_metric(
                out_ft, data_ft.y) * data_ft.num_nodes
            final_nr_nodes_train += data_ft.num_nodes
        final_loss_train /= final_nr_nodes_train
        final_metric_train /= final_nr_nodes_train

        _run.log_scalar(
            'loss_train_final',
            final_loss_train,
            config.training_epochs)
        _run.log_scalar(
            'accuracy_train_final',
            final_metric_train,
            config.training_epochs)

        # test loss
        data_loader_test = DataLoader(
            test_dataset,
            batch_size=config.batch_size_eval,
            shuffle=False,
            num_workers=config.num_workers)
        test_loss = 0.0
        test_metric = 0.0
        nr_nodes_test = 0
        test_predictions = []
        test_targets = []

        for data_fe in data_loader_test:
            data_fe = data_fe.to(device)
            out_fe = model(data_fe)
            test_loss += model.loss(out_fe, data_fe.y,
                                    data_fe.mask).item() * data_fe.num_nodes
            test_metric += model.out_to_metric(out_fe,
                                               data_fe.y) * data_fe.num_nodes
            nr_nodes_test += data_fe.num_nodes
            pred = model.out_to_predictions(out_fe)
            test_predictions.extend(model.predictions_to_list(pred))
            test_targets.extend(data_fe.y.tolist())
        test_loss /= nr_nodes_test
        test_metric /= nr_nodes_test

        _run.log_scalar('loss_test', test_loss, config.training_epochs)
        _run.log_scalar('accuracy_test', test_metric, config.training_epochs)

        # final print routine
        print('')
        train_dataset.print_summary()

        _log.info(
            f'Total number of parameters: {total_params}')
        _log.info(
            f'Mean train loss ({train_dataset.__len__()} samples): {final_loss_train:.3f}')
        _log.info(
            f'Mean accuracy on train set: {final_metric_train:.3f}')
        _log.info(
            f'Mean test loss ({test_dataset.__len__()} samples): {test_loss:.3f}')
        _log.info(
            f'Mean accuracy on test set: {test_metric:.3f}')

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
        # data=test_dataset, predictions=test_predictions,
        # targets=test_targets)

        # plot the graphs in the test dataset for visual inspection
        if config.plot_graphs_testset:
            if config.plot_graphs_testset < 0 or config.plot_graphs_testset > test_dataset.__len__():
                plot_limit = test_dataset.__len__()
            else:
                plot_limit = config.plot_graphs_testset

            for i in range(plot_limit):
                g = test_dataset[i]
                g.to(device)
                out_p = model(g)
                g.plot_predictions(
                    config=config,
                    pred=model.predictions_to_list(
                        model.out_to_predictions(out_p)),
                    graph_nr=i,
                    run=_run,
                    acc=model.out_to_metric(
                        out_p,
                        g.y),
                    logger=_log)

        return '\n{0}\ntrain acc: {1:.3f}\ntest acc: {2:.3f}'.format(
            _run.meta_info['options']['--comment'], final_metric_train, test_metric)

    atexit.register(atexit_tasks, model=model)

    # -----------------------------------------------
    # ---------------- TRAINING LOOP ----------------
    # -----------------------------------------------

    for epoch in range(model.epoch, config.training_epochs):
        start_epoch_train = time.time()

        # put model in training mode (e.g. use dropout)
        model.train()
        epoch_loss = 0.0
        epoch_metric_train = 0.0
        nr_nodes_train = 0
        _log.info('epoch {} ...'.format(epoch))
        for batch_i, data in enumerate(data_loader_train):
            data = data.to(device)
            # call the forward method
            out = model(data)

            loss = model.loss(out, data.y, data.mask)
            model.print_current_loss(epoch, batch_i, _log)
            _log.debug(f'total num nodes: {data.num_nodes}')

            epoch_loss += loss.item() * data.num_nodes
            epoch_metric_train += model.out_to_metric(
                out, data.y) * data.num_nodes
            nr_nodes_train += data.num_nodes

            # clear the gradient variables of the model
            model.optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                if config.clip_method == 'value':
                    torch.nn.utils.clip_grad_value_(
                        parameters=model.parameters(),
                        clip_value=config.clip_value
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=config.clip_value,
                        norm_type=float(config.clip_method)
                    )

            model.optimizer.step()
            model.train_batch_iteration += 1

        epoch_loss /= nr_nodes_train
        epoch_metric_train /= nr_nodes_train

        if config.write_summary:
            train_writer.add_scalar('_per_epoch/loss', epoch_loss, epoch)
            train_writer.add_scalar(
                '_per_epoch/metric', epoch_metric_train, epoch)
        _run.log_scalar('loss_train', epoch_loss, epoch)
        _run.log_scalar('accuracy_train', epoch_metric_train, epoch)

        _log.info(f'training in {time.time() - start_epoch_train:.3f} s')
        start_epoch_val = time.time()

        # validation
        model.eval()
        validation_loss = 0.0
        epoch_metric_val = 0.0
        nr_nodes_val = 0
        for batch_i, data in enumerate(data_loader_validation):
            data = data.to(device)
            out = model(data)
            loss = model.loss(out, data.y, data.mask)
            # model.print_current_loss(
            # epoch, 'validation {}'.format(batch_i), _log)
            validation_loss += loss.item() * data.num_nodes
            epoch_metric_val += model.out_to_metric(
                out, data.y) * data.num_nodes
            nr_nodes_val += data.num_nodes
            model.val_batch_iteration += 1

        # The numbering of train and val does not correspond 1-to-1!
        # Here we skip some numbers for maintaining loose correspondence
        model.val_batch_iteration = model.train_batch_iteration

        validation_loss /= nr_nodes_val
        epoch_metric_val /= nr_nodes_val

        if config.write_summary:
            val_writer.add_scalar('_per_epoch/loss', validation_loss, epoch)
            val_writer.add_scalar('_per_epoch/metric', epoch_metric_val, epoch)

        _run.log_scalar('loss_val', validation_loss, epoch)
        _run.log_scalar('accuracy_val', epoch_metric_val, epoch)
        _run.result = f'train acc: {epoch_metric_train:.3f}, val acc: {epoch_metric_val:.3f}'

        model.epoch += 1

        _log.info(f'validation in {time.time() - start_epoch_val:.3f} s')

        # save intermediate models
        if model.epoch % config.checkpoint_interval == 0:
            _log.info('saving model ...')
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

    return atexit_tasks(model=model)


if __name__ == '__main__':
    config_dict, remaining_args = Config().parse_args()
    ex.add_config(config_dict)

    # sacred_default_flags = ['--enforce_clean', '-l', 'INFO']
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
    # os._exit(0)
    sys.exit()
