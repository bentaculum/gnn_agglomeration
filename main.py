import torch  # noqa
torch.multiprocessing.set_sharing_strategy('file_system')
# with cudnn enabled, validation with fixed params breaks
# cudnn speedups for PyG seem to be negligible anyway
torch.backends.cudnn.enabled = False

import sacred  # noqa
from sacred.observers import MongoObserver, TelegramObserver  # noqa
from sacred.stflow import LogFileWriter  # noqa
import logging  # noqa

import os  # noqa
import os.path as osp  # noqa
import shutil  # noqa
from torch_geometric.data import DataLoader  # noqa
from tensorboardX import SummaryWriter  # noqa

import sys  # noqa
import atexit  # noqa
import tarfile  # noqa
import argparse  # noqa
import json  # noqa
import time  # noqa
from time import time as now  # noqa
import numpy as np  # noqa
import datetime  # noqa
import pytz  # noqa
from funlib.segment.arrays import replace_values  # noqa

from gnn_agglomeration import utils  # noqa
from gnn_agglomeration.pyg_datasets import *  # noqa
from gnn_agglomeration.nn.models import *  # noqa


from gnn_agglomeration.experiment import ex  # noqa
from gnn_agglomeration.config import Config  # noqa

# TODO does this work together with sacred?
# Init logging module
logging.basicConfig(level=logging.INFO)


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
    summary_dir = os.path.join(config.run_abs_path, config.summary_dir)
    model_dir = os.path.join(config.run_abs_path, config.model_dir)
    outputs_dir = os.path.join(config.run_abs_path, config.outputs_dir)
    if not config.load_model:
        if os.path.isdir(summary_dir):
            shutil.rmtree(summary_dir)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        if os.path.isdir(outputs_dir):
            shutil.rmtree(outputs_dir)

        # make dir structure in temp dir
        os.makedirs(summary_dir)
        os.makedirs(model_dir)
        os.makedirs(os.path.join(outputs_dir, 'train'))
        os.makedirs(os.path.join(outputs_dir, 'val'))

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

    start_load_datasets = now()
    # create and load datasets
    if config.dataset_type_train.startswith('HemibrainDataset'):
        _log.info('Preparing training dataset ...')
        train_dataset = globals()[config.dataset_type_train](
            root=config.dataset_abs_path_train,
            config=config,
            db_name=config.db_name_train,
            embeddings_collection=config.embeddings_collection_train,
            roi_offset=config.train_roi_offset,
            roi_shape=config.train_roi_shape,
            length=config.samples,
            save_processed=config.save_processed_train
        )

        _log.info('Preparing validation dataset ...')
        validation_dataset = globals()[config.dataset_type_val](
            root=config.dataset_abs_path_val,
            config=config,
            db_name=config.db_name_val,
            embeddings_collection=config.embeddings_collection_val,
            roi_offset=config.val_roi_offset,
            roi_shape=config.val_roi_shape,
            save_processed=config.save_processed_val
        )
        if config.final_test_pass:
            _log.info('Preparing test dataset ...')
            test_dataset = globals()[config.dataset_type_test](
                root=config.dataset_abs_path_test,
                config=config,
                db_name=config.db_name_test,
                embeddings_collection=config.embeddings_collection_test,
                roi_offset=config.test_roi_offset,
                roi_shape=config.test_roi_shape,
                save_processed=config.save_processed_test
            )

    else:
        dataset = globals()[config.dataset_type_train](
            root=config.dataset_abs_path_train, config=config)
        # split into train and test
        split_train_idx = int(
            config.samples * (1 - config.test_split - config.validation_split))
        split_validation_idx = int(config.samples * (1 - config.test_split))

        train_dataset = dataset[:split_train_idx]
        validation_dataset = dataset[split_train_idx:split_validation_idx]
        test_dataset = dataset[split_validation_idx:]

        # new feature: if model is loaded, use the same train val test split.
        # shuffle can return the permutation of the dataset, which can then be used to permute the same way
        # dataset, perm = dataset.shuffle(return_perm=True)
        # when loading a model:
        # dataset = dataset.__indexing__(permutation)

    train_dataset.update_config(config)
    assert train_dataset.__getitem__(0).edge_attr.size(
        1) == config.pseudo_dimensionality

    if config.standardize_targets and config.model_type == 'RegressionProblem':
        config.targets_mean, config.targets_std = train_dataset.targets_mean_std()

    _log.info(f'Datasets ready in {now() - start_load_datasets} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.debug(f'num of gpus available: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        _log.info(f'current device: {torch.cuda.current_device()}')
    else:
        _log.info(f'current device: cpu')

    data_sampler_train = torch.utils.data.RandomSampler(
        data_source=train_dataset,
        replacement=True,
        num_samples=config.epoch_samples_train
    )
    data_sampler_val = torch.utils.data.RandomSampler(
        data_source=validation_dataset,
        replacement=True,
        num_samples=config.epoch_samples_val
    )

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=False,
        sampler=data_sampler_train,
        num_workers=config.num_workers,
        pin_memory=config.dataloader_pin_memory,
        worker_init_fn=lambda idx: np.random.seed()
    )
    data_loader_validation = DataLoader(
        validation_dataset,
        batch_size=config.batch_size_eval,
        shuffle=False,
        sampler=data_sampler_val,
        num_workers=config.num_workers,
        worker_init_fn=lambda idx: np.random.seed()
    )

    start_load_model = now()
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
        # find latest state of model
        _log.info(f'root dir {config.root_dir}')
        _log.info(f'run_abs_path {config.run_abs_path}')
        _log.info(f'model_dir {config.model_dir}')
        load_model_dir = os.path.join(
            config.root_dir, config.run_abs_path, config.model_dir)
        checkpoint_versions = [name for name in os.listdir(
            load_model_dir) if name.endswith('.tar')]
        if config.load_model_version == 'latest':
            if 'final.tar' in checkpoint_versions:
                checkpoint_to_load = 'final.tar'
            else:
                checkpoint_versions = sorted([
                    x for x in checkpoint_versions if x.startswith('epoch')])
                checkpoint_to_load = checkpoint_versions[-1]
        else:
            checkpoint_to_load = f'{config.load_model_version}.tar'

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
    _log.info(f'nr params: {total_params}')
    _run.log_scalar('nr_params', total_params, config.training_epochs)
    _log.info(f'Model ready in {now() - start_load_model} s')
    utils.log_max_memory_allocated(device)

    # save config to file and store in DB
    config_filepath = os.path.join(config.run_abs_path, 'config.json')
    with open(config_filepath, 'w') as f:
        json.dump(vars(config), f)
    _run.add_artifact(filename=config_filepath)

    def atexit_tasks(model):

        # -----------------------------------------------
        # ---------------- EVALUATION ROUTINE -----------
        # -----------------------------------------------

        # _log.info('saving tensorboardx summary files ...')
        # # save the tensorboardx summary files
        # summary_dir_exit = os.path.join(
        #     config.run_abs_path, config.summary_dir)
        # summary_compressed = summary_dir_exit + '.tar.gz'
        # # remove old tar file
        # if os.path.isfile(summary_compressed):
        #     os.remove(summary_compressed)
        #
        # with tarfile.open(summary_compressed, mode='w:gz') as archive:
        #     archive.add(summary_dir_exit, arcname='summary', recursive=True)
        # # _run.add_artifact(filename=summary_compressed, name='summary.tar.gz')

        model.eval()
        model.current_writer = None

        # final print routine
        train_dataset.print_summary()

        _log.info(
            f'Total number of parameters: {total_params}')

        if config.final_training_pass:
            # TODO seems to be buggy at the moment
            # train loss
            final_loss_train = 0.0
            final_metric_train = 0.0
            final_nr_nodes_train = 0

            _log.info('final training pass ...')
            start = time.time()
            for data_ft in data_loader_train:
                data_ft = data_ft.to(device)
                out_ft = model(data_ft)
                final_loss_train += model.loss(out_ft,
                                               data_ft.y,
                                               data_ft.mask).item() * data_ft.mask.sum().item()
                final_metric_train += model.out_to_metric(
                    out_ft, data_ft.y, data_ft.mask) * data_ft.mask.sum().item()
                final_nr_nodes_train += data_ft.mask.sum().item()
                utils.log_max_memory_allocated(device)
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
            _log.info(f'final training pass in {time.time() - start:.3f}s')
        else:
            # report training loss of last epoch
            try:
                final_loss_train = epoch_loss
                final_metric_train = epoch_metric_train
            except NameError as e:
                _log.warning(e)
                final_loss_train = 0.0
                final_metric_train = 0.0

        _log.info(
            f'Mean train loss ({train_dataset.__len__()} samples): {final_loss_train:.3f}')
        _log.info(
            f'Mean accuracy on train set: {final_metric_train:.3f}')

        if config.final_test_pass:

            # test loss
            data_loader_test = DataLoader(
                test_dataset,
                batch_size=config.batch_size_eval,
                shuffle=False,
                num_workers=config.num_workers,
                worker_init_fn=lambda idx: np.random.seed()
            )
            test_loss = 0.0
            test_metric = 0.0
            edge_weights_test = 0
            test_predictions = []
            test_targets = []

            test_1d_outputs = dict()
            test_embeddings = dict()

            paired_embeddings = {}

            _log.info('test pass ...')
            start_test_pass = time.time()
            for i, data_fe in enumerate(data_loader_test):
                _log.info(
                    f'batch {i}: num nodes {data_fe.num_nodes}, num edges {data_fe.num_edges}')
                data_fe = data_fe.to(device)
                out_fe = model(data_fe)
                utils.log_max_memory_allocated(device)

                if config.our_conv_output_node_embeddings:
                    nodes_mask = data_fe.nodes_mask.cpu().numpy().astype(np.bool)
                    _log.info(
                        f'adding embeddings for {np.sum(nodes_mask)} nodes')

                    embeddings = out_fe.cpu().numpy()[nodes_mask]
                    ids = data_fe.node_ids.cpu().numpy()[nodes_mask]
                    for k, v in zip(ids, embeddings):
                        if k not in test_embeddings:
                            test_embeddings[k] = v
                        else:
                            _log.warning(
                                f'embedding for node {k} already exists')
                    continue

                if config.write_to_db:
                    start = time.time()
                    out_1d = model.out_to_one_dim(out_fe)
                    # TODO this assumes again that every pairs of directed edges are next to each other
                    # and we grab the original representation (u,v) from the DB? Does not seem to work
                    edges = torch.transpose(data_fe.edge_index, 0, 1)[0::2]

                    # mask outputs
                    edges = edges[data_fe.roi_mask.byte()].cpu(
                    ).numpy().astype(np.int64)
                    out_1d = out_1d[data_fe.roi_mask.byte()].cpu().numpy()

                    out0 = out_fe[0][data_fe.roi_mask.byte()].cpu().numpy()
                    out1 = out_fe[1][data_fe.roi_mask.byte()].cpu().numpy()
                    labels = data_fe.y.cpu().numpy()

                    if len(edges) == 0:
                        _log.warning(
                            f'test pass: no edges in block after masking')
                        continue

                    edges_orig_labels = np.zeros_like(edges, dtype=np.int64)
                    edges_orig_labels = replace_values(
                        in_array=edges,
                        out_array=edges_orig_labels,
                        old_values=np.arange(
                            data_fe.num_nodes, dtype=np.int64),
                        new_values=data_fe.node_ids.cpu().numpy().astype(np.int64),
                        inplace=False
                    )

                    edges_list = [tuple(i)
                                  for i in edges_orig_labels]

                    for k, v, o1, o2, la in zip(edges_list, out_1d, out0, out1, labels):
                        # TODO this is super hacky, only applies for RAG
                        # remove artificial self-loops:
                        if k[0] == k[1]:
                            continue

                        paired_embeddings[k] = (o1, o2, la)

                        if k not in test_1d_outputs:
                            test_1d_outputs[k] = v
                        else:
                            # TODO adapt strategy here if desired
                            if config.graph_type == 'HemibrainGraphMasked':
                                test_1d_outputs[k] = max(test_1d_outputs[k], v)
                                _log.warning(
                                    'Masking should lead to a single prediction per edge in blockwise dataset, unless a block is doubled because another one is empty')
                                _log.warning(
                                    f'Edge {k} with value {test_1d_outputs[k]} already exists, new value would be {v}')
                            else:
                                test_1d_outputs[k] = max(test_1d_outputs[k], v)

                    _log.debug(
                        f'writing outputs to dict in {time.time() - start}s')

                test_loss += model.loss(out_fe, data_fe.y,
                                        data_fe.mask).item() * data_fe.mask.sum().item()
                test_metric += model.out_to_metric(out_fe,
                                                   data_fe.y, data_fe.mask) * data_fe.mask.sum().item()
                edge_weights_test += data_fe.mask.sum().item()
                pred = model.out_to_predictions(out_fe)
                test_predictions.extend(model.predictions_to_list(pred))
                test_targets.extend(data_fe.y.tolist())

            # save pairwise embeddings plus labels
            emb_pair_path = osp.join(config.run_abs_path, 'outputs.npz')
            _log.info(f'save outputs and labels to {emb_pair_path}')

            out0 = np.array([i[0] for i in list(paired_embeddings.values())])
            out1 = np.array([i[1] for i in list(paired_embeddings.values())])
            labels = np.array([i[2] for i in list(paired_embeddings.values())])
            np.savez(
                emb_pair_path,
                out0=out0,
                out1=out1,
                labels=labels
            )
            # stop here
            return

            if config.our_conv_output_node_embeddings:
                # save embeddings to file
                emb_path = osp.join(config.run_abs_path, 'embeddings.npz')
                _log.info(f'save embeddings to {emb_path}')
                np.savez(
                    emb_path,
                    node_ids=np.array(
                        list(test_embeddings.keys()), dtype=np.int64),
                    embeddings=np.array(
                        list(test_embeddings.values()), dtype=np.float32)
                )
                return

            test_loss /= edge_weights_test
            test_metric /= edge_weights_test

            _run.log_scalar('loss_test', test_loss, config.training_epochs)
            _run.log_scalar('accuracy_test', test_metric,
                            config.training_epochs)
            _log.info(f'test pass in {time.time() - start_test_pass:.3f}s\n')

            _log.info(
                f'Mean test loss ({test_dataset.__len__()} samples): {test_loss:.3f}')
            _log.info(
                f'Mean accuracy on test set: {test_metric:.3f}\n')

            if config.write_to_db:
                comment = _run.meta_info['options']['--comment']
                timestamp = str(_run.start_time).replace(' ', 'T')
                test_dataset.write_outputs_to_db(
                    outputs_dict=test_1d_outputs,
                    collection_name=f'{timestamp}_{comment}',
                )

            if config.plot_targets_vs_predictions:
                # TODO fix to run on cluster
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
                            g.y, g.mask),
                        logger=_log)
        else:
            try:
                # report validation loss of last epoch
                test_loss = validation_loss
                test_metric = epoch_metric_val
            except NameError as e:
                _log.warning(e)
                test_loss = 0.0
                test_metric = 0.0
            _log.info(
                f'Mean validation loss ({validation_dataset.__len__()} samples): {test_loss:.3f}')
            _log.info(
                f'Mean accuracy on validation set: {test_metric:.3f}\n')

        return '\n{0}\ntrain acc: {1:.3f}\ntest acc: {2:.3f}'.format(
            _run.meta_info['options']['--comment'], final_metric_train, test_metric)

    atexit.register(atexit_tasks, model=model)

    # -----------------------------------------------
    # ---------------- TRAINING LOOP ----------------
    # -----------------------------------------------

    # no training if we simply want to produce node embeddings
    if config.our_conv_output_node_embeddings:
        atexit.unregister(atexit_tasks)
        return atexit_tasks(model=model)

    for epoch in range(model.epoch, config.training_epochs):
        start_epoch_train = time.time()

        # put model in training mode (e.g. use dropout)
        model.train()
        epoch_loss = 0.0
        epoch_metric_train = 0.0
        edge_weights_train = 0
        _log.info('epoch {} ...'.format(epoch))
        for batch_i, data in enumerate(data_loader_train):
            start_batch = now()

            # mask is half as long as num edges, because it is not directed
            _log.info(
                f'batch {batch_i}: num nodes {data.num_nodes},'
                f'num edges in loss/total {int(2 * data.mask.sum().item())}/{data.num_edges}'
            )

            data = data.to(device)

            # call the forward method
            _log.debug('forward pass')
            out = model(data)

            loss = model.loss(out, data.y, data.mask)

            _log.debug('backward pass')
            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                if config.clip_method == 'value':
                    torch.nn.utils.clip_grad_value_(
                        parameters=filter(
                            lambda p: p.requires_grad, model.parameters()),
                        clip_value=config.clip_value
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=filter(
                            lambda p: p.requires_grad, model.parameters()),
                        max_norm=config.clip_value,
                        norm_type=float(config.clip_method)
                    )

            model.optimizer.step()
            # clear the gradient variables of the model
            model.optimizer.zero_grad()
            utils.log_max_memory_allocated(device)

            model.print_current_loss(epoch, batch_i, _log)

            epoch_loss += loss.item() * data.mask.sum().item()
            epoch_metric_train += model.out_to_metric(
                out, data.y, data.mask) * data.mask.sum().item()
            edge_weights_train += data.mask.sum().item()

            if batch_i % config.outputs_interval == 0:
                if isinstance(out, tuple):
                    # first dim: u,v second dim: num_edges, third dim = number of output node features
                    # store pairs of node embeddings
                    out = torch.stack([out[0], out[1]], dim=0)

                np.savez(
                    os.path.join(outputs_dir, 'train',
                                 f'epoch_{epoch}_batch_{batch_i}'),
                    out=out.detach().cpu().numpy(),
                    labels=data.y.detach().cpu().numpy(),
                    mask=data.mask.detach().cpu().numpy()
                )

            if config.summary_per_batch:
                train_writer.add_scalar(
                    '00/weighted_loss',
                    loss.item(),
                    epoch * data_loader_train.__len__() + batch_i
                )
                train_writer.add_scalar(
                    '00/weighted_accuracy',
                    model.out_to_metric(out, data.y, data.mask),
                    epoch * data_loader_train.__len__() + batch_i
                )

            model.train_batch_iteration += 1
            _log.debug(f'batch {batch_i} in {now() - start_batch} s')

        epoch_loss /= edge_weights_train
        epoch_metric_train /= edge_weights_train

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
        edge_weights_val = 0
        for batch_i, data in enumerate(data_loader_validation):
            data = data.to(device)
            out = model(data)
            loss = model.loss(out, data.y, data.mask)
            utils.log_max_memory_allocated(device)
            # model.print_current_loss(
            # epoch, 'validation {}'.format(batch_i), _log)
            validation_loss += loss.item() * data.mask.sum().item()
            epoch_metric_val += model.out_to_metric(
                out, data.y, data.mask) * data.mask.sum().item()
            edge_weights_val += data.mask.sum().item()

            if batch_i % config.outputs_interval == 0:
                if isinstance(out, tuple):
                    # first dim: u,v second dim: num_edges, third dim = number of output node features
                    # store pairs of node embeddings
                    out = torch.stack([out[0], out[1]], dim=0)

                np.savez(
                    os.path.join(outputs_dir, 'val',
                                 f'epoch_{epoch}_batch_{batch_i}'),
                    out=out.detach().cpu().numpy(),
                    labels=data.y.detach().cpu().numpy(),
                    mask=data.mask.detach().cpu().numpy()
                )

            if config.summary_per_batch:
                val_writer.add_scalar(
                    '00/weighted_loss',
                    loss.item(),
                    epoch * data_loader_train.__len__() + batch_i
                )
                val_writer.add_scalar(
                    '00/weighted_accuracy',
                    model.out_to_metric(out, data.y, data.mask),
                    epoch * data_loader_train.__len__() + batch_i
                )
                # for cosine embedding loss
                if isinstance(out, tuple):
                    utils.output_similarities_split(
                        writer=val_writer,
                        iteration=epoch * data_loader_train.__len__() + batch_i,
                        out0=out[0],
                        out1=out[1],
                        labels=data.y
                    )

            model.val_batch_iteration += 1

        # The numbering of train and val does not correspond 1-to-1!
        # Here we skip some numbers for maintaining loose correspondence
        model.val_batch_iteration = model.train_batch_iteration

        validation_loss /= edge_weights_val
        epoch_metric_val /= edge_weights_val

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

    # After training loop is over, the exit function is called directly
    atexit.unregister(atexit_tasks)
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
