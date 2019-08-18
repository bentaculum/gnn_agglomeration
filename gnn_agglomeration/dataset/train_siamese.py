import logging
import torch
from torch.utils import tensorboard
import numpy as np
import os
import os.path as osp
from time import time as now
import datetime
import pytz
import atexit
import tarfile
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

from node_embeddings.config_siamese import config as config_siamese, p as parser_siamese  # noqa
from config import config, p as parser_ds  # noqa

from node_embeddings.siamese_dataset_train import SiameseDatasetTrain  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa
from node_embeddings import utils  # noqa


def save(model, optimizer, model_dir, iteration):
    """
    Should only be called after the end of a training+validation epoch
    Args:
        model:
        optimizer:
        model_dir:
        iteration:

    Returns:

    """
    # find latest state of the model
    def extract_number(f):
        s = re.findall(r'\d+', f)
        return int(s[0]) if s else -1, f

    # delete older models
    checkpoint_versions = [name for name in os.listdir(model_dir) if (
        name.endswith('.tar') and name.startswith('iteration'))]
    if len(checkpoint_versions) >= 3:
        checkpoint_to_remove = min(checkpoint_versions, key=extract_number)
        os.remove(osp.join(model_dir, checkpoint_to_remove))

    # save the new one
    model_tar = osp.join(model_dir, f'iteration_{iteration}.tar')
    logger.info(f'saving model to {model_tar} ...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_tar)

    # save config files
    parser_ds.write_config_file(
        parsed_namespace=config,
        output_file_paths=[osp.join(model_dir, 'config.ini')],
        exit_after=False
    )
    parser_siamese.write_config_file(
        parsed_namespace=config_siamese,
        # TODO write is reported three times, but seems to work fine
        output_file_paths=[osp.join(model_dir, 'config_siamese.ini')],
        exit_after=False
    )


def atexit_tasks(loss, writer, summary_dir):
    if writer:
        writer.close()
    # not needed
    with tarfile.open(f'{summary_dir}.tar.gz', mode='w:gz') as archive:
        archive.add(summary_dir, arcname='summary', recursive=True)
    logger.info(f'save tensorboard summaries')

    # report current loss
    # logger.info(f'training loss last iteration: {loss}')


def write_variable_to_summary(writer, iteration, var, namespace, var_name):
    """
    TODO
    Write summary statistics for a Tensor (for tensorboardX visualization)
    """
    # plot gradients of weights
    grad = var.grad
    if grad is not None:
        grad_mean = torch.mean(grad)
        writer.add_scalar(
            osp.join(namespace, var_name, 'gradients_mean'),
            grad_mean,
            iteration)
        grad_stddev = torch.std(grad)
        writer.add_scalar(
            osp.join(namespace, var_name, 'gradients_stddev'),
            grad_stddev,
            iteration)

        writer.add_histogram(
            osp.join(namespace, var_name, 'gradients'), grad, iteration)

    mean = torch.mean(var.data)
    writer.add_scalar(
        osp.join(namespace, var_name, 'mean'),
        mean,
        iteration)
    stddev = torch.std(var.data)
    writer.add_scalar(
        osp.join(namespace, var_name, 'stddev'),
        stddev,
        iteration)
    writer.add_histogram(
        osp.join(namespace, var_name),
        var.data,
        iteration)


def write_network_to_summary(writer, iteration, model):
    # write to tensorboard
    for num_module, module in enumerate(model.features):
        try:
            write_variable_to_summary(
                writer=writer,
                iteration=str(iteration),
                var=module.weight,
                namespace=f'{num_module}_{module._get_name()}',
                var_name='weight'
            )
            write_variable_to_summary(
                writer=writer,
                iteration=str(iteration),
                var=module.bias,
                namespace=f'{num_module}_{module._get_name()}',
                var_name='bias'
            )
        except AttributeError:
            pass

    num_pre_fc_modules = len(model.features)
    for num_module, module in enumerate(model.fully_connected):
        try:
            write_variable_to_summary(
                writer=writer,
                iteration=str(iteration),
                var=module.weight,
                namespace=f'{num_pre_fc_modules + num_module}_{module._get_name()}',
                var_name='weight'
            )
            write_variable_to_summary(
                writer=writer,
                iteration=str(iteration),
                var=module.bias,
                namespace=f'{num_pre_fc_modules + num_module}_{module._get_name()}',
                var_name='bias'
            )
        except AttributeError:
            pass


def run_validation(model, loss_function, dataloader, device, writer, train_iteration):
    start = now()
    model.eval()
    for i, data in enumerate(dataloader):
        input0, input1, labels = data

        input0 = input0.to(device)
        input1 = input1.to(device)
        labels = labels.to(device)

        out0, out1 = model(input0, input1)
        loss = loss_function(
            input1=out0,
            input2=out1,
            target=labels
        )

        writer.add_scalar(
            tag='00_loss',
            scalar_value=loss,
            global_step=train_iteration + i
        )

    model.train()
    print(f'run validation in {now() - start} s', end='\r')


def train():
    logger.info('start training function')
    timestamp = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

    # make necessary logging directories
    model_dir = osp.join(config_siamese.runs_dir, timestamp, 'model')
    os.makedirs(model_dir)
    summary_dir = osp.join(config_siamese.runs_dir, timestamp, 'summary')
    os.makedirs(summary_dir)

    start = now()
    dataset = SiameseDatasetTrain(
        patch_size=config_siamese.patch_size,
        raw_channel=config_siamese.raw_channel,
        mask_channel=config_siamese.mask_channel,
        raw_mask_channel=config_siamese.raw_mask_channel,
        num_workers=config.num_workers,
        in_memory=config_siamese.in_memory,
        rag_block_size=config_siamese.rag_block_size
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()
    if config_siamese.use_validation:
        split_size = int(len(dataset.samples_weights) * 0.2)
        zeros = torch.zeros(split_size)
        ones = torch.ones(len(dataset.samples_weights) - split_size)

        train_val_indices = torch.cat((zeros, ones))
        train_val_indices = train_val_indices[torch.randperm(
            len(train_val_indices))]

        samples_weights_train = train_val_indices.float() * dataset.samples_weights
        samples_weights_val = (~(train_val_indices.byte())
                               ).float() * dataset.samples_weights

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weights_train,
            num_samples=config_siamese.training_samples,
            replacement=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler,
            batch_size=config_siamese.batch_size,
            num_workers=config_siamese.num_workers_dataloader,
            pin_memory=True,
            worker_init_fn=lambda idx: np.random.seed()
        )

        sampler_val = torch.utils.data.WeightedRandomSampler(
            weights=samples_weights_val,
            num_samples=config_siamese.validation_samples,
            replacement=True
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler_val,
            batch_size=config_siamese.batch_size,
            num_workers=config_siamese.num_workers_dataloader,
            pin_memory=True,
            worker_init_fn=lambda idx: np.random.seed()
        )

    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=dataset.samples_weights,
            num_samples=config_siamese.training_samples,
            replacement=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler,
            batch_size=config_siamese.batch_size,
            num_workers=config_siamese.num_workers_dataloader,
            pin_memory=True,
            worker_init_fn=lambda idx: np.random.seed()
        )
    logger.info(f'init dataloader in {now() - start} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    start = now()

    # init tensorboard summary writer
    if config_siamese.summary_loss or config_siamese.summary_detailed:
        if config_siamese.use_validation:
            writer_val = torch.utils.tensorboard.SummaryWriter(
                log_dir=osp.join(config_siamese.runs_dir,
                                 timestamp, 'summary', 'val')
            )
            writer = torch.utils.tensorboard.SummaryWriter(
                log_dir=osp.join(config_siamese.runs_dir,
                                 timestamp, 'summary', 'train')
            )
        else:
            writer = torch.utils.tensorboard.SummaryWriter(
                log_dir=osp.join(config_siamese.runs_dir, timestamp, 'summary')
            )
    else:
        writer = None

    model = SiameseVgg3d(
        writer=writer,
        input_size=np.array(config_siamese.patch_size) /
        np.array(config.voxel_size),
        input_fmaps=int(config_siamese.raw_channel) +  # noqa
            int(config_siamese.mask_channel) +  # noqa
            int(config_siamese.raw_mask_channel),  # noqa
        fmaps=config_siamese.fmaps,
        fmaps_max=config_siamese.fmaps_max,
        output_features=config_siamese.output_features,
        downsample_factors=config_siamese.downsample_factors
    )
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of diff'able params: {total_params}")
    model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config_siamese.adam_lr,
        weight_decay=config_siamese.adam_weight_decay
    )

    if config_siamese.load_model is not None:
        checkpoint = utils.load_checkpoint(
            load_model=config_siamese.load_model,
            load_model_version=config_siamese.load_model_version,
            runs_dir=config_siamese.runs_dir
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f'init model in {now() - start} s')

    loss_function = torch.nn.CosineEmbeddingLoss(
        margin=config_siamese.cosine_loss_margin,
        reduction='mean'
    )

    logger.info(
        f'start training loop for {config_siamese.training_samples} samples')
    start_training = now()
    model.train()
    for i, data in enumerate(dataloader):
        # if i % config_siamese.console_update_interval == 1:
        #     start_console_update = now()
        start_batch = now()
        logger.debug(f'batch {i} ...')

        input0, input1, labels = data
        unique_labels = labels.unique()
        counts = [len(np.where(labels.numpy() == l.item())[0])
                  for l in unique_labels]
        for l, c in zip(unique_labels.tolist(), counts):
            logger.debug(f'# class {int(l)}: {int(c)}')

        # make sure the dimensionality is ok
        assert input0.dim() == 5, input0.shape
        assert labels.dim() == 1, labels.shape

        input0 = input0.to(device)
        input1 = input1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out0, out1 = model(input0, input1)
        loss = loss_function(
            input1=out0,
            input2=out1,
            target=labels
        )

        # TODO not sure if that's the intended use
        # somehow, the exit function gets called multiple times in some instances
        # register exit routines, in case there is an interrupt, e.g. via keyboard
        atexit.unregister(atexit_tasks)
        atexit.register(
            atexit_tasks,
            loss=loss,
            writer=writer,
            summary_dir=summary_dir
        )

        # TODO move to subprocess for speed?
        if config_siamese.summary_loss:
            start = now()
            if i % config_siamese.summary_interval == 0:
                writer.add_scalar(
                    tag='00_loss',
                    scalar_value=loss,
                    global_step=i
                )
            logger.debug(f'write to summary in {now() - start}')

        loss.backward()
        optimizer.step()

        if config_siamese.summary_detailed:
            write_network_to_summary(
                writer=writer,
                iteration=i,
                model=model)

        # if i % config_siamese.console_update_interval == 0 and i > 0:
        print(f'batch {i} done in {now() - start_batch} s', end='\r')
        # logging.info(f'batches {i} done in {now() - start_console_update} s')

        if config_siamese.use_validation:
            if i % config_siamese.validation_interval == 0 and i > 0:
                run_validation(
                    model=model,
                    loss_function=loss_function,
                    dataloader=dataloader_val,
                    device=device,
                    writer=writer_val,
                    train_iteration=i
                )

        # save model
        if i % config_siamese.checkpoint_interval == 0 and i > 0:
            start = now()
            save(
                model=model,
                optimizer=optimizer,
                model_dir=model_dir,
                iteration=i
            )
            logger.debug(f'save checkpoint in {now() - start}')

    logger.info(
        f'training {config_siamese.training_samples} samples took {now() - start_training} s')

    # parameters here are placeholders
    dataset.built_pipeline.__exit__(type=None, value=None, traceback=None)


if __name__ == '__main__':
    train()
