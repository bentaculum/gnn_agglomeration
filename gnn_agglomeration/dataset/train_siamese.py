import logging
import torch
import numpy as np
import os
import os.path as osp
from time import time as now
import datetime
import pytz
import atexit
import configargparse


from node_embeddings.config_siamese import config as config_siamese  # noqa
from config import config  # noqa

from node_embeddings.siamese_dataset import SiameseDataset  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save(model, optimizer, model_dir, iteration):
    """
    TODO
    Should only be called after the end of a training+validation epoch
    Args:
        model:
        optimizer:
        model_dir:
        iteration:

    Returns:

    """
    # delete older models
    checkpoint_versions = [name for name in os.listdir(model_dir) if (
            name.endswith('.tar') and name.startswith('iteration'))]
    if len(checkpoint_versions) >= 3:
        checkpoint_versions.sort()
        os.remove(os.path.join(model_dir, checkpoint_versions[0]))

    # save the new one
    torch.save({
        'epoch': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, osp.join(model_dir, f'iteration_{iteration}.tar'))

    # save config file
    parser_writing = configargparse.ArgParser()
    parser_writing.write_config_file(
        parsed_namespace=config_siamese,
        output_file_paths=[osp.join(model_dir, 'config.ini')],
        exit_after=False
    )


def atexit_tasks(writer, loss):
    # TODO check if that works
    writer.close()
    # summary_compressed = osp.join(
    #     config_siamese.runs_dir,
    #     timestamp,
    #     'summary',
    #     '.tar.gz'
    #
    # with tarfile.open(summary_compressed, mode='w:gz') as archive:
    #     archive.add(summary_dir_exit, arcname='summary', recursive=True)
    logger.info(f'save tensorboard summaries')

    # report current loss
    logger.info(f'training loss: {loss}')


def train():
    timestamp = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

    # make necessary logging directories
    model_dir = osp.join(config_siamese.runs_dir, timestamp, 'model')
    os.makedirs(model_dir)
    summary_dir = osp.join(config_siamese.runs_dir, timestamp, 'summary')
    os.makedirs(summary_dir)

    # TODO
    data_augmentation_transform = None

    start = now()
    dataset = SiameseDataset(
        length=config_siamese.samples,
        patch_size=config_siamese.patch_size,
        raw_channel=config_siamese.raw_channel,
        mask_channel=config.mask_channel,
        transform=data_augmentation_transform
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=config_siamese.num_workers,
        worker_init_fn=lambda idx: np.random.seed()
    )
    logger.info(f'init dataloader in {now() - start} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    start = now()
    model = SiameseVgg3d(
        input_size=config_siamese.patch_size,
        input_fmaps=int(config_siamese.raw_channel) + int(config_siamese.mask_channel),
        fmaps=config_siamese.fmaps,
        output_features=config_siamese.output_features,
        downsample_factors=config_siamese.downsample_factors
    )
    model.to(device)
    logger.info(f'init model in {now() - start} s')

    optimizer = torch.optim.Adam(
        params=model.parameters,
        lr=config_siamese.adam_lr,
        weight_decay=config_siamese.adam_weight_decay
    )

    loss_function = torch.nn.CosineEmbeddingLoss(
        margin=0.5,
        reduction='mean'
    )

    if config_siamese.write_summary:
        writer = torch.utils.tensorboard.SummaryWriter(
            logdir=osp.join(config_siamese.runs_dir, timestamp, 'summary')
        )

    # register exit routines, in case there is an interrupt, e.g. via keyboard
    loss = np.inf
    atexit.register(
        func=atexit_tasks,
        writer=writer,
        loss=loss
    )

    for i, data in enumerate(dataloader):
        input0, input1, labels = data
        input0 = input0.to(device)
        input1 = input1.to(device)
        labels = labels.to(device)

        # make sure the dimensionality is ok
        assert input0.dim() == 5, input0.shape
        assert labels.dim() == 1, labels.shape

        optimizer.zero_grad()
        out0, out1 = model(input0, input1)
        loss = loss_function(
            input1=out0,
            input2=out1,
            target=labels
        )

        # TODO move to subprocess for speed?
        if config_siamese.write_summary:
            if i % config_siamese.summary_interval == 0:
                writer.add_scalar(
                    tag='loss',
                    scalar_value=loss,
                    global_step=i
                )

        loss.backward()
        optimizer.step()

        # save model
        if i % config_siamese.checkpoint_interval == 0:
            logger.info('Saving model ...')
            model.save(
                model=model,
                optimizer=optimizer,
                model_dir=model_dir,
                iteration=i
            )


if __name__ == '__main__':
    train()
