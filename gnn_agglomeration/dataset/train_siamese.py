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

from node_embeddings.config_siamese import config as config_siamese, p as parser_siamese  # noqa
from config import config  # noqa

from node_embeddings.siamese_dataset_train import SiameseDatasetTrain  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


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
    # delete older models
    checkpoint_versions = [name for name in os.listdir(model_dir) if (
        name.endswith('.tar') and name.startswith('iteration'))]
    if len(checkpoint_versions) >= 3:
        checkpoint_versions.sort()
        os.remove(os.path.join(model_dir, checkpoint_versions[0]))

    # save the new one
    model_tar = osp.join(model_dir, f'iteration_{iteration}.tar')
    logger.info(f'saving model to {model_tar} ...')
    torch.save({
        'epoch': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_tar)

    # save config file
    parser_siamese.write_config_file(
        parsed_namespace=config_siamese,
        # TODO write is reported three times, but seems to work fine
        output_file_paths=[osp.join(model_dir, 'config.ini')],
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
    logger.info(f'training loss: {loss}')


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
        num_workers=config.num_workers,
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=dataset.samples_weights,
        num_samples=dataset.__len__(),
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
    model = SiameseVgg3d(
        input_size=np.array(config_siamese.patch_size) /
        np.array(config.voxel_size),
        input_fmaps=int(config_siamese.raw_channel) +
        int(config_siamese.mask_channel),
        fmaps=config_siamese.fmaps,
        fmaps_max=config_siamese.fmaps_max,
        output_features=config_siamese.output_features,
        downsample_factors=config_siamese.downsample_factors
    )
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of diff'able params: {total_params}")

    model.to(device)
    logger.info(f'init model in {now() - start} s')

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config_siamese.adam_lr,
        weight_decay=config_siamese.adam_weight_decay
    )

    loss_function = torch.nn.CosineEmbeddingLoss(
        margin=0.5,
        reduction='mean'
    )

    if config_siamese.summary:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=osp.join(config_siamese.runs_dir, timestamp, 'summary')
        )
    else:
        writer = None

    logger.info('start training loop')
    samples_count = 0
    start_training = now()
    for i, data in enumerate(dataloader):
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
        # register exit routines, in case there is an interrupt, e.g. via keyboard
        atexit.unregister(atexit_tasks)
        atexit.register(
            atexit_tasks,
            loss=loss,
            writer=writer,
            summary_dir=summary_dir
        )

        # TODO move to subprocess for speed?
        if config_siamese.summary:
            start = now()
            if i % config_siamese.summary_interval == 0:
                writer.add_scalar(
                    tag='loss',
                    scalar_value=loss,
                    global_step=i
                )
            logger.debug(f'write to summary in {now() - start}')

        loss.backward()
        optimizer.step()

        logger.info(f'batch {i} done in {now() - start_batch} s')

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

        samples_count += config_siamese.batch_size
        if samples_count >= config_siamese.training_samples:
            break

    logger.info(
        f'training {samples_count} samples took {now() - start_training} s')
    dataset.built_pipeline.__exit__()


if __name__ == '__main__':
    train()
