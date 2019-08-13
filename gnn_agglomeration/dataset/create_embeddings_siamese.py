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

from node_embeddings.siamese_dataset import SiameseDataset  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


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
    parser_siamese.write_config_file(
        parsed_namespace=config_siamese,
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


def create_embeddings():
    timestamp = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

    start = now()
    dataset = SiameseDataset(
        patch_size=config_siamese.patch_size,
        raw_channel=config_siamese.raw_channel,
        mask_channel=config_siamese.mask_channel,
        num_workers=config.num_workers,
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=config_siamese.batch_size,
        num_workers=config_siamese.num_workers_dataloader,
        # TODO check whether pinning memory makes sense
        pin_memory=True,
        # TODO check whether this randomization really works for torch
        worker_init_fn=lambda idx: np.random.seed()
    )
    logger.info(f'init dataloader in {now() - start} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #########################

    start = now()

    if config_siamese['load_model'] == 'latest':
        # find latest model in the runs path
        # TODO filter for correct format of directory name, instead of
        # '2019'
        runs = sorted([name for name in os.listdir(
            config_siamese.runs_dir) if name.startswith('2019')])

        run_path = runs[-1]
    else:
        run_path = config_siamese['load_model']

    # find latest state of model
    load_model_dir = os.path.join(run_path, 'model')
    checkpoint_versions = [name for name in os.listdir(
        load_model_dir) if name.endswith('.tar')]
    if config_siamese.load_model_version == 'latest':
        if 'final.tar' in checkpoint_versions:
            checkpoint_to_load = 'final.tar'
        else:
            checkpoint_versions = sorted([
                x for x in checkpoint_versions if x.startswith('iteration')])
            checkpoint_to_load = checkpoint_versions[-1]
    else:
        checkpoint_to_load = f'config.load_model_version'

    logger.info(f'Load model {run_path}, checkpoint {checkpoint_to_load}')
    checkpoint = torch.load(os.path.join(
        load_model_dir, checkpoint_to_load))

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

    # model.to(device) has to be executed before loading the state
    # dicts
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'load model in {now() - start} s')

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of diff'able params: {total_params}")

    samples_count = 0
    if config_siamese.inference_samples == 'all':
        samples_limit = np.iinfo(np.int_).max
    else:
        samples_limit = int(config_siamese.inference_samples)

    for i, data in enumerate(dataloader):
        if samples_count >= samples_limit:
            break

        logger.info(f'batch {i} ...')
        input0, input1, _ = data

        # make sure the dimensionality is ok
        assert input0.dim() == 5, input0.shape
        assert labels.dim() == 1, labels.shape

        input0 = input0.to(device)
        input1 = input1.to(device)
        labels = labels.to(device)

        out0, out1 = model(input0, input1)

        # register exit routines, in case there is an interrupt, e.g. via keyboard
        # TODO write to database
        atexit.unregister(atexit_tasks)
        atexit.register(
            atexit_tasks,
            loss=loss,
            writer=writer,
            summary_dir=summary_dir
        )

        samples_count += config_siamese.batch_size

        # save model
        if i % config_siamese.checkpoint_interval == 0:
            logger.info('Saving model ...')
            save(
                model=model,
                optimizer=optimizer,
                model_dir=model_dir,
                iteration=i
            )


if __name__ == '__main__':
    create_embeddings()
