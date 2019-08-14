import logging
import torch
from torch.utils import tensorboard
import numpy as np
import os
import os.path as osp
from time import time as now
import datetime
import pytz

from node_embeddings.config_siamese import config as config_siamese, p as parser_siamese  # noqa
from config import config  # noqa

from node_embeddings.siamese_dataset_inference import SiameseDatasetInference  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)


def create_embeddings():
    timestamp = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

    start = now()
    dataset = SiameseDatasetInference(
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
        pin_memory=True,
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

    total_params = sum(p.numel()
                       for p in model.parameters())
    logger.info(f"number of params: {total_params}")

    model.eval()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.debug(f'number of trainable params: {trainable_params}')
    logger.info(f'load model in {now() - start} s')

    # limited number of samples
    samples_count = 0
    if config_siamese.inference_samples == 'all':
        samples_limit = np.iinfo(np.int_).max
    else:
        samples_limit = int(config_siamese.inference_samples)

    # ----------- training loop -----------
    node_ids = []
    embeddings = []

    for i, data in enumerate(dataloader):
        if samples_count >= samples_limit:
            break

        logger.info(f'batch {i} ...')
        patches, node_ids_batch = data

        # make sure the dimensionality is ok
        assert patches.dim() == 5, patches.shape

        patches = patches.to(device)

        out = model.forward_once(patches)

        node_ids.extend(node_ids_batch)
        embeddings.extend(list(out.numpy()))

        samples_count += config_siamese.batch_size

    dataset.write_embeddings_to_db(
        node_ids=node_ids,
        embeddings=embeddings,
        collection_name=f'nodes_embeddings_{timestamp}_{config_siamese.comment}',
    )


if __name__ == '__main__':
    create_embeddings()
