import logging
import torch
from torch.utils import tensorboard
import numpy as np
from time import time as now
import datetime
import pytz

from node_embeddings.config_siamese import config as config_siamese, p as parser_siamese  # noqa
from config import config  # noqa

from node_embeddings.siamese_dataset_inference import SiameseDatasetInference  # noqa
from node_embeddings.siamese_vgg_3d import SiameseVgg3d  # noqa
from node_embeddings import utils  # noqa

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
        raw_mask_channel=config_siamese.raw_mask_channel,
        num_workers=config.num_workers,
        in_memory=config_siamese.in_memory,
        inference_samples=config_siamese.inference_samples,
        rag_block_size=config_siamese.rag_block_size,
        rag_from_file=config_siamese.rag_from_file,
        dump_rag=config_siamese.dump_rag
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=config_siamese.batch_size_eval,
        num_workers=config_siamese.num_workers_dataloader,
        pin_memory=config_siamese.pin_memory,
    )
    logger.info(f'init dataloader in {now() - start} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #########################

    start = now()

    model = SiameseVgg3d(
        input_size=np.array(config_siamese.patch_size) /
        np.array(config.voxel_size_emb),
        input_fmaps=int(config_siamese.raw_channel) +  # noqa
            int(config_siamese.mask_channel) +  # noqa
            int(config_siamese.raw_mask_channel),  # noqa
        fmaps=config_siamese.fmaps,
        fmaps_max=config_siamese.fmaps_max,
        output_features=config_siamese.output_features,
        downsample_factors=config_siamese.downsample_factors
    )

    # model.to(device) has to be executed before loading the state
    # dicts
    model.to(device)
    checkpoint = utils.load_checkpoint(
        load_model=config_siamese.load_model,
        load_model_version=config_siamese.load_model_version,
        runs_dir=config_siamese.runs_dir
    )
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

    # ----------- inference loop -----------
    node_ids = []
    embeddings = []

    logger.info('start inference loop')
    for i, data in enumerate(dataloader):
        start_batch = now()
        logger.info(f'batch {i} ...')
        patches, node_ids_batch = data

        # make sure the dimensionality is ok
        assert patches.dim() == 5, patches.shape

        patches = patches.to(device)
        out = model.forward_once(patches)

        if torch.cuda.is_available():
            logger.debug(
                f'max GPU memory allocated: {torch.cuda.max_memory_allocated(device=device)/(2**30)} GiB')

        node_ids.extend(list(node_ids_batch.numpy()))
        embeddings.extend(list(out.cpu().numpy()))

        samples_count += config_siamese.batch_size_eval
        if samples_count >= samples_limit:
            break

        logger.debug(f'batch {i} in {now() - start_batch} s')

    dataset.write_embeddings_to_db(
        node_ids=node_ids,
        embeddings=embeddings,
        collection_name=f'nodes_embeddings_{timestamp}_{config_siamese.comment}',
    )


if __name__ == '__main__':
    create_embeddings()
