import logging
import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')  # noqa

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    # # find latest state of the model
    # def extract_number(f):
    #     s = re.findall(r'\d+', f)
    #     return int(s[0]) if s else -1, f
    #
    # # delete older models
    # checkpoint_versions = [name for name in os.listdir(model_dir) if (
    #     name.endswith('.tar') and name.startswith('iteration'))]
    # if len(checkpoint_versions) >= 3:
    #     checkpoint_to_remove = min(checkpoint_versions, key=extract_number)
    #     os.remove(osp.join(model_dir, checkpoint_to_remove))

    # save the new one
    model_tar = osp.join(model_dir, f'iteration_{iteration}.tar')
    logger.info(f'saving model to {model_tar} ...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_tar)

    # save config files
    # TODO save validation config as well
    # TODO check if config files get nicely separated
    parser_siamese.write_config_file(
        parsed_namespace=config_siamese,
        output_file_paths=[osp.join(model_dir, 'config_siamese.ini')],
        exit_after=False
    )
    parser_ds.write_config_file(
        parsed_namespace=config,
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


def output_similarities_split(writer, iteration, out0, out1, labels):
    mask = labels == 1
    output_similarities = torch.nn.functional.cosine_similarity(
        out0, out1, dim=1)

    if len(output_similarities[mask]) > 0:
        writer.add_histogram(
            '01/output_similarities/class1',
            output_similarities[mask],
            iteration
        )
    if len(output_similarities[~mask]) > 0:
        writer.add_histogram(
            '01/output_similarities/class-1',
            output_similarities[~mask],
            iteration
        )


def accuracy_thresholded(out0, out1, labels, threshold):
    cosine_similarity = torch.nn.functional.cosine_similarity(
        out0, out1, dim=1)
    mask = cosine_similarity > threshold
    pred = torch.zeros_like(cosine_similarity)
    pred[mask] = 1.0
    pred[~mask] = -1.0
    correct = pred.eq(labels.float()).sum().item()
    acc = correct / labels.size(0)
    return acc


def write_accuracy_to_summary(writer, iteration, out0, out1, labels):
    threshold = config_siamese.accuracy_threshold
    accuracy = accuracy_thresholded(out0, out1, labels, threshold)
    writer.add_scalar(
        tag=f'00/accuracy/threshold_{threshold:.2f}',
        scalar_value=accuracy,
        global_step=iteration
    )


def run_validation(model, loss_function, labels, dataloader, device, writer, train_iteration, outputs_dir):
    start = now()
    model.eval()

    out0_list = []
    out1_list = []
    labels_list = []
    for i, data in enumerate(dataloader):
        start_batch = now()
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

        val_iteration = train_iteration + i
        write_accuracy_to_summary(
            writer=writer,
            iteration=val_iteration,
            out0=out0,
            out1=out1,
            labels=labels
        )
        output_similarities_split(
            writer=writer,
            iteration=val_iteration,
            out0=out0,
            out1=out1,
            labels=labels
        )
        writer.add_scalar(
            tag='00/loss',
            scalar_value=loss,
            global_step=train_iteration + i
        )

        # TODO this is only a patch
        out0_list.append(list(out0.detach().cpu().numpy()))
        out1_list.append(list(out1.detach().cpu().numpy()))
        labels_list.append(list(labels.detach().cpu().numpy()))

        logger.info(f'validation batch {i} in {now() - start_batch}')

    model.train()
    np.savez(
        os.path.join(outputs_dir, f'validation_set_{train_iteration}'),
        out0=np.array(out0_list),
        out1=np.array(out1_list),
        labels=np.array(labels_list)
    )
    logger.info(f'run validation in {now() - start} s')


def train():
    logger.info('start training function')
    timestamp = datetime.datetime.now(
        pytz.timezone('US/Eastern')).strftime('%Y%m%dT%H%M%S.%f%z')

    # make necessary logging directories
    run_dir = osp.join(config_siamese.runs_dir,
                       f'{timestamp}_{config_siamese.comment}')
    model_dir = osp.join(run_dir, 'model')
    os.makedirs(model_dir)
    summary_dir = osp.join(run_dir, 'summary')
    os.makedirs(summary_dir)
    outputs_dir = osp.join(run_dir, 'outputs')
    os.makedirs(outputs_dir)

    start = now()
    dataset = SiameseDatasetTrain(
        patch_size=config_siamese.patch_size,
        raw_channel=config_siamese.raw_channel,
        mask_channel=config_siamese.mask_channel,
        raw_mask_channel=config_siamese.raw_mask_channel,
        num_workers=config.num_workers,
        in_memory=config_siamese.in_memory,
        rag_block_size=config_siamese.rag_block_size,
        rag_from_file=config_siamese.rag_from_file,
        dump_rag=config_siamese.dump_rag,
        snapshots=config_siamese.snapshots
    )
    logger.info(f'init dataset in {now() - start} s')

    start = now()

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=dataset.samples_weights,
        num_samples=config_siamese.training_samples,
        replacement=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=config_siamese.batch_size_train,
        num_workers=config_siamese.num_workers_dataloader,
        pin_memory=config_siamese.pin_memory,
        worker_init_fn=lambda idx: np.random.seed()
    )
    logger.info(f'init dataloader in {now() - start} s')

    if config_siamese.use_validation:
        start = now()
        dataset_val = SiameseDatasetTrain(
            patch_size=config_siamese.patch_size,
            raw_channel=config_siamese.raw_channel,
            mask_channel=config_siamese.mask_channel,
            raw_mask_channel=config_siamese.raw_mask_channel,
            num_workers=config.num_workers,
            in_memory=config_siamese.in_memory,
            rag_block_size=config_siamese.rag_block_size,
            rag_from_file=config_siamese.rag_from_file_val,
            dump_rag=config_siamese.dump_rag_val,
            snapshots=config_siamese.snapshots,
            config_from_file=config_siamese.validation_config
        )
        logger.info(f'init dataset val in {now() - start} s')

        # sampler_val = torch.utils.data.WeightedRandomSampler(
        #     weights=dataset_val.samples_weights,
        #     num_samples=config_siamese.validation_samples,
        #     replacement=True
        # )

        sampler_val = torch.utils.data.SequentialSampler(
            data_source=dataset_val
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler_val,
            batch_size=config_siamese.batch_size_eval,
            num_workers=config_siamese.num_workers_dataloader,
            pin_memory=config_siamese.pin_memory,
            worker_init_fn=lambda idx: np.random.seed()
        )
        logger.info(f'init dataloader val in {now() - start} s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    start = now()

    # init tensorboard summary writer
    if config_siamese.summary_loss or config_siamese.summary_detailed:
        if config_siamese.use_validation:
            writer_val = torch.utils.tensorboard.SummaryWriter(
                log_dir=osp.join(summary_dir, 'val')
            )
            writer = torch.utils.tensorboard.SummaryWriter(
                log_dir=osp.join(summary_dir, 'train')
            )
        else:
            writer = torch.utils.tensorboard.SummaryWriter(
                log_dir=summary_dir
            )
    else:
        writer = None

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
        if i % config_siamese.console_update_interval == 0:
            start_console_update = now()
        # start_batch = now()
        logger.debug(f'batch {i} ...')

        input0, input1, labels = data

        if logger.getEffectiveLevel() <= logging.DEBUG:
            unique_labels = labels.unique()
            counts = [len(np.where(labels.numpy() == l.item())[0])
                      for l in unique_labels]
            for l, c in zip(unique_labels.tolist(), counts):
                logger.debug(f'# class {int(l)}: {int(c)}')

        # make sure the dimensionality is ok
        # assert input0.dim() == 5, input0.shape
        # assert labels.dim() == 1, labels.shape

        start_to_gpu = now()
        labels = labels.to(device)
        input0 = input0.to(device)
        input1 = input1.to(device)
        logger.debug(f'tensors to gpu in {now() - start_to_gpu} s')

        start_forward = now()
        out0, out1 = model(input0, input1)
        logger.debug(f'forward in {now() - start_forward} s')

        start_loss = now()
        loss = loss_function(
            input1=out0,
            input2=out1,
            target=labels
        )
        logger.debug(f'loss in {now() - start_loss} s')

        start_register = now()
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
        logger.debug(f'register atexit in {now() - start_register} s')

        start_backward = now()
        loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        logger.debug(f'backward + step in {now() - start_backward} s')

        utils.log_max_memory_allocated(device)

        if config_siamese.summary_loss:
            if i % config_siamese.summary_interval == 0:
                start_summary = now()
                write_accuracy_to_summary(
                    writer=writer,
                    iteration=i,
                    out0=out0,
                    out1=out1,
                    labels=labels
                )
                logger.debug(
                    f'write accuracy to summary in {now() - start_summary} s')

                start_summary = now()
                writer.add_scalar(
                    tag='00/loss',
                    scalar_value=loss,
                    global_step=i
                )
                logger.debug(
                    f'write loss to summary in {now() - start_summary} s')

        if config_siamese.summary_detailed:
            write_network_to_summary(
                writer=writer,
                iteration=i,
                model=model)

        # print(f'batch {i} done in {now() - start_batch} s', end='\r')
        if i % config_siamese.console_update_interval == config_siamese.console_update_interval - 1 \
                or config_siamese.console_update_interval == 1:
            logging.info(
                f'batches {i} done in {now() - start_console_update} s')

        if config_siamese.use_validation:
            if i % config_siamese.validation_interval == 0:
                logger.info('starting validation')
                run_validation(
                    model=model,
                    loss_function=loss_function,
                    labels=labels,
                    dataloader=dataloader_val,
                    device=device,
                    writer=writer_val,
                    train_iteration=i,
                    outputs_dir=outputs_dir
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
