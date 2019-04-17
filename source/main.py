from config import Config
from gcn_model import GcnModel
from gmm_conv_model import GmmConvModel

from random_graph_dataset import RandomGraphDataset
from my_graph import MyGraph

import torch
import os
import shutil
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

if __name__  == '__main__':
    config = Config().parse_args()

    # load model
    # Overwrite the config file with the one that has been saved
    # Then overwrite again with possible new params that have been set in the command line
    # TODO make this nicer
    checkpoint = None
    if config.load_model is not None:
        path = config.load_model
        if config.load_model == 'latest':
            path = os.path.join(config.temp_dir, config.model_dir, 'final.tar')

        checkpoint = torch.load(path)
        new_config = vars(config)
        config = checkpoint['config']

        # overwrite with possible new config variables
        for k, v in new_config.items():
            setattr(config, k, v)

    # make necessary directory structure
    if not os.path.isdir(config.temp_dir):
        os.makedirs(config.temp_dir)


    # clear old stuff from the temp dir
    summary_dir = os.path.join(config.temp_dir, config.summary_dir)
    if os.path.isdir(summary_dir):
        shutil.rmtree(summary_dir)
    model_dir = os.path.join(config.temp_dir, config.model_dir)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)

    # make dir structure in temp dir
    os.makedirs(summary_dir)
    os.makedirs(model_dir)


    # set up the summary writer for tensorboardX
    train_writer = SummaryWriter(os.path.join(config.temp_dir, 'summary', 'training'))
    val_writer = SummaryWriter(os.path.join(config.temp_dir, 'summary', 'validation'))

    # create and load dataset
    dataset = RandomGraphDataset(root=config.dataset_path, config=config)
    config.max_neighbors = dataset.max_neighbors()
    dataset = dataset.shuffle()

    # split into train and test
    split_train_idx = int(config.samples * (1 - config.test_split - config.validation_split))
    split_validation_idx = int(config.samples * (1 - config.test_split))

    train_dataset = dataset[:split_train_idx]
    validation_dataset = dataset[split_train_idx:split_validation_idx]
    test_dataset = dataset[split_validation_idx:]

    device = torch.device('cpu')

    data_loader_train = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True)
    data_loader_validation = DataLoader(validation_dataset, batch_size=config.batch_size_eval, shuffle=False)

    try:
        if checkpoint is None:
            model = globals()[config.model](
                config=config,
                train_writer=train_writer,
                val_writer=val_writer,
                model_type=config.model_type
            )
        else:
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
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    except Exception as e:
        print(e)
        raise NotImplementedError('The model you have specified is not implemented')

    model = model.to(device)

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
            epoch_metric_train += model.out_to_metric(out, data.y) * data.num_graphs

            # clear the gradient variables of the model
            model.optimizer.zero_grad()

            loss.backward()
            model.optimizer.step()
            model.train_batch_iteration += 1

        epoch_loss /= train_dataset.__len__()
        train_writer.add_scalar('per_epoch/loss', epoch_loss, epoch)
        epoch_metric_train /= train_dataset.__len__()
        train_writer.add_scalar('per_epoch/metric', epoch_metric_train, epoch)

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
            epoch_metric_val += model.out_to_metric(out, data.y) * data.num_graphs
            model.val_batch_iteration += 1

        # The numbering of train and val does not correspond 1-to-1!
        # Here we skip some numbers for maintaining loose correspondence
        model.val_batch_iteration = model.train_batch_iteration

        validation_loss /= validation_dataset.__len__()
        val_writer.add_scalar('per_epoch/loss', validation_loss, epoch)
        epoch_metric_val /= validation_dataset.__len__()
        val_writer.add_scalar('per_epoch/metric', epoch_metric_val, epoch)

        model.epoch += 1

    # save the final model
    model.save('final.tar')

    ###########################

    model.eval()
    model.current_writer = None

    # train loss
    final_loss_train = 0.0
    final_metric_train = 0.0
    for i, data in enumerate(data_loader_train):
        data = data.to(device)
        out = model(data)
        final_loss_train += model.loss(out, data.y).item() * data.num_graphs
        final_metric_train += model.out_to_metric(out, data.y) * data.num_graphs
    final_loss_train /= train_dataset.__len__()
    final_metric_train /= train_dataset.__len__()

    # test loss
    data_loader_test = DataLoader(test_dataset, batch_size=config.batch_size_eval, shuffle=False)
    test_loss = 0.0
    test_metric = 0.0

    for i, data in enumerate(data_loader_test):
        data = data.to(device)
        out = model(data)
        test_loss += model.loss(out, data.y).item() * data.num_graphs
        test_metric += model.out_to_metric(out, data.y) * data.num_graphs
    test_loss /= test_dataset.__len__()
    test_metric /= test_dataset.__len__()

    # final print routine
    print('')
    print('Maximum # of neighbors within distance {} in dataset: {}'.format(config.theta, config.max_neighbors))
    print('# of neighbors, distribution:')
    dic = dataset.neighbors_distribution()
    for key, value in sorted(dic.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print('')
    print('Mean train loss ({} samples): {}'.format(
        train_dataset.__len__(),
        final_loss_train))
    print('Mean accuracy on train set: {}'.format(
        final_metric_train))
    print('Mean test loss ({} samples): {}'.format(
        test_dataset.__len__(),
        test_loss))
    print('Mean accuracy on test set: {}'.format(
        test_metric))
    print('')

    # plot the first graph in the dataset
    g = MyGraph(config, train_dataset[0])
    g.plot_predictions(model.predictions_to_list(model.out_to_predictions(model(train_dataset[0]))))
