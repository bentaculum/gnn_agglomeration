from config import Config
from gcn_regression import GcnRegression
from gcn_classification import GcnClassification
from gmmconv_classification_n_layers import GmmConvClassification

from random_graph_dataset import RandomGraphDataset
from my_graph import MyGraph

import torch
import os
import shutil
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

if __name__  == '__main__':
    config = Config().parse_args()

    # make necessary directory structure
    if not os.path.isdir(config.temp_dir):
        os.makedirs(config.temp_dir)


    # clear old summaries from the temp dir
    summary_dir = os.path.join(config.temp_dir, config.summary_dir)
    if os.path.isdir(summary_dir):
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

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
        model = globals()[config.model](config=config, train_writer=train_writer, val_writer=val_writer)
    except:
        raise NotImplementedError('The model you have specified is not implemented')

    model = model.to(device)

    model.train_batch_iteration = 0
    model.val_batch_iteration = 0
    for epoch in range(config.training_epochs):
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

    test_predictions = []
    test_targets = []
    all_x_pos = []
    all_y_pos = []

    for i, data in enumerate(data_loader_test):
        data = data.to(device)
        out = model(data)
        test_loss += model.loss(out, data.y).item() * data.num_graphs
        test_metric += model.out_to_metric(out, data.y) * data.num_graphs

        test_predictions.extend(model.predictions_to_list(model.out_to_predictions(out)))
        test_targets.extend(data.y.tolist())
        all_x_pos.extend(data.pos[:, 0].tolist())
        all_y_pos.extend(data.pos[:, 1].tolist())

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

    import pandas as pd
    import numpy as np
    import chartify

    ch = chartify.Chart(blank_labels=True)
    error = np.array(test_predictions) - np.array(test_targets)
    df = pd.DataFrame({'x': all_x_pos, 'y': all_y_pos, 'error': error})
    df = df[df['error'] != 0]

    ch.plot.text(
        data_frame=df,
        x_column='x',
        y_column='y',
        text_column='error'
    ).axes.set_xaxis_label('x')\
        .axes.set_yaxis_label('y')\
        .set_title('Errors by location in euclidian space')\
        .set_subtitle('Each number corresponds to prediction-target for a misclassified node')
    ch.save(filename=os.path.join(config.temp_dir, 'errors_by_location.png'), format='png')

    # plot the first graph in the dataset
    g = MyGraph(config, train_dataset[0])
    g.plot_predictions(model.predictions_to_list(model.out_to_predictions(model(train_dataset[0]))))
