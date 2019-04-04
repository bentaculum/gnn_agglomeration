from config import Config
from gcn_regression import GcnRegression
from gcn_classification import GcnClassification
from gmmconv_classification_n_layers import GmmConvClassification

from random_graph_dataset import RandomGraphDataset
from my_graph import MyGraph

import torch
import os
from torch_geometric.data import DataLoader


if __name__  == '__main__':
    config = Config().parse_args()

    # make necessary directory structure
    if not os.path.isdir(config.temp_dir):
        os.makedirs(config.temp_dir)

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
        model = globals()[config.model](config=config)
    except:
        raise NotImplementedError('The model you have specified is not implemented')
    model = model.to(device)



    for epoch in range(config.training_epochs):
        # put model in training mode (e.g. use dropout)
        model.train()
        for data in data_loader_train:
            data = data.to(device)
            # clear the gr  adient variables of the model
            model.optimizer.zero_grad()
            # call the forward method
            out = model(data)
            loss = model.loss(out, data.y)
            model.print_current_loss(epoch)
            loss.backward()
            model.optimizer.step()

        # validation
        model.eval()
        for data in data_loader_validation:
            data = data.to(device)
            model.evaluate(data)

    # train loss
    train_loss_values = []
    train_metric_values = []
    for i, data in enumerate(data_loader):
        data = data.to(device)
        train_loss_values.append(model.evaluate(data))
        train_metric_values.append(model.evaluate_metric(data))

    # test loss
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size_eval, shuffle=False)
    eval_loss_values = []
    eval_metric_values = []

    for i, data in enumerate(test_dataloader):
        data = data.to(device)
        eval_loss_values.append(model.evaluate(data, i))
        eval_metric_values.append(model.evaluate_metric(data))

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
        torch.mean(torch.tensor(train_loss_values))))
    print('Mean accuracy on train set: {}'.format(
        torch.mean(torch.tensor(train_metric_values))))
    print('Mean test loss ({} samples): {}'.format(
        test_dataset.__len__(),
        torch.mean(torch.tensor(eval_loss_values))))
    print('Mean accuracy on test set: {}'.format(
        torch.mean(torch.tensor(eval_metric_values))))
    print('')

    # plot the first graph in the dataset
    g = MyGraph(config, train_dataset[0])
    g.plot_predictions(model.evaluate_as_list(train_dataset[0]))
