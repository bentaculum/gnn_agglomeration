from source.config import Config
from source.gcn_classification import GcnClassification
from source.random_graph_dataset import RandomGraphDataset
from source.my_graph import MyGraph
from source.gcn_regression import GcnRegression

import torch
import torch.nn.functional as F
import os

if __name__  == '__main__':
    config = Config().parse_args()

    # make necessary directory structure
    if not os.path.isdir(config.temp_dir):
        os.makedirs(config.temp_dir)

    dataset = RandomGraphDataset(root=config.dataset_path, config=config)
    config.max_neighbors = dataset.max_neighbors()
    # TODO make the config editable?

    device = torch.device('cpu')
    model = GcnClassification(config=config).to(device)
    data = dataset[0].to(device)

    # put model in training mode (e.g. use dropout)
    model.train()

    for epoch in range(config.training_epochs):
        # clear the gradient variables of the model
        model.optimizer.zero_grad()
        # call the forward method
        out = model(data)
        loss = model.loss(out, data.y)
        model.print_current_loss(epoch)
        loss.backward()
        model.optimizer.step()

    model.evaluate_metric(data)

    # print the first graph in the dataset
    g = MyGraph(config, dataset[0])
    g.plot_predictions(model.evaluate_as_list(data))
    print('Done')
