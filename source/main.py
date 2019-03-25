from source.config import Config
from source.random_graph_dataset import RandomGraphDataset
from source.my_graph import MyGraph
from source.gcn_regression import GcnRegression

import torch
import torch.nn.functional as F

if __name__  == '__main__':
    config = Config().parse_args()

    dataset = RandomGraphDataset(root=config.dataset_path, config=config)


    device = torch.device('cpu')
    model = GcnRegression(config=config).to(device)
    data = dataset[0].to(device)

    # TODO move loss display to the model as well

    # put model in training mode (e.g. use dropout)
    model.train()

    for epoch in range(config.training_epochs):
        # clear the gradient variables of the model
        model.optimizer.zero_grad()
        # call the forward method
        out = model(data)
        loss = model.loss(out, data.y)
        print('epoch {} MSE loss: {} '.format(epoch, loss))
        loss.backward()
        model.optimizer.step()

    model.evaluate_metric(data)

    # print the first graph in the dataset
    g = MyGraph(config, dataset[0])
    g.plot_predictions(model.evaluate_as_list(data))
    print('Done')
