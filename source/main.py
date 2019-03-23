from source.config import Config
from source.random_graph_dataset import RandomGraphDataset
from source.gnn_example import GnnExample

import torch
import torch.nn.functional as F

if __name__  == '__main__':
    config = Config().parse_args()

    # graph = RandomGraph(config)
    # graph.create_graph()

    dataset = RandomGraphDataset(root=config.dataset_path, config=config)

    device = torch.device('cpu')
    model = GnnExample(config=config).to(device)
    data = dataset[0].to(device)

    # TODO move optimizer, loss, evaluation metric to the model class
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # put model in training mode (e.g. use dropout)
    model.train()

    for epoch in range(config.training_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        print('epoch {} MSE loss: {} '.format(epoch, loss))
        loss.backward()
        optimizer.step()

    # put model in evaluation mode
    model.eval()

    pred = model(data).round()
    # print(torch.squeeze(pred))
    # print(data.y)
    correct = torch.squeeze(pred).eq(data.y).sum().item()
    acc = correct / data.num_nodes
    print('Accuracy: {:.4f}'.format(acc))

    print('Done')
