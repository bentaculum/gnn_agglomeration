# GNN Toy Example
Getting familiar with Graph Neural Nets and the [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) library

### Instructions:
1. Set up a python3 conda environment, then `pip install -r requirements.txt`
1. Conda install pytorch as specified [here](https://pytorch.org/get-started/locally/)
1. Install pytorch_geometric with `./install_pytorch_geometric.sh`, or one-by-one as specified [here](https://github.com/rusty1s/pytorch_geometric)
1. Run 'main.py' out of the directory `./source`

### This repo addresses the following problem:

For a given random geometric graph in a plane, predict the number of
neighbors within a threshold distance θ. In particular:

1. Randomly sample N nodes in a unit square.
2. Connect nodes within a threshold distance θ_max.
3. Label each node with the number of direct neighbors within
  distance θ < θ_max.
4. Train a GNN to predict the label of each node.

<img src='imgs/graph_with_predictions.png' width="800px"/>
