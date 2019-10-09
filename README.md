# Supervoxel Agglomeration using Graph Neural Networks

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7a92c3102bdf4bb4ac5257a800754932)](https://app.codacy.com/app/benjamin9555/gnn_agglomeration?utm_source=github.com&utm_medium=referral&utm_content=benjamin9555/gnn_agglomeration&utm_campaign=Badge_Grade_Dashboard)

using [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)

### Abstract

Fully automating neural circuit reconstruction from electron microscopy (EM) data has the potential to provide a huge leap for neuroscience. The final step of many neuron reconstruction pipelines consists of agglomerating small compact fragments to form entire neurons. This work explores using Graph Neural Networks to learn that task from the Region Adjacency Graph (RAG) of said fragments, combined with features learned by a Convolutional Neural Network directly from EM data. For this purpose we present a generalizable, geometric Graph Neural Network called _RagNet_. It labels RAG edges as _merge_ or _split_, which allows the direct extraction of segmentations. In contrast to previous agglomeration methods, the _RagNet_ allows an increase in the context considered for making an edge merge decision. This benefit is empirically confirmed by applying _RagNet_ on top of an edge-wise prediction method and increasing the class-balanced accuracy by ten percentage points. Lastly, we fuse our implementation with an existing large-scale neuron reconstruction pipeline and report initial results on a recently imaged _Drosophila melanogaster_ brain dataset.

### Installation
1. Set up a python3 conda environment, then `pip install -r requirements.txt` or `conda env create -f environment.yml`
1. Conda install pytorch as specified [here](https://pytorch.org/get-started/locally/)
1. Install pytorch_geometric with `./install_pytorch_geometric.sh`, or one-by-one as specified [here](https://github.com/rusty1s/pytorch_geometric)
1. Adapt `gnn_agglomeration/config.py` and run `main.py`

### Experiment tracking with Sacred
1. To install MongoDB on MacOS, execute `brew install mongodb`, then start it as a service with `brew services start mongodb`
1. Access the mongodb via the _mongo shell_ with `mongo`
1. In there, set up a new database called sacred with `use sacred`
1. Insert a dummy entry with `db.dummy.insert({"dummy":"dummy"})`
1. To set up Omniboard, follow the steps [here](https://vivekratnavel.github.io/omniboard/#/quick-start)
1. Execute `omniboard -m <host>:27017:sacred` to start up Omniboard at `localhost:9000`
