# Supervoxel Agglomeration using Graph Neural Networks
using [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)

### Installation
1. Set up a python3 conda environment, then `pip install -r requirements.txt` or `conda env create -f environment.yml`
1. Conda install pytorch as specified [here](https://pytorch.org/get-started/locally/)
1. Install pytorch_geometric with `./install_pytorch_geometric.sh`, or one-by-one as specified [here](https://github.com/rusty1s/pytorch_geometric)
1. Run 'main.py' out of the directory `./source`

### Experiment tracking with Sacred
1. To install MongoDB on MacOS, execute `brew install mongodb`, then start it as a service with `brew services start mongodb`
1. Access the mongodb via the _mongo shell_ with `mongo`
1. In there, set up a new database called sacred with `use sacred`
1. Insert a dummy entry with `db.dummy.insert({"dummy":"dummy"})`
1. To set up Omniboard, follow the steps [here](https://vivekratnavel.github.io/omniboard/#/quick-start)
1. Execute `omniboard -m <host>:27017:sacred` to start up Omniboard at `localhost:9000`
