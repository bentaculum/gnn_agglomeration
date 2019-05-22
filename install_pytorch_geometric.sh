#!/bin/sh
pip install --no-cache-dir torch-scatter
pip install --no-cache-dir torch-sparse
pip install --no-cache-dir torch-cluster
pip install --no-cache-dir torch-spline-conv 
pip install -e git+https://github.com/rusty1s/pytorch_geometric.git@74c4c3c677d9319fcebbfcb421fd711c321f1afe#egg=torch-geometric --no-cache-dir
#pip install torch-geometric
