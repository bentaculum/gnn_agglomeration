from pymongo import MongoClient
import sys
import pandas as pd

client_address = sys.argv[1]
client = MongoClient(client_address)

db = client.hemi_mtlsd_400k
db_nodes = db.nodes
db_edges = db.edges

print('Loading RAG nodes from DB ...')
cursor = db_nodes.find()
nodes = pd.DataFrame(list(cursor))

print('Loading RAG edges from DB ...')
cursor = db_edges.find()
edges = pd.DataFrame(list(cursor))
