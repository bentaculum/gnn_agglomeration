import logging
import pymongo
import numpy as np
import configargparse
import configparser
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    p = configargparse.ArgParser(allow_abbrev=False)

    p.add('--db_host', type=str)
    p.add('--db_name', type=str)
    p.add('--collection', type=str)
    p.add('--out_path', type=str)

    config = p.parse_args()

    pw_parser = configparser.ConfigParser()
    pw_parser.read(config.db_host)
    config.db_host = pw_parser['DEFAULT']['db_host']

    return config


def embeddings_db_to_npz(
        db_host,
        db_name,
        collection,
        out_path):

    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    collection_generator = db[collection]

    node_ids = []
    embeddings = []
    for line in collection_generator.find():
        node_ids.append(line['id'])
        embeddings.append(pickle.loads(line['embedding']))

    np.savez(out_path, node_ids=node_ids, embeddings=embeddings)


if __name__ == "__main__":
    config = parse_args()
    embeddings_db_to_npz(
        db_host=config.db_host,
        db_name=config.db_name,
        collection=config.collection,
        out_path=config.out_path
    )
