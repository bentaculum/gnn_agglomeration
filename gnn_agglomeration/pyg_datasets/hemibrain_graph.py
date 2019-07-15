import torch
from torch_geometric.data import Data
import logging
import daisy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraph(Data, ABC):

    # can't overwrite __init__ using different args than base class

    @abstractmethod
    def read_and_process(
            self,
            graph_provider,
            block_offset,
            block_shape,
            inner_block_offset,
            inner_block_shape):
        pass

    def parse_rag_excerpt(self, node_attrs, edge_attrs):
        df_nodes = pd.DataFrame(node_attrs)
        # columns in desired order
        # Careful, I use int64 instead of uint64 here
        df_nodes = df_nodes[['id', 'center_z',
                             'center_y', 'center_x']].astype(np.int64)

        df_edges = pd.DataFrame(edge_attrs)
        # columns in desired order
        # TODO account for directed edges
        # TODO parametrize the used names
        merge_score_field = 'merge_score'
        gt_merge_score_field = 'gt_merge_score'
        merge_labeled_field = 'merge_labeled'

        df_edges = df_edges[['u', 'v', merge_score_field,
                             gt_merge_score_field, merge_labeled_field]]
        df_edges[merge_score_field] = df_edges[merge_score_field].astype(
            np.float32)
        df_edges[gt_merge_score_field] = df_edges[gt_merge_score_field].astype(
            np.int_)
        df_edges[merge_labeled_field] = df_edges[merge_labeled_field].astype(
            np.int_)

        nodes_remap = dict(zip(df_nodes['id'], range(len(df_nodes))))
        node_ids = torch.tensor(df_nodes['id'].values, dtype=torch.long)

        df_edges['u'] = df_edges['u'].map(nodes_remap)
        df_edges['v'] = df_edges['v'].map(nodes_remap)

        # Drop edges for which one of the incident nodes is not in the
        # extracted node set
        df_edges = df_edges[np.isfinite(
            df_edges['u']) & np.isfinite(df_edges['v'])]

        df_edges['u'] = df_edges['u'].astype(np.int64)
        df_edges['v'] = df_edges['v'].astype(np.int64)

        # edge index requires dimensionality of (2,e)
        # pyg works with directed edges, duplicate each edge here
        edge_index_undir = df_edges[['u', 'v']].values
        edge_index_dir = np.repeat(edge_index_undir, 2, axis=0)
        edge_index_dir[1::2, :] = np.flip(edge_index_dir[1::2, :], axis=1)
        edge_index = torch.tensor(edge_index_dir.transpose(), dtype=torch.long)

        edge_attr_undir = df_edges[merge_score_field].values
        edge_attr_dir = np.repeat(edge_attr_undir, 2, axis=0)
        edge_attr = torch.tensor(edge_attr_dir, dtype=torch.float)

        pos = torch.tensor(
            df_nodes[['center_z', 'center_y', 'center_x']].values, dtype=torch.float)

        # TODO node features go here
        x = torch.ones(len(df_nodes), 1, dtype=torch.float)

        # Targets operate on undirected edges, therefore no duplicate necessary
        mask = torch.tensor(
            df_edges[merge_labeled_field].values,
            dtype=torch.float)
        y = torch.tensor(
            df_edges[gt_merge_score_field].values,
            dtype=torch.long)

        return edge_index, edge_attr, x, pos, node_ids, mask, y

    # TODO update this
    def plot_predictions(self, config, pred, graph_nr, run, acc, logger):
        pass
