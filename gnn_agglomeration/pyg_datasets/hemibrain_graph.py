import torch
from torch_geometric.data import Data
import logging
import numpy as np
from abc import ABC, abstractmethod
import time
from funlib.segment.arrays import replace_values

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

    def parse_rag_excerpt(self, nodes_list, edges_list):

        # TODO parametrize the used names
        id_field = 'id'
        node1_field = 'u'
        node2_field = 'v'
        merge_score_field = 'merge_score'
        gt_merge_score_field = 'gt_merge_score'
        merge_labeled_field = 'merge_labeled'

        def to_np_arrays(inp):
            d = {}
            for i in inp:
                for k, v in i.items():
                    d.setdefault(k, []).append(v)
            for k, v in d.items():
                d[k] = np.array(v)
            return d

        node_attrs = to_np_arrays(nodes_list)
        # TODO maybe port to numpy, but generally fast
        # Drop edges for which one of the incident nodes is not in the
        # extracted node set
        start = time.time()
        for e in reversed(edges_list):
            if e[node1_field] not in node_attrs[id_field] or e[node2_field] not in node_attrs[id_field]:
                edges_list.remove(e)
        logger.debug(f'drop edges at the border in {time.time() - start}s')

        # If all edges were removed in the step above, raise a ValueError
        # that is caught later on
        if len(edges_list) == 0:
            raise ValueError(f'Removed all edges in ROI, as one node is outside of ROI')

        edges_attrs = to_np_arrays(edges_list)

        node_ids = torch.tensor(node_attrs[id_field].astype(np.int64), dtype=torch.long)

        start = time.time()
        # TODO only call once on merged array?
        edges_attrs[node1_field] = replace_values(
            in_array=edges_attrs[node1_field].astype(np.int64),
            old_values=node_attrs[id_field].astype(np.int64),
            new_values=np.arange(len(node_attrs[id_field]), dtype=np.int64),
            inplace=True
        )
        logger.debug(f'remapping {len(edges_attrs[node1_field])} edges (u) in {time.time() - start} s')
        start = time.time()
        edges_attrs[node2_field] = replace_values(
            in_array=edges_attrs[node2_field].astype(np.int64),
            old_values=node_attrs[id_field].astype(np.int64),
            new_values=np.arange(len(node_attrs[id_field]), dtype=np.int64),
            inplace=True
        )
        logger.debug(f'remapping {len(edges_attrs[node2_field])} edges (v) in {time.time() - start} s')

        # TODO I could potentially avoid transposing twice
        # edge index requires dimensionality of (2,e)
        # pyg works with directed edges, duplicate each edge here
        edge_index_undir = np.array([edges_attrs[node1_field], edges_attrs[node2_field]]).transpose()
        edge_index_dir = np.repeat(edge_index_undir, 2, axis=0)
        edge_index_dir[1::2, :] = np.flip(edge_index_dir[1::2, :], axis=1)
        edge_index = torch.tensor(edge_index_dir.astype(np.int64).transpose(), dtype=torch.long)

        edge_attr_undir = np.expand_dims(edges_attrs[merge_score_field], axis=1)
        edge_attr_dir = np.repeat(edge_attr_undir, 2, axis=0)
        edge_attr = torch.tensor(edge_attr_dir, dtype=torch.float)

        pos = torch.transpose(
            input=torch.tensor([node_attrs['center_z'], node_attrs['center_y'], node_attrs['center_x']],
                               dtype=torch.float),
            dim0=0,
            dim1=1
        )

        # TODO node features go here
        x = torch.ones(len(node_attrs[id_field]), 1, dtype=torch.float)

        # Targets operate on undirected edges, therefore no duplicate necessary
        mask = torch.tensor(
            edges_attrs[merge_labeled_field],
            dtype=torch.float)
        y = torch.tensor(
            edges_attrs[gt_merge_score_field],
            dtype=torch.long)

        return edge_index, edge_attr, x, pos, node_ids, mask, y

    # TODO update this
    def plot_predictions(self, config, pred, graph_nr, run, acc, logger):
        pass
