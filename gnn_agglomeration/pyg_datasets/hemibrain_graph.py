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
    # TODO can I use kwargs for that?

    @abstractmethod
    def read_and_process(
            self,
            graph_provider,
            block_offset,
            block_shape,
            inner_block_offset,
            inner_block_shape):
        """
        Initiates reading the graph from DB and converting it to the desired format for torch_geometric.
        Assigns values to all torch_geometric.Data attributes

        Args:
            graph_provider (daisy.persistence.MongoDbGraphProvider):

                connection to RAG DB

            block_offset (``list`` of ``int``):

                block offset of extracted graph, in nanometers

            block_shape (``list`` of ``int``):

                block shape of extracted graph, in nanometers

            inner_block_offset (``list`` of ``int``):

                offset of sub-block, which might be used for masking, in nanometers

            inner_block_shape (``list`` of ``int``):

                shape of sub-block, which might be used for masking, in nanometers
        """

        pass

    def assert_graph(self):
        """
        check whether bi-directed edges are next to each other in edge_index
        """
        uv = self.edge_index[:, 0::2]
        vu = torch.flip(self.edge_index, dims=[0])[:, 1::2]

        assert torch.equal(uv, vu)

    def parse_rag_excerpt(self, nodes_list, edges_list):

        # TODO parametrize the used names
        id_field = 'id'
        node1_field = 'u'
        node2_field = 'v'
        merge_score_field = 'merge_score'
        gt_merge_score_field = 'gt_merge_score'
        merge_labeled_field = 'merge_labeled'

        # TODO remove duplicate code, this is also used in hemibrain_graph
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
            raise ValueError(
                f'Removed all edges in ROI, as one node is outside of ROI')

        edges_attrs = to_np_arrays(edges_list)

        node_ids_np = node_attrs[id_field].astype(np.int64)
        node_ids = torch.tensor(node_ids_np, dtype=torch.long)

        # By not operating inplace and providing out_array, we always use
        # the C++ implementation of replace_values

        logger.debug(
            f'before: interval {node_ids_np.max() - node_ids_np.min()}, min id {node_ids_np.min()}, max id {node_ids_np.max()}, shape {node_ids_np.shape}')
        start = time.time()
        edges_node1 = np.zeros_like(
            edges_attrs[node1_field], dtype=np.int64)
        edges_node1 = replace_values(
            in_array=edges_attrs[node1_field].astype(np.int64),
            old_values=node_ids_np,
            new_values=np.arange(len(node_attrs[id_field]), dtype=np.int64),
            inplace=False,
            out_array=edges_node1
        )
        edges_attrs[node1_field] = edges_node1
        logger.debug(
            f'remapping {len(edges_attrs[node1_field])} edges (u) in {time.time() - start} s')
        logger.debug(
            f'edges after: min id {edges_attrs[node1_field].min()}, max id {edges_attrs[node1_field].max()}')

        start = time.time()
        edges_node2 = np.zeros_like(
            edges_attrs[node2_field], dtype=np.int64)
        edges_node2 = replace_values(
            in_array=edges_attrs[node2_field].astype(np.int64),
            old_values=node_ids_np,
            new_values=np.arange(len(node_attrs[id_field]), dtype=np.int64),
            inplace=False,
            out_array=edges_node2)
        edges_attrs[node2_field] = edges_node2
        logger.debug(
            f'remapping {len(edges_attrs[node2_field])} edges (v) in {time.time() - start} s')
        logger.debug(
            f'edges after: min id {edges_attrs[node2_field].min()}, max id {edges_attrs[node2_field].max()}')

        # TODO I could potentially avoid transposing twice
        # edge index requires dimensionality of (2,e)
        # pyg works with directed edges, duplicate each edge here
        edge_index_undir = np.array(
            [edges_attrs[node1_field], edges_attrs[node2_field]]).transpose()
        edge_index_dir = np.repeat(edge_index_undir, 2, axis=0)
        edge_index_dir[1::2, :] = np.flip(edge_index_dir[1::2, :], axis=1)
        edge_index = torch.tensor(edge_index_dir.astype(
            np.int64).transpose(), dtype=torch.long)

        edge_attr_undir = np.expand_dims(
            edges_attrs[merge_score_field], axis=1)
        edge_attr_dir = np.repeat(edge_attr_undir, 2, axis=0)
        edge_attr = torch.tensor(edge_attr_dir, dtype=torch.float)

        pos = torch.transpose(
            input=torch.tensor(
                [
                    node_attrs['center_z'],
                    node_attrs['center_y'],
                    node_attrs['center_x']],
                dtype=torch.float),
            dim0=0,
            dim1=1)

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
