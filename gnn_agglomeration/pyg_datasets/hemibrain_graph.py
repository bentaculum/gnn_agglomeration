import torch
from torch_geometric.data import Data
import logging
import numpy as np
from abc import ABC, abstractmethod
import time
from time import time as now
from funlib.segment.arrays import replace_values

from gnn_agglomeration import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraph(Data, ABC):

    # can't overwrite __init__ using different args than base class, but use kwargs

    @abstractmethod
    def read_and_process(
            self,
            graph_provider,
            embeddings,
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
        # remove config property so Data object can be saved with torch
        del self.config

    def parse_rag_excerpt(self, nodes_list, edges_list, embeddings, all_nodes):

        # TODO parametrize the used names
        id_field = 'id'
        node1_field = 'u'
        node2_field = 'v'
        merge_score_field = 'merge_score'
        gt_merge_score_field = self.config.gt_merge_score_field
        merge_labeled_field = self.config.merge_labeled_field

        node_attrs = utils.to_np_arrays(nodes_list)
        edges_attrs = utils.to_np_arrays(edges_list)

        start = now()
        u_in = np.isin(edges_attrs[node1_field], node_attrs[id_field])
        # sanity check: all u nodes should be contained in the nodes extracted by mongodb
        assert np.sum(~u_in) == 0

        # TODO this is dirty. clean up
        v_in = np.isin(edges_attrs[node2_field], node_attrs[id_field])
        missing_node_ids = np.unique(edges_attrs[node2_field][v_in])
        for i in missing_node_ids:
            node_attrs[id_field] = np.append(node_attrs[id_field], i)
            node_attrs['center_z'] = np.append(node_attrs['center_z'], all_nodes[i]['center_z'])
            node_attrs['center_y'] = np.append(node_attrs['center_y'], all_nodes[i]['center_y'])
            node_attrs['center_x'] = np.append(node_attrs['center_x'], all_nodes[i]['center_x'])

        logger.debug(f'add missing nodes to node_attrs in {now() - start} s')

        # drop edges for which one of the incident nodes is not in the
        # extracted node set
        edges_attrs = utils.drop_outgoing_edges(
            node_attrs=node_attrs,
            edge_attrs=edges_attrs,
            id_field=id_field,
            node1_field=node1_field,
            node2_field=node2_field
        )

        # If all edges were removed in the step above, raise a ValueError
        # that is caught later on
        if len(edges_attrs[node1_field]) == 0:
            raise ValueError(
                f'Removed all edges in ROI, as one node is outside of ROI for each edge')

        start = now()
        if embeddings is None:
            x = torch.ones(len(node_attrs[id_field]), 1, dtype=torch.float)
        else:
            # TODO this is for debugging. Later, I should have an embedding for each node
            # embeddings_list = [embeddings[i] if i in embeddings else np.random.rand(10) for i in
            #                    node_attrs[id_field]]
            embeddings_list = [embeddings[i] for i in node_attrs[id_field]]
            x = torch.tensor(embeddings_list, dtype=torch.float)
        logger.info(f'load embeddings from dict in {now() - start} s')

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
        edge_attr_undir = edges_attrs[merge_score_field]

        # add edges, together with a dummy merge score. extend mask, edgewise label
        if self.config.self_loops:
            num_nodes = len(node_attrs[id_field])
            loops = np.stack([np.arange(num_nodes, dtype=np.int64), np.arange(num_nodes, dtype=np.int64)])
            edge_index_undir = np.concatenate([edge_index_undir, loops])

            edge_attr_undir = np.concatenate([edge_attr_undir, np.zeros(num_nodes)], axis=0)

            edges_attrs[merge_labeled_field] = np.concatenate(
                [edges_attrs[merge_labeled_field],
                 np.zeros(num_nodes)]
            )

            edges_attrs[gt_merge_score_field] = np.concatenate(
                [edges_attrs[gt_merge_score_field],
                 np.zeros(num_nodes)]
            )

        edge_index_dir = np.repeat(edge_index_undir, 2, axis=0)
        edge_index_dir[1::2, :] = np.flip(edge_index_dir[1::2, :], axis=1)
        edge_index = torch.tensor(edge_index_dir.astype(
            np.int64).transpose(), dtype=torch.long)

        edge_attr_undir = np.expand_dims(
            edge_attr_undir, axis=1)
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
