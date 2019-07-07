import torch
import logging
import daisy
import numpy as np
import pandas as pd

from .hemibrain_graph import HemibrainGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraphMasked(HemibrainGraph):

    # can't overwrite __init__ using different args than base class

    def read_and_process(self, graph_provider, block_offset, block_shape, inner_block_offset, inner_block_shape):
        # TODO remove duplicate code
        logger.debug(
            'read\n'
            f'\tblock offset: {block_offset}\n'
            f'\tblock shape: {block_shape}'
        )

        roi = daisy.Roi(list(block_offset), list(block_shape))
        node_attrs = graph_provider.read_nodes(roi=roi)
        edge_attrs = graph_provider.read_edges(roi=roi, nodes=node_attrs)

        if len(node_attrs) == 0:
            raise ValueError('No nodes found in roi %s' % roi)
        if len(edge_attrs) == 0:
            raise ValueError('No edges found in roi %s' % roi)

        self.edge_index, \
        self.edge_attr, \
        self.x, \
        self.pos, \
        self.node_ids, \
        mask, \
        self.y = self.parse_rag_excerpt(node_attrs, edge_attrs)

        self.mask = self.mask_target_edges(
            graph_provider=graph_provider,
            inner_roi=daisy.Roi(list(inner_block_offset), list(inner_block_shape)),
            mask=mask
        )

    def mask_target_edges(self, graph_provider, inner_roi, mask):
        # parse inner block
        inner_nodes = graph_provider.read_nodes(roi=inner_roi)
        inner_edges = graph_provider.read_edges(roi=inner_roi, nodes=inner_nodes)

        inner_edge_index, _, _, _, inner_node_ids, _, _ = self.parse_rag_excerpt(inner_nodes, inner_edges)

        # remap inner and outer node ids to original ids
        inner_idx_flat = torch.transpose(inner_edge_index, 0, 1).flatten()
        inner_orig_edge_index = inner_node_ids[inner_idx_flat].reshape((-1, 2)).tolist()
        outer_idx_flat = torch.transpose(self.edge_index, 0, 1).flatten()
        outer_orig_edge_index = self.node_ids[outer_idx_flat].reshape((-1, 2)).tolist()

        for i, edge in enumerate(outer_orig_edge_index):
            if edge not in inner_orig_edge_index:
                # we have n labels (undirected), but 2n directed edges
                # --> div 2
                mask[int(i/2)] = 0

        return mask
