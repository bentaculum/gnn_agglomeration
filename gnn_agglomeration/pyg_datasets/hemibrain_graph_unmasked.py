import torch
import logging
import daisy

from .hemibrain_graph import HemibrainGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraphUnmasked(HemibrainGraph):

    # can't overwrite __init__ using different args than base class

    def read_and_process(self, graph_provider, block_offset, block_shape, inner_block_offset, inner_block_shape):
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
        self.mask, \
        self.y = self.parse_rag_excerpt(node_attrs, edge_attrs)
