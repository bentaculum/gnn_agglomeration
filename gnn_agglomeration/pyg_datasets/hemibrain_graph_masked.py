import torch
import logging
import daisy
import time

from .hemibrain_graph import HemibrainGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraphMasked(HemibrainGraph):

    # can't overwrite __init__ using different args than base class

    def read_and_process(
            self,
            graph_provider,
            block_offset,
            block_shape,
            inner_block_offset,
            inner_block_shape):

        # TODO remove duplicate code
        # logger.debug(
        #     'read\n'
        #     f'\tblock offset: {block_offset}\n'
        #     f'\tblock shape: {block_shape}'
        # )

        start = time.time()
        roi = daisy.Roi(list(block_offset), list(block_shape))
        node_attrs = graph_provider.read_nodes(roi=roi)
        edge_attrs = graph_provider.read_edges(roi=roi, nodes=node_attrs)
        logger.debug(f'read block in {time.time() - start} s')

        if len(node_attrs) == 0:
            raise ValueError('No nodes found in roi %s' % roi)
        if len(edge_attrs) == 0:
            raise ValueError('No edges found in roi %s' % roi)

        start = time.time()
        self.edge_index, \
            self.edge_attr, \
            self.x, \
            self.pos, \
            self.node_ids, \
            mask, \
            self.y = self.parse_rag_excerpt(node_attrs, edge_attrs)
        logger.debug(f'parse rag excerpt in {time.time() - start} s')

        start = time.time()
        self.mask = self.mask_target_edges(
            inner_roi=daisy.Roi(
                list(inner_block_offset),
                list(inner_block_shape)),
            mask=mask)
        logger.debug(f'mask target edges in {time.time() - start} s')

    def mask_target_edges(self, inner_roi, mask):
        lower_limit = torch.tensor(inner_roi.get_offset(), dtype=torch.float)
        upper_limit = lower_limit + torch.tensor(inner_roi.get_shape(), dtype=torch.float)

        nodes_in = torch.all(self.pos > lower_limit, dim=1) & torch.all(self.pos < upper_limit, dim=1)
        edge_index_flat = torch.transpose(self.edge_index, 0, 1).flatten()
        edge_index_bool = nodes_in[edge_index_flat].reshape(-1, 2)

        edge_valid_directed = torch.all(edge_index_bool, dim=1)
        inner_mask = edge_valid_directed[0::2]

        return (inner_mask & mask.byte()).float()
