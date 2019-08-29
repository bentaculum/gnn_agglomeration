import torch
import numpy as np
import logging
import daisy
import time
from time import time as now

from .hemibrain_graph import HemibrainGraph
from gnn_agglomeration.utils import TooManyEdgesException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainGraphMasked(HemibrainGraph):

    # can't overwrite __init__ using different args than base class

    def read_and_process(
            self,
            graph_provider,
            embeddings,
            all_nodes,
            block_offset,
            block_shape,
            inner_block_offset,
            inner_block_shape):

        # TODO remove duplicate code
        logger.debug(
            'read\n'
            f'\tblock offset: {block_offset}\n'
            f'\tblock shape: {block_shape}'
        )

        assert self.config is not None

        start_read_and_process = now()
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
        self.y = self.parse_rag_excerpt(
            node_attrs, edge_attrs, embeddings, all_nodes)
        logger.debug(f'parse rag excerpt in {time.time() - start} s')

        if self.edge_index.size(1) > self.config.max_edges:
            raise TooManyEdgesException(
                f'extracted graph has {self.edge_index.size(1)} edges, but the limit is set to {self.config.max_edges}')

        start = time.time()
        self.mask, self.roi_mask = self.mask_target_edges(
            inner_roi=daisy.Roi(
                list(inner_block_offset),
                list(inner_block_shape)),
            mask=mask)

        self.mask = self.class_balance_mask(y=self.y, mask=self.mask)
        self.nodes_mask = self.mask_nodes(
            pos=self.pos,
            inner_block_offset=inner_block_offset,
            inner_block_shape=inner_block_shape
        )

        logger.debug(f'mask target edges in {time.time() - start} s')

        logger.debug(f'read_and_process in {now() - start_read_and_process} s')

        self.assert_graph()

    def mask_nodes(self, pos, inner_block_offset, inner_block_shape):
        lower_limit = np.array(inner_block_offset, dtype=np.int64)
        upper_limit = lower_limit + np.array(inner_block_shape, dtype=np.int64)
        pos = pos.numpy().astype(np.int64)
        nodes_in = np.all(pos >= lower_limit, axis=1) & \
                   np.all(pos < upper_limit, axis=1)
        return nodes_in

    def mask_target_edges(self, inner_roi, mask):
        logger.debug('masking target edges, zero for all context edges')
        lower_limit = torch.tensor(inner_roi.get_offset(), dtype=torch.long)
        upper_limit = lower_limit + \
                      torch.tensor(inner_roi.get_shape(), dtype=torch.long)

        # Careful, we might be off by 1 here due to casting back and forth between long and float
        pos_long = self.pos.long()
        nodes_in = torch.all(
            pos_long >= lower_limit,
            dim=1) & torch.all(
            pos_long < upper_limit,
            dim=1)

        # only check u (the first node, first direction), as each edge should be
        # unmasked exactly once if we go blockwise
        edge_index_u = self.edge_index[0, 0::2]
        inner_mask = nodes_in[edge_index_u]

        # inner mask is needed for inference
        return inner_mask.float() * mask, inner_mask.byte()
