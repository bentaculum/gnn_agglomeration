import numpy as np
import logging

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainDatasetBlockwise(HemibrainDataset):

    def prepare(self):
        self.define_block_offsets()

    def define_block_offsets(self):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        logger.debug(f'block size: {self.config.block_size}')
        logger.debug(f'padding: {self.config.block_padding}')

        blocks_per_dim = (np.array(self.roi_shape) /
                          np.array(self.config.block_size)).astype(int)
        logger.debug(f'blocks per dim: {blocks_per_dim}')
        self.len = int(np.prod(blocks_per_dim))
        logger.info(f'num blocks in dataset: {self.len}')

        self.block_offsets = []

        # Create offsets for all blocks in ROI
        for i in range(blocks_per_dim[0]):
            for j in range(blocks_per_dim[1]):
                for k in range(blocks_per_dim[2]):
                    block_offset_new = np.array(self.roi_offset) + np.array([i, j, k]) * np.array(
                        self.config.block_size, dtype=np.int_)
                    self.block_offsets.append(block_offset_new)

    def get_from_db(self, idx):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        # TODO remove duplicate code

        # Get precomputed block offset, pad the block
        inner_offset = self.block_offsets[idx]
        outer_offset, outer_shape = self.pad_block(
            inner_offset, self.config.block_size)

        graph = globals()[self.config.graph_type]()
        try:
            graph.read_and_process(
                graph_provider=self.graph_provider,
                block_offset=outer_offset,
                block_shape=outer_shape,
                inner_block_offset=inner_offset,
                inner_block_shape=self.config.block_size,
            )
            return graph
        except ValueError as e:
            # TODO this might lead to unnecessary redundancy,
            logger.warning(f'{e}, duplicating previous graph')
            if idx > 0:
                return self.get_from_db(idx - 1)
            else:
                raise NotImplementedError(
                    f'Error for last block in block-wise loading: {e}'
                    'Cannot replace the first block if it is empty'
                )
