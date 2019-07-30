import torch
import numpy as np
import logging

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainDatasetBlockwise(HemibrainDataset):

    def prepare(self):
        self.define_blocks()

    def define_blocks(self):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        logger.debug(f'block size: {self.config.block_size}')
        logger.debug(f'padding: {self.config.block_padding}')

        # Blockwise dataset should cover the entire dataset
        # Therefore the last block in each dimension will be
        # - overlapping with the penultimate
        # - shrunk
        blocks_per_dim = np.ceil(
            (np.array(
                self.roi_shape) /
             np.array(
                 self.config.block_size))).astype(int)
        logger.debug(f'blocks per dim: {blocks_per_dim}')

        self.len = int(np.prod(blocks_per_dim))
        logger.info(f'num blocks in dataset: {self.len}')

        self.block_offsets = []
        self.block_shapes = []

        if self.config.block_fit == 'overlap':
            assert np.all(np.array(self.config.block_size)
                          <= np.array(self.roi_shape))

        # Create offsets for all blocks in ROI
        for i in range(blocks_per_dim[0]):
            for j in range(blocks_per_dim[1]):
                for k in range(blocks_per_dim[2]):

                    if self.config.block_fit == 'shrink':
                        block_offset_new = (
                                np.array(self.roi_offset) +
                                np.array([i, j, k]) *
                                np.array(self.config.block_size)
                        ).astype(np.int_)

                        block_shape_new = (
                                np.minimum(
                                    block_offset_new +
                                    np.array(self.config.block_size),
                                    self.roi_offset + self.roi_shape
                                ) - block_offset_new
                        ).astype(np.int_)

                    elif self.config.block_fit == 'overlap':
                        block_offset_new = np.minimum(
                            np.array(self.roi_offset) +
                            np.array([i, j, k]) *
                            np.array(self.config.block_size, dtype=np.int_),
                            np.array(self.roi_offset) +
                            np.array(self.roi_shape) -
                            np.array(self.config.block_size)
                        ).astype(np.int_)
                        block_shape_new = np.array(
                            self.config.block_size, dtype=np.int_)
                    else:
                        raise NotImplementedError(
                            f'block_fit {self.config.block_fit} not implemented')

                    # TODO remove asserts

                    # lower corner
                    assert np.all(block_offset_new >=
                                  np.array(self.roi_offset))

                    # shape
                    assert np.all(block_shape_new <= np.array(
                        self.config.block_size))

                    # upper corner
                    assert np.all(
                        block_offset_new +
                        block_shape_new <= np.array(
                            self.roi_offset) +
                        np.array(
                            self.roi_shape))

                    self.block_offsets.append(block_offset_new)
                    self.block_shapes.append(block_shape_new)

        # check whether the entire ROI seems to be covered by the created blocks
        lower_corner_idx = np.array(self.block_offsets).sum(axis=1).argmin()
        assert np.array_equal(self.block_offsets[lower_corner_idx], self.roi_offset)
        upper_corner_idx = (np.array(self.block_offsets) + np.array(self.block_shapes)).sum(axis=1).argmax()
        assert np.array_equal(
            self.block_offsets[upper_corner_idx] + self.block_shapes[upper_corner_idx],
            self.roi_offset + self.roi_shape)

    def get_from_db(self, idx):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        # TODO remove duplicate code

        # Get precomputed block offset, pad the block
        inner_offset = self.block_offsets[idx]
        outer_offset, outer_shape = self.pad_block(
            inner_offset, self.block_shapes[idx])

        graph = globals()[self.config.graph_type]()
        try:
            graph.read_and_process(
                graph_provider=self.graph_provider,
                block_offset=outer_offset,
                block_shape=outer_shape,
                inner_block_offset=inner_offset,
                inner_block_shape=self.block_shapes[idx],
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
