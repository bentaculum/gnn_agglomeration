import torch
import numpy as np
import logging
import daisy
from time import time as now

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
        logger.debug(f'max padding: {self.config.block_padding}')

        # Blockwise dataset should cover the entire dataset
        # Therefore the last block in each dimension will be
        # - overlapping with the penultimate
        # - shrunk
        blocks_per_dim = np.ceil(
            (np.array(
                self.roi_shape) /
             np.array(
                 self.config.block_size))).astype(np.int_)
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
                                np.array(self.roi_offset, dtype=np.int_) +
                                np.array([i, j, k], dtype=np.int_) *
                                np.array(self.config.block_size, dtype=np.int_)
                        ).astype(np.int_)

                        block_shape_new = (
                                np.minimum(
                                    block_offset_new +
                                    np.array(self.config.block_size, dtype=np.int_),
                                    np.array(self.roi_offset, dtype=np.int_) + np.array(self.roi_shape, dtype=np.int_)
                                ) - block_offset_new
                        ).astype(np.int_)

                    elif self.config.block_fit == 'overlap':
                        block_offset_new = np.minimum(
                            np.array(self.roi_offset, dtype=np.int_) +
                            np.array([i, j, k], dtype=np.int_) *
                            np.array(self.config.block_size, dtype=np.int_),
                            np.array(self.roi_offset, dtype=np.int_) +
                            np.array(self.roi_shape, dtype=np.int_) -
                            np.array(self.config.block_size, dtype=np.int_)
                        ).astype(np.int_)
                        block_shape_new = np.array(
                            self.config.block_size, dtype=np.int_)
                    else:
                        raise NotImplementedError(
                            f'block_fit {self.config.block_fit} not implemented')

                    # TODO remove asserts

                    # lower corner
                    assert np.all(block_offset_new >=
                                  np.array(self.roi_offset, dtype=np.int_))

                    # shape
                    assert np.all(block_shape_new <= np.array(
                        self.config.block_size, dtype=np.int_))

                    # upper corner
                    assert np.all(
                        block_offset_new +
                        block_shape_new <= np.array(
                            self.roi_offset, dtype=np.int_) +
                        np.array(
                            self.roi_shape, dtype=np.int_))

                    self.block_offsets.append(block_offset_new)
                    self.block_shapes.append(block_shape_new)

        logger.debug('generated blocks')
        for o, s in zip(self.block_offsets, self.block_shapes):
            logger.debug(daisy.Roi(offset=o, shape=s))

        # check whether the entire ROI seems to be covered by the created blocks
        lower_corner_idx = np.array(self.block_offsets, dtype=np.int_).sum(axis=1).argmin()
        assert np.array_equal(
            self.block_offsets[lower_corner_idx], np.array(self.roi_offset, dtype=np.int_))
        upper_corner_idx = (np.array(self.block_offsets, dtype=np.int_) +
                            np.array(self.block_shapes, dtype=np.int_)).sum(axis=1).argmax()
        assert np.array_equal(
            self.block_offsets[upper_corner_idx] +
            self.block_shapes[upper_corner_idx],
            np.array(self.roi_offset, dtype=np.int_) + np.array(self.roi_shape, dtype=np.int_))

    def get_from_db(self, idx):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        start = now()
        # TODO remove duplicate code

        # Get precomputed block offset, pad the block
        inner_offset = self.block_offsets[idx]
        outer_offset, outer_shape = self.pad_block(
            inner_offset, self.block_shapes[idx])

        logger.info(
            f'get graph {idx} from {daisy.Roi(outer_offset, outer_shape)}')

        graph = globals()[self.config.graph_type](config=self.config)
        # try:

        graph.read_and_process(
            graph_provider=self.graph_provider,
            embeddings=self.embeddings,
            all_nodes=self.all_nodes,
            block_offset=outer_offset,
            block_shape=outer_shape,
            inner_block_offset=inner_offset,
            inner_block_shape=self.block_shapes[idx],
        )

        # logger.debug(f'get_from_db in {now() - start} s')
        # return graph
        # except ValueError as e:
        #     TODO this might lead to unnecessary redundancy,
        # logger.warning(f'{e}, duplicating previous graph')
        # return self.get_from_db((idx - 1) % self.len)
