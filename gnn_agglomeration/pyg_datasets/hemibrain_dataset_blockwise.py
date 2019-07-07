import torch
import torch_geometric.transforms as T
import numpy as np
import logging

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainDatasetBlockwise(HemibrainDataset):
    def __init__(self, root, config, roi_offset, roi_shape, length=None):
        self.config = config
        self.roi_offset = roi_offset
        self.roi_shape = roi_shape

        transform = getattr(T, config.data_transform)(norm=True, cat=True)
        super(HemibrainDataset, self).__init__(
            root=root, transform=transform, pre_transform=None)

        self.pad_total_roi()
        self.define_block_offsets()
        self.connect_to_db()


    def define_block_offsets(self):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        logger.info(f'block size: {self.config.block_size}')
        logger.debug(f'padding: {self.config.block_padding}')

        blocks_per_dim = (np.array(self.roi_shape) / np.array(self.config.block_size)).astype(int)
        logger.info(f'blocks per dim: {blocks_per_dim}')
        self.len = int(np.sum(blocks_per_dim))

        self.block_offsets = []

        # Create offsets for all blocks in ROI
        for i in range(blocks_per_dim[0]):
            for j in range(blocks_per_dim[1]):
                for k in range(blocks_per_dim[2]):
                    block_offset_new = np.array(self.roi_offset) + np.array([i, j, k]) * np.array(
                        self.config.block_size, dtype=np.int_)
                    self.block_offsets.append(block_offset_new)

    def get(self, idx):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        # Get precomputed block offset, pad the block
        inner_offset = self.block_offsets[idx]
        outer_offset, outer_shape = self.pad_block(inner_offset, self.config.block_size)

        graph = globals()[self.config.graph_type]()
        graph.read_and_process(
            graph_provider=self.graph_provider,
            block_offset=outer_offset,
            block_shape=outer_shape,
            inner_block_offset=inner_offset,
            inner_block_shape=self.config.block_size,
        )

        # TODO dynamic data augmentation

        return graph
