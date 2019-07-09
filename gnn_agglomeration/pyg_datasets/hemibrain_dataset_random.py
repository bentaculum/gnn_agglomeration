import torch
import numpy as np

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked


class HemibrainDatasetRandom(HemibrainDataset):

    def get_from_db(self, idx):
        """
        block size from global config file, roi_offset and roi_shape
        are local attributes
        """

        # TODO remove duplicate code

        random_offset = np.zeros(3, dtype=np.int_)
        random_offset[0] = np.random.randint(
            low=0, high=self.roi_shape[0] - self.config.block_size[0])
        random_offset[1] = np.random.randint(
            low=0, high=self.roi_shape[1] - self.config.block_size[1])
        random_offset[2] = np.random.randint(
            low=0, high=self.roi_shape[2] - self.config.block_size[2])
        total_offset = self.roi_offset + random_offset

        outer_offset, outer_shape = self.pad_block(total_offset, self.config.block_size)
        graph = globals()[self.config.graph_type]()
        graph.read_and_process(
            graph_provider=self.graph_provider,
            block_offset=outer_offset,
            block_shape=outer_shape,
            inner_block_offset=total_offset,
            inner_block_shape=self.config.block_size
        )

        # TODO dynamic data augmentation

        return graph
