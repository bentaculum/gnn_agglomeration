import numpy as np
import logging
import daisy

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        logger.debug(
            f'get graph {idx} from {daisy.Roi(total_offset, self.config.block_size)}')

        outer_offset, outer_shape = self.pad_block(
            total_offset, self.config.block_size)
        graph = globals()[self.config.graph_type](config=self.config)

        try:
            graph.read_and_process(
                graph_provider=self.graph_provider,
                embeddings=self.embeddings,
                all_nodes=self.all_nodes,
                block_offset=outer_offset,
                block_shape=outer_shape,
                inner_block_offset=total_offset,
                inner_block_shape=self.config.block_size
            )
            return graph
        except ValueError as e:
            logger.warning(f'{e}, getting graph from another random block')
            return self.get_from_db(idx)
