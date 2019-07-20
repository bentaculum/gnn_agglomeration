import torch
import numpy as np
import logging
import daisy
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from .hemibrain_dataset import HemibrainDataset
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO check whether inheritance order is correct
class HemibrainDatasetRandomInMemory(InMemoryDataset, HemibrainDataset):
    def __init__(
            self,
            root,
            config,
            roi_offset,
            roi_shape,
            length=None,
            save_processed=False):

        HemibrainDataset.__init__(
            self,
            root=root,
            config=config,
            roi_offset=roi_offset,
            roi_shape=roi_shape,
            length=length,
            save_processed=save_processed
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        raise NotImplementedError('Dataset not available for download')

    def process(self):
        logger.info(f'Loading {self.len} graphs and saving them to {self.root} ...')
        data_list = []
        # TODO use multiprocessing here to speed it up
        for i in tqdm(range(self.len)):
            data = self.get_from_db(i)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        InMemoryDataset._process(self)

    def get(self, idx):
        return InMemoryDataset.get(self, idx)

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
        logger.debug(f'get RAG from {daisy.Roi(total_offset, self.config.block_size)}')

        outer_offset, outer_shape = self.pad_block(
            total_offset, self.config.block_size)
        graph = globals()[self.config.graph_type]()

        try:
            graph.read_and_process(
                graph_provider=self.graph_provider,
                block_offset=outer_offset,
                block_shape=outer_shape,
                inner_block_offset=total_offset,
                inner_block_shape=self.config.block_size
            )
            return graph
        except ValueError as e:
            logger.warning(f'{e}, getting graph from another random block')
            return self.get_from_db(idx)

