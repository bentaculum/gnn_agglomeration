import torch
import logging
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from .hemibrain_dataset_blockwise import HemibrainDatasetBlockwise
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked
from .hemibrain_graph_masked import HemibrainGraphMasked

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainDatasetBlockwiseInMemory(InMemoryDataset, HemibrainDatasetBlockwise):
    def __init__(
            self,
            root,
            config,
            roi_offset,
            roi_shape,
            length=None,
            save_processed=False):

        HemibrainDatasetBlockwise.__init__(
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
