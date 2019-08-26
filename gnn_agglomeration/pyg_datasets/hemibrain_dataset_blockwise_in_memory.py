import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import logging  # noqa
from torch_geometric.data import InMemoryDataset  # noqa
import time  # noqa
import multiprocessing  # noqa
import numpy as np  # noqa

from .hemibrain_dataset_blockwise import HemibrainDatasetBlockwise  # noqa
from .hemibrain_graph_unmasked import HemibrainGraphUnmasked  # noqa
from .hemibrain_graph_masked import HemibrainGraphMasked  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HemibrainDatasetBlockwiseInMemory(
        InMemoryDataset,
        HemibrainDatasetBlockwise):
    def __init__(
            self,
            root,
            config,
            db_name,
            embeddings_collection,
            roi_offset,
            roi_shape,
            length=None,
            save_processed=False):

        HemibrainDatasetBlockwise.__init__(
            self,
            root=root,
            config=config,
            db_name=db_name,
            embeddings_collection=embeddings_collection,
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
        logger.info(
            f'Loading {self.len} graphs with {self.config.num_workers} workers and saving them to {self.root} ...')
        start = time.time()

        pool = multiprocessing.Pool(
            processes=self.config.num_workers,
            initializer=np.random.seed,
            initargs=())
        data_mapresult = pool.map_async(
            func=self.get_from_db,
            iterable=range(self.len))
        pool.close()
        pool.join()

        # strange multiprocessing syntax
        data_list = data_mapresult.get()

        logger.info(f'processed {self.len} in {time.time() - start}s')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process(self):
        InMemoryDataset._process(self)

    def get(self, idx):
        data = InMemoryDataset.get(self, idx)
        if data.num_edges > self.config.max_edges:
            logger.warning(
                f'graph {idx} has {data.num_edges} edges, but the limit is set to {self.config.max_edges}.'
                f'\nDuplicating previous graph')
            logger.info(f'self.len {self.len}')
            return self.get((idx - 1) % self.len)
        else:
            return data
