import logging
import torch
import daisy

# dataset configs for all many params
from ..config import config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SiameseDataset(torch.utils.data.Data):
    """
    Each data point is actually a mini-batch of volume pairs
    """
    def __init__(self, length, patch_size, raw_channel, mask_channel,):
        """

        Args:
            length:
            patch_size:
            raw_channel:
            mask_channel:
        """
        self.len = length
        self.patch_size = patch_size
        self.raw_channel = raw_channel
        self.mask_channel = mask_channel
        assert raw_channel or mask_channel

        # connect to one RAG DB

        # get all nodes
        # get all edges, with gt merge score, as dict of numpy arrays

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """

        Args:
            index(int): not used here, needed for inheritance

        Returns:

            a mini-batch of volume pairs

        """
        # get all neighbors of random node in RAG
        # get the raw data for each neighbor, with daisy, using Nils's synful script

        #
        pass
