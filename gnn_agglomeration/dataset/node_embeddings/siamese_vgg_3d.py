import logging
import torch
import numpy as np
import os.path as osp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SiameseVgg3d(torch.nn.Module):
    """
    TODO
    """

    def __init__(self,
                 writer,
                 input_size,
                 input_fmaps=1,
                 fmaps=32,
                 fmaps_max=512,
                 output_features=10,
                 downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]):
        """
        TODO
        Args:
            input_size:
            input_fmaps:
            fmaps:
            downsample_factors:
        """
        super(SiameseVgg3d, self).__init__()

        self.writer = writer

        downsample_factors = np.array(downsample_factors)
        current_fmaps = input_fmaps
        current_size = np.array(input_size)

        features = []
        for i in range(len(downsample_factors)):
            features += [
                torch.nn.Conv3d(
                    in_channels=current_fmaps,
                    out_channels=fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm3d(
                    num_features=fmaps),
                torch.nn.ReLU(
                    inplace=True),
                torch.nn.Conv3d(
                    in_channels=fmaps,
                    out_channels=fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm3d(
                    num_features=fmaps),
                torch.nn.ReLU(
                    inplace=True),
                torch.nn.MaxPool3d(
                    kernel_size=tuple(downsample_factors[i]))
            ]

            current_fmaps = fmaps
            fmaps = min(fmaps_max, fmaps * 2)

            size = np.floor(current_size / downsample_factors[i])
            # TODO come up with a better rule
            # assert np.all((size * downsample_factors[i]) == current_size), \
            #     "Can not downsample %s by chosen downsample factor" % current_size
            current_size = size

            logger.info(
                f'VGG level {i}: ({current_size}), {current_fmaps} fmaps')

        self.features = torch.nn.Sequential(*features)

        num_features = int(
            current_size[0] * current_size[1] * current_size[2] * current_fmaps)
        logger.info(f'inputs to fc: {num_features}')

        fc_size = [2048]
        fully_connected = [
            torch.nn.Linear(
                num_features,
                fc_size[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                fc_size[0],
                output_features)
        ]

        logger.info(f'fc level 0: {fc_size[0]}')
        logger.info(f'output: {output_features}')

        self.fully_connected = torch.nn.Sequential(*fully_connected)

    def forward_once(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.fully_connected(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def train(self, mode=True):
        """
        on top of pytorch's module mode, change the `requires_grad`
        attribute of all module parameters
        Args:
            mode (bool): True = train, False = eval

        Returns:

        """
        ret = super().train(mode=mode)
        for param in self.parameters():
            param.requires_grad = mode
        return ret
