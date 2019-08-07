import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SiameseVgg3d(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, input_size, input_fmaps=1, fmaps=32, output_features=10, downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]):
        """
        TODO
        Args:
            input_size:
            input_fmaps:
            fmaps:
            downsample_factors:
        """
        super(SiameseVgg3d, self).__init__()

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
                    kernel_size=downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = current_size / downsample_factors[i]
            assert np.all((size * downsample_factors[i]) == current_size), \
                "Can not downsample %s by chosen downsample factor" % current_size
            current_size = size

            logging.info(
                "VGG level %d: (%s), %d fmaps",
                i,
                current_size,
                current_fmaps)

        self.features = torch.nn.Sequential(*features)

        fully_connected = [
            torch.nn.Linear(
                int(current_size[0] * current_size[1] * current_size[2] * current_fmaps),
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                output_features)
        ]

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
