from gunpowder import *
from gunpowder.profiling import Timing
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MergeFragments(BatchFilter):
    """
    Request two patches with centers center_u, center_v, and stack them together, introducing a new
    dimension for each array. This is useful to create batches for a siamese network

    This node only supports batches containing arrays, not points.

    """

    def move_roi(self, request, offset):
        request = request.copy()
        for key, spec in request.array_specs.items():
            spec.roi += offset
            orig_shape = spec.roi.get_shape()
            spec.roi = spec.roi.snap_to_grid(spec.voxel_size, mode='closest')
            spec.roi.set_shape(orig_shape)
        logger.debug(request)
        return request

    def provide(self, request):
        """
        Args:
            request: The request must have the attributes center_u, center_v.
        """

        timing = Timing(self)
        timing.start()

        center = request.get_total_roi().get_center()
        offset_u = request.center_u - center
        offset_v = request.center_v - center

        request_u = self.move_roi(request, offset_u)
        request_v = self.move_roi(request, offset_v)

        batch_u = self.get_upstream_provider().provide(request_u)
        batch_v = self.get_upstream_provider().provide(request_v)

        batch = Batch()
        for key, spec in request.items():
            logger.debug(f'{key}, {batch_u[key].data.shape}')
            logger.debug(f'{key}, {batch_v[key].data.shape}')
            data = np.stack([batch_u[key].data, batch_v[key].data])
            batch[key] = Array(
                data=data,
                spec=request[key].copy())

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
