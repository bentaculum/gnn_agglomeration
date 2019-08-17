from gunpowder import *
from gunpowder.profiling import Timing
import numpy as np


class MergeFragments(BatchFilter):
    """
    Request two patches with centers center_u, center_v, and stack them together, introducing a new
    dimension for each array. This is useful to create batches for a siamese network

    This node only supports batches containing arrays, not points.

    """

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

        request_u = request.copy()
        for key, spec in request_u.arrays:
            spec.roi += offset_u
        request_v = request.copy()
        for key, spec in request_v.arrays:
            spec.roi += offset_v

        batch_u = self.get_upstream_provider().provide(request_u)
        batch_v = self.get_upstream_provider().provide(request_v)

        batch = Batch()
        for key, spec in request.items():

            data = np.stack([batch_u[key].data, batch_v[key].data])
            batch[key] = Array(
                data,
                request[key].spec.copy())

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch