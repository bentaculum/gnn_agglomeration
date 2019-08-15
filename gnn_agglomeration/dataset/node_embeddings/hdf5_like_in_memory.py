import logging
import numpy as np
from time import time as now

from gunpowder.batch import Batch
from gunpowder.profiling import Timing
from gunpowder.array import Array
from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource

from gunpowder.compat import ensure_str
from gunpowder.coordinate import Coordinate
from gunpowder.ext import ZarrFile

logger = logging.getLogger(__name__)


class Hdf5InMemory(Hdf5LikeSource):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.in_mem_datasets = {}

    def setup(self):
        super().setup()
        with self._open_file(self.filename) as data_file:
            for (array_key, ds_name) in self.datasets.items():

                if ds_name not in data_file:
                    raise RuntimeError("%s not in %s" %
                                       (ds_name, self.filename))

                spec = self._Hdf5LikeSource__read_spec(
                    array_key, data_file, ds_name)
                # logger.info(spec)
                # logger.info(spec.roi)
                # logger.info(spec.roi.get_offset())
                # logger.info((spec.roi - spec.roi.get_offset()) /
                # spec.voxel_size)

                start = now()
                logger.info(
                    f'start loading {ds_name} into memory')
                self.in_mem_datasets[array_key] = self._Hdf5LikeSource__read(
                    data_file,
                    self.datasets[array_key],
                    (spec.roi - spec.roi.get_offset()) / spec.voxel_size,
                )
                logger.info(
                    f'loaded {ds_name} into memory in {now() - start} s')

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for (array_key, request_spec) in request.array_specs.items():
            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = (
                dataset_roi -
                self.spec[array_key].roi.get_offset() / voxel_size
            )

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = Array(
                self.__read(array_key, dataset_roi), array_spec
            )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, array_key, roi):
        in_mem_array = self.in_mem_datasets[array_key]
        c = len(in_mem_array.shape) - self.ndims

        if self.channels_first:
            array = np.asarray(
                in_mem_array[(slice(None),) * c + roi.to_slices()])
        else:
            array = np.asarray(
                in_mem_array[roi.to_slices() + (slice(None),) * c])
            array = np.transpose(
                array, axes=[
                    i + self.ndims for i in range(c)] + list(range(self.ndims))
            )

        return array

    def __repr__(self):

        return self.filename


class InMemZarrSource(Hdf5InMemory):
    '''A `zarr <https://github.com/zarr-developers/zarr>`_ data source.

    Provides arrays from zarr datasets. If the attribute ``resolution`` is set
    in a zarr dataset, it will be used as the array's ``voxel_size``. If the
    attribute ``offset`` is set in a dataset, it will be used as the offset of
    the :class:`Roi` for this array. It is assumed that the offset is given in
    world units.

    Args:

        filename (``string``):

            The zarr directory.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        channels_first (``bool``, optional):

            Specifies the ordering of the dimensions of the HDF5-like data source.
            If channels_first is set (default), then the input shape is expected
            to be (channels, spatial dimensions). This is recommended because of
            better performance. If channels_first is set to false, then the input
            data is read in channels_last manner and converted to channels_first.
    '''

    def _get_voxel_size(self, dataset):

        if 'resolution' not in dataset.attrs:
            return None

        if self.filename.endswith('.n5'):
            return Coordinate(dataset.attrs['resolution'][::-1])
        else:
            return Coordinate(dataset.attrs['resolution'])

    def _get_offset(self, dataset):

        if 'offset' not in dataset.attrs:
            return None

        if self.filename.endswith('.n5'):
            return Coordinate(dataset.attrs['offset'][::-1])
        else:
            return Coordinate(dataset.attrs['offset'])

    def _open_file(self, filename):
        return ZarrFile(ensure_str(filename), mode='r')
