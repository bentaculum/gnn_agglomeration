import logging
import numpy as np
import configargparse
import daisy
from funlib.segment.arrays import replace_values
import zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
from time import time as now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    p = configargparse.ArgParser(
        default_config_files=['config_relabel_fragments_with_embeddings.ini'],
        allow_abbrev=False
    )

    p.add('--config_file', is_config_file=True)
    p.add('--embeddings_path', type=str)
    p.add('--fragments_zarr', type=str)
    p.add('--fragments_ds', type=str)
    p.add('--chunks_size', type=int, action='append',
          help='zarr array chunk size in voxels. Can be 1d if isotropic or explicitly 3d')
    p.add('--num_workers', type=int)
    p.add('--out_ds', type=str)

    config = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return config


def load_embeddings(path):
    data = np.load(path)
    return data['node_ids'].astype(np.uint64), data['embeddings'].astype(np.float32)


def pca(embeddings):
    # mean 0, variance 1
    embeddings = scale(embeddings)
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    logger.info(
        f'PCA explained variance ratio {pca.explained_variance_ratio_}')
    return embeddings_3d


def relabel_block(block, fragments, out_volume, node_ids, embeddings):
    start = now()
    fragments = fragments.to_ndarray(block.write_roi)
    logger.info(f'copy fragments into memory in {now() - start} s')
    logger.debug(f'fragments shape {fragments.shape}')

    out_channels = [np.zeros_like(fragments, dtype=np.uint64)
                    for _ in range(3)]
    logger.debug(f'out_block channel shape {out_channels[0].shape}')

    # force C++ implementation
    for i, c in enumerate(out_channels):
        c = replace_values(
            in_array=fragments,
            out_array=c,
            old_values=node_ids,
            new_values=embeddings[:, i],
            inplace=False
        )

    # present_ids = np.unique(fragments)
    # for i, c in enumerate(out_channels):
    # for j in present_ids:
    # c[fragments == j] = embeddings[np.where(node_ids == j)[0][0], i]

    # TODO we might want to switch to floats again here
    out_block = np.stack(out_channels)
    out_volume[block.write_roi] = out_block


def relabel_fragments(file, ds, chunks_size, out_ds, node_ids, embeddings, num_workers):
    fragments = daisy.open_ds(file, ds)

    if len(chunks_size) == 1:
        chunks_size = [chunks_size[0] for _ in range(3)]
    block_size = daisy.Coordinate(chunks_size) * fragments.voxel_size

    out_volume = daisy.prepare_ds(
        filename=file,
        ds_name=out_ds,
        total_roi=fragments.data_roi,
        voxel_size=fragments.voxel_size,
        dtype=np.float32,
        write_roi=daisy.Roi(offset=(0, 0, 0), shape=block_size),
        num_channels=3
    )

    daisy.run_blockwise(
        total_roi=fragments.data_roi,
        read_roi=daisy.Roi(offset=(0, 0, 0), shape=block_size),
        write_roi=daisy.Roi(offset=(0, 0, 0), shape=block_size),
        process_function=lambda block: relabel_block(
            block=block,
            fragments=fragments,
            out_volume=out_volume,
            node_ids=node_ids,
            embeddings=embeddings
        ),
        fit='shrink',
        num_workers=num_workers,
        read_write_conflict=False
    )


def rgb_ints_to_unit_floats(array):
    pass


def floats_to_rgb_ints(array):
    return (minmax_scale(array, feature_range=(0, 1)) * 255).astype(np.uint64)


if __name__ == "__main__":
    config = parse_args()

    node_ids, embeddings = load_embeddings(path=config.embeddings_path)
    embeddings_3d = pca(embeddings)
    embeddings_3d = floats_to_rgb_ints(embeddings_3d)
    logger.debug(f'embeddings_3d dtype {embeddings_3d.dtype}')

    relabel_fragments(
        file=config.fragments_zarr,
        ds=config.fragments_ds,
        chunks_size=config.chunks_size,
        out_ds=config.out_ds,
        node_ids=node_ids,
        embeddings=embeddings_3d,
        num_workers=config.num_workers
    )
