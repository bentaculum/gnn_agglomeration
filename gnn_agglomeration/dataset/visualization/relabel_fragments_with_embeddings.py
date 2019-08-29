import logging
import numpy as np
import configargparse
import daisy
from funlib.segment.arrays import replace_values
import zarr
from sklearn.decomposition import PCA
from time import time as now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    p.add('--out_ds', type=int)

    return p.parse_args()


def load_embeddings(path):
    data = np.load(path)
    return data['node_ids'].astype(np.uint64), data['embeddings'].astype(np.float32)


def pca(embeddings):
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    logger.info(f'PCA explained variance ratio {pca.explained_variance_ratio_}')
    return embeddings_3d


def relabel_block(block, fragments, out_volume, node_ids, embeddings):
    start = now()
    fragments = fragments.to_ndarray(block.write_roi)
    logger.info(f'copy fragments into memory in {now() - start} s')

    out_channels = [np.zeros_like(fragments, dtype=np.float32) for _ in range(3)]
    out_block = np.stack(out_channels)

    # force C++ implementation
    for i, c in enumerate(out_block):
        c = replace_values(
            in_array=fragments,
            out_array=c,
            old_values=node_ids,
            new_values=embeddings[:, i],
            inplace=False
        )

    out_volume[block.write_roi] = out_block


def relabel_fragments(file, ds, chunks_size, out_ds, node_ids, embeddings, num_workers):
    fragments = daisy.open_ds(file, ds)

    if isinstance(chunks_size, int):
        chunks_size = [chunks_size] * 3
    block_size = daisy.Coordinate(chunks_size)

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


if __name__ == "__main__":
    config = parse_args()

    node_ids, embeddings = load_embeddings(path=config.embeddings_path)
    embeddings_3d = pca(embeddings)

    relabel_fragments(
        file=config.fragments_zarr,
        ds=config.fragments_ds,
        chunks_size=config.chunks_size,
        out_ds=config.out_ds,
        node_ids=node_ids,
        embeddings=embeddings_3d,
        num_workers=config.num_workers
    )
