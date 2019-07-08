import sys
from scale_pyramid import create_scale_pyramid

if __name__ == "__main__":

    in_file = sys.argv[1]
    in_ds = sys.argv[2]

    scales = [
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2)
    ]
    # chunk_shape = [125, 125, 125]

    create_scale_pyramid(
        in_file=in_file,
        in_ds_name=in_ds,
        scales=scales,
        chunk_shape=None)
