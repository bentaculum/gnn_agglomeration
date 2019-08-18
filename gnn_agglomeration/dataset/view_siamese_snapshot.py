from funlib.show.neuroglancer import add_layer, ScalePyramid
import argparse
import daisy
import glob
import neuroglancer
import os
import webbrowser
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file',
    '-f',
    type=str,
    action='append',
    required=True,
    help="The path to the container to show")
parser.add_argument(
    '--datasets',
    '-d',
    type=str,
    nargs='+',
    action='append',
    required=True,
    help="The datasets in the container to show")
parser.add_argument(
    '--no_browser',
    '-n',
    action='store_true',
    help="If set, do not open browser with generated URL")

args = parser.parse_args()

neuroglancer.set_server_bind_address('0.0.0.0')
viewer = neuroglancer.Viewer()

for f, datasets in zip(args.file, args.datasets):

    arrays_doubled = []
    datasets_doubled = []
    for ds in datasets:
        print("Adding %s, %s" % (f, ds))
        a = daisy.open_ds(f, ds)
        arrays_doubled.append(a)
        datasets_doubled.append(ds)
        # a0 = daisy.Array(a.to_ndarray()[0], a.roi, a.voxel_size)
        # arrays_doubled.append(a0)
        # datasets_doubled.append(ds)

        # a1 = daisy.Array(a.to_ndarray()[1], a.roi, a.voxel_size)
        # arrays_doubled.append(a1)
        # datasets_doubled.append(ds)

    with viewer.txn() as s:
        for array, dataset in zip(arrays_doubled, datasets_doubled):
            add_layer(s, array, dataset)

url = str(viewer)
print('\nURL:\n%s\n' % url)

if not args.no_browser:
    webbrowser.open_new(url)

print("Press ENTER to quit")
input()
