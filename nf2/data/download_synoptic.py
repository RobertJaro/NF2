import argparse
import glob
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_HARP, donwload_ds

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
parser.add_argument('--carrington_rotation_start', type=int, required=True)
parser.add_argument('--carrington_rotation_end', type=int, required=True)
args = parser.parse_args()

os.makedirs(args.download_dir, exist_ok=True)
client = drms.Client(email=(args.email), verbose=True)

for carrington_rotation in range(args.carrington_rotation_start, args.carrington_rotation_end + 1):
    # download corrected synoptic data
    ds = f'hmi.synoptic_mr_polfil_720s[{carrington_rotation}]'
    donwload_ds(ds, args.download_dir, client)

    ds = f'hmi.b_synoptic[{carrington_rotation}]{{Br, Bt, Bp}}'
    donwload_ds(ds, args.download_dir, client)