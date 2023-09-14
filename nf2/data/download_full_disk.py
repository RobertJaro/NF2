import argparse
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_HARP, donwload_ds

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
parser.add_argument('--carrington_rotation', type=int, required=True)
parser.add_argument('--t_start', type=str, required=True)
args = parser.parse_args()


time = parse(args.t_start)

os.makedirs(args.download_dir, exist_ok=True)
client = drms.Client(email=(args.email), verbose=True)

ds = f'hmi.synoptic_mr_polfil_720s[{args.carrington_rotation}]'
donwload_ds(ds, args.download_dir, client)

ds = f'hmi.b_synoptic[{args.carrington_rotation}]{{Bt, Bp}}'
donwload_ds(ds, args.download_dir, client)

ds = f"hmi.B_720s[{time.isoformat('_', timespec='seconds')}]"
donwload_ds(ds, args.download_dir, client, process={'HmiB2ptr': {'l': 1}})
