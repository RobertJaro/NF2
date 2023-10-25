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
parser.add_argument('--t_end', type=str, required=False, default=None)
args = parser.parse_args()


time = parse(args.t_start)
t_end = parse(args.t_end) if args.t_end is not None else None

os.makedirs(args.download_dir, exist_ok=True)
client = drms.Client(email=(args.email), verbose=True)

ds = f'hmi.synoptic_mr_polfil_720s[{args.carrington_rotation}]'
donwload_ds(ds, args.download_dir, client)

ds = f'hmi.b_synoptic[{args.carrington_rotation}]{{Bt, Bp}}'
donwload_ds(ds, args.download_dir, client)

if t_end is None:
    ds = f"hmi.B_720s[{time.isoformat('_', timespec='seconds')}]"
else:
    ds = f"hmi.B_720s[{time.isoformat('_', timespec='seconds').replace('-', '.')}-{t_end.isoformat('_', timespec='seconds').replace('-', '.')}]"
full_disk_dir = os.path.join(args.download_dir, 'full_disk')
donwload_ds(ds, full_disk_dir, client, process={'HmiB2ptr': {'l': 1}})
