import argparse
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_HARP

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
parser.add_argument('--harpnum', type=int, required=True)
parser.add_argument('--t_start', type=str, required=True)
parser.add_argument('--series', type=str, required=False, default='sharp_cea_720s')
parser.add_argument('--no_error', action='store_false')
args = parser.parse_args()

os.makedirs(args.download_dir, exist_ok=True)
client = drms.Client(email=(args.email), verbose=True)
download_HARP(args.harpnum, parse(args.t_start), args.download_dir, client, args.series, download_error=args.no_error)
