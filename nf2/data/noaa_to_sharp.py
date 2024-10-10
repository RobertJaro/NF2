import argparse

import drms
from dateutil.parser import parse

from nf2.data.download import find_HARP

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, required=True)
parser.add_argument('--email', type=str, required=True, help='email address for JSOC')
parser.add_argument('--noaa_nums', type=int, nargs='+', required=False, default=None)
args = parser.parse_args()

client = drms.Client(email=args.email, verbose=True)
time = parse(args.time)
harp_numbers = find_HARP(time, args.noaa_nums, client)

# print the HARP numbers
print(f"HARP numbers for {time.isoformat()}: {harp_numbers}")

def main(): # workaround for entry_points
    pass