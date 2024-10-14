import argparse

import drms
from dateutil.parser import parse

from nf2.data.download import find_HARP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=str, required=True)
    parser.add_argument('--email', type=str, required=True, help='email address for JSOC')
    parser.add_argument('--noaa_nums', type=int, nargs='+', required=False, default=None)
    args = parser.parse_args()

    client = drms.Client(email=args.email)
    time = parse(args.time)
    harp_numbers = {noaa_num: find_HARP(time, noaa_num, client) for noaa_num in args.noaa_nums}

    # print the HARP numbers
    print(f"HARP numbers for {time.isoformat()}:")
    for noaa_num, harp_num in harp_numbers.items():
        print(f"NOAA {noaa_num}: HARP {harp_num}")


if __name__ == '__main__':
    main()