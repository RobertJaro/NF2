import argparse
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_HARP_series, find_HARP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--harp_num', type=int, required=False, default=None)
    parser.add_argument('--noaa_num', type=int, required=False, default=None)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--cadence', type=str, required=False, default='720s')
    parser.add_argument('--series', type=str, required=False, default='sharp_cea_720s')
    parser.add_argument('--segments', type=str, required=False, default='Br, Bp, Bt, Br_err, Bp_err, Bt_err')
    args = parser.parse_args()

    assert args.harp_num is not None or args.noaa_num is not None, 'Either harp_num or noaa_num must be provided'

    os.makedirs(args.download_dir, exist_ok=True)
    client = drms.Client(email=(args.email), verbose=True)

    t_start = parse(args.t_start)
    t_end = parse(args.t_end) if args.t_end is not None else None

    if args.noaa_num is not None:
        harp_num = find_HARP(t_start, args.noaa_num, client)
    else:
        harp_num = args.harp_num

    download_HARP_series(harp_num=harp_num,
                         t_start=t_start, t_end=t_end, cadence=args.cadence,
                         download_dir=args.download_dir, client=client,
                         series=args.series, segments=args.segments)


if __name__ == '__main__':
    main()
