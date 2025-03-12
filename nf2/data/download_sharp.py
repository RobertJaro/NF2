import argparse
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_SHARP_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--sharp_num', type=int, required=False, default=None)
    parser.add_argument('--noaa_num', type=int, required=False, default=None)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--series', type=str, required=False, default='sharp_cea_720s')
    parser.add_argument('--segments', type=str, required=False, default='Br, Bp, Bt, Br_err, Bp_err, Bt_err')
    args = parser.parse_args()

    sharp_num = args.sharp_num
    noaa_num = args.noaa_num
    t_start = args.t_start

    segments = args.segments
    series = args.series

    download_dir = args.download_dir
    email = args.email

    t_start = parse(t_start)

    download_SHARP_series(download_dir=download_dir,
                          email=email,
                          t_start=t_start,
                          noaa_num=noaa_num,
                          sharp_num=sharp_num,
                          segments=segments,
                          series=series)


if __name__ == '__main__':
    main()  # workaround for entry_points
