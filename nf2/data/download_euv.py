import argparse
import os

import drms
from dateutil.parser import parse

from nf2.data.download import download_euv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--cadence', type=str, required=False, default='60s')
    parser.add_argument('--channel', type=str, required=False, default='131')
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    client = drms.Client(email=(args.email))

    start_time = parse(args.t_start)
    end_time = parse(args.t_end) if args.t_end is not None else None

    download_euv(start_time=start_time, end_time=end_time,
                 cadence=args.cadence, channel=args.channel,
                 dir=args.download_dir, client=client)


if __name__ == '__main__':
    main()  # workaround for entry_points
