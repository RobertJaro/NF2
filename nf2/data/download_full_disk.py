import argparse
import glob
import os

import drms
from dateutil.parser import parse

from nf2.data.download import donwload_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--carrington_rotation', type=int, required=False, default=None)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--convert_ptr', action='store_true')
    parser.add_argument('--download_synoptic', action='store_true')
    args = parser.parse_args()

    time = parse(args.t_start)
    t_end = parse(args.t_end) if args.t_end is not None else None

    os.makedirs(args.download_dir, exist_ok=True)
    client = drms.Client(email=args.email)

    # download synoptic data
    if args.download_synoptic:
        assert args.carrington_rotation is not None, "Carrington rotation must be specified to download synoptic data"
        # download corrected synoptic data
        ds = f'hmi.synoptic_mr_polfil_720s[{args.carrington_rotation}]'
        donwload_ds(ds, args.download_dir, client)

        ds = f'hmi.b_synoptic[{args.carrington_rotation}]{{Br, Bt, Bp}}'
        donwload_ds(ds, args.download_dir, client)

    # download full disk data
    segments = '' if args.convert_ptr else '{field, inclination, azimuth, disambig}'
    process = {'HmiB2ptr': {'l': 1}} if args.convert_ptr else None

    if t_end is None:
        ds = f"hmi.B_720s[{time.isoformat('_', timespec='seconds')}]{segments}"
    else:
        ds = f"hmi.B_720s[{time.isoformat('_', timespec='seconds').replace('-', '.')}-{t_end.isoformat('_', timespec='seconds').replace('-', '.')}]{segments}"
    full_disk_dir = os.path.join(args.download_dir, 'full_disk')
    donwload_ds(ds, full_disk_dir, client, process=process)

    [os.remove(f) for f in glob.glob(os.path.join(full_disk_dir, '*lat.fits'))]
    [os.remove(f) for f in glob.glob(os.path.join(full_disk_dir, '*lon.fits'))]


if __name__ == '__main__':
    main()  # workaround for entry_points
