"""Unified download command for NF2 example data sources."""

from __future__ import annotations

import argparse
import glob
import math
import os

import drms
from dateutil.parser import parse


DEFAULT_SHARP_SEGMENTS = "Br,Bp,Bt,Br_err,Bp_err,Bt_err"
DEFAULT_VECTOR_SEGMENTS = "Br,Bt,Bp"
DEFAULT_MR_POLFIL_SEGMENTS = "Mr_polfil"


def download_SHARP_series(download_dir, email, t_start, t_end=None, noaa_num=None, sharp_num=None,
                          cadence='720s', segments=DEFAULT_SHARP_SEGMENTS, series='sharp_cea_720s'):
    if sharp_num is None and noaa_num is None:
        raise ValueError('Either sharp_num or noaa_num must be provided.')
    os.makedirs(download_dir, exist_ok=True)
    client = drms.Client(email=email)
    if noaa_num is not None:
        sharp_num = find_HARP(t_start, noaa_num, client)
        print(f'Found HARP number {sharp_num} for NOAA number {noaa_num}')
    else:
        sharp_num = sharp_num

    if t_end is None:
        ds = f'hmi.{series}[{sharp_num}][{t_start.isoformat("_", timespec="seconds")}]{{{segments}}}'
    else:
        duration = (t_end - t_start).total_seconds()
        ds = f'hmi.{series}[{sharp_num}][{t_start.isoformat("_", timespec="seconds")}/{duration}s@{cadence}]{{{segments}}}'
    return download_ds(ds, download_dir, client)


def download_ds(ds, dir, client, process=None):
    os.makedirs(dir, exist_ok=True)
    r = client.export(ds, protocol='fits', process=process)
    r.wait()
    download_result = r.download(dir)
    return download_result


def find_HARP(start_time, noaa_num, client):
    ar_mapping = client.query('hmi.Mharp_720s[][%sZ]' % start_time.isoformat('_', timespec='seconds'),
                              key=['NOAA_AR', 'HARPNUM'])
    if len(ar_mapping) == 0:
        return None
    harpnum = ar_mapping[ar_mapping['NOAA_AR'] == int(noaa_num)]['HARPNUM']
    if len(harpnum) > 0:
        return harpnum.iloc[0]
    return None


def carrington_rotation_from_time(time):
    """Return the integer Carrington rotation containing ``time``."""
    from sunpy.coordinates.sun import carrington_rotation_number

    return math.floor(carrington_rotation_number(time))


def download_hmi_sharp(
    download_dir,
    email,
    t_start,
    t_end=None,
    noaa_num=None,
    sharp_num=None,
    cadence='720s',
    segments=DEFAULT_SHARP_SEGMENTS,
    series='sharp_cea_720s',
):
    """Download an HMI SHARP or SHARP CEA vector magnetogram series.

    Parameters mirror the ``nf2-download --source hmi_sharp`` command. Provide
    either ``sharp_num`` or ``noaa_num``.
    """
    return download_SHARP_series(
        download_dir=download_dir,
        email=email,
        t_start=t_start,
        t_end=t_end,
        noaa_num=noaa_num,
        sharp_num=sharp_num,
        cadence=cadence,
        segments=segments,
        series=series,
    )


def download_hmi_synoptic(
    download_dir,
    email,
    carrington_rotation=None,
    carrington_rotation_end=None,
    t_start=None,
    t_end=None,
    segments=DEFAULT_VECTOR_SEGMENTS,
    series='b_synoptic',
    include_mr_polfil=False,
    synoptic_product='vector',
):
    """Download HMI synoptic maps for one or more Carrington rotations.

    Rotations can be provided explicitly or inferred from ``t_start`` and
    optional ``t_end``.
    """
    if carrington_rotation is None:
        if t_start is None:
            raise ValueError('Either carrington_rotation or t_start must be provided.')
        carrington_rotation = carrington_rotation_from_time(t_start)
    if carrington_rotation_end is None and t_end is not None:
        carrington_rotation_end = carrington_rotation_from_time(t_end)

    os.makedirs(download_dir, exist_ok=True)
    client = drms.Client(email=email)
    carrington_rotation_end = carrington_rotation if carrington_rotation_end is None else carrington_rotation_end

    results = []
    for rotation in range(carrington_rotation, carrington_rotation_end + 1):
        if synoptic_product in {'mr_polfil', 'both'} or include_mr_polfil:
            results.append(download_ds(
                f'hmi.synoptic_mr_polfil_720s[{rotation}]{{{DEFAULT_MR_POLFIL_SEGMENTS}}}',
                download_dir,
                client,
            ))
        if synoptic_product in {'vector', 'both'}:
            results.append(download_ds(f'hmi.{series}[{rotation}]{{{segments}}}', download_dir, client))
    return results


def download_hmi_full_disk(
    download_dir,
    email,
    t_start,
    t_end=None,
    cadence='720s',
    series='B_720s',
    segments='field,inclination,azimuth,disambig',
    convert_ptr=True,
    keep_coordinates=False,
):
    """Download HMI full-disk vector data, optionally converted to Br/Bt/Bp.

    Conversion uses the JSOC ``HmiB2ptr`` export process by default.
    """
    os.makedirs(download_dir, exist_ok=True)
    client = drms.Client(email=email)

    if t_end is None:
        time_selector = t_start.isoformat('_', timespec='seconds')
    else:
        duration = (t_end - t_start).total_seconds()
        time_selector = f'{t_start.isoformat("_", timespec="seconds")}/{duration}s@{cadence}'

    process = {'HmiB2ptr': {'l': 1}} if convert_ptr else None
    segment_selector = '' if convert_ptr else f'{{{segments}}}'
    ds = f'hmi.{series}[{time_selector}]{segment_selector}'
    result = download_ds(ds, download_dir, client, process=process)

    if convert_ptr and not keep_coordinates:
        for path in glob.glob(os.path.join(download_dir, '*lat.fits')):
            os.remove(path)
        for path in glob.glob(os.path.join(download_dir, '*lon.fits')):
            os.remove(path)
    return result


def _add_common_args(parser):
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)


def _parse_time(value):
    return parse(value) if value is not None else None


def main():
    parser = argparse.ArgumentParser(description='Download data for NF2 runs.')
    parser.add_argument(
        '--source',
        choices=['hmi_sharp', 'hmi_synoptic', 'hmi_full_disk'],
        default='hmi_sharp',
        help='Download source to use.',
    )
    _add_common_args(parser)

    parser.add_argument('--sharp_num', type=int, default=None)
    parser.add_argument('--noaa_num', type=int, default=None)
    parser.add_argument('--t_start', type=str, default=None)
    parser.add_argument('--t_end', type=str, default=None)
    parser.add_argument('--cadence', type=str, default='720s')
    parser.add_argument('--series', type=str, default=None)
    parser.add_argument('--segments', type=str, default=None)
    parser.add_argument('--carrington_rotation', type=int, default=None)
    parser.add_argument('--carrington_rotation_end', type=int, default=None)
    parser.add_argument(
        '--synoptic_product',
        choices=['vector', 'mr_polfil', 'both'],
        default='vector',
        help='Synoptic product to download for --source hmi_synoptic.',
    )
    parser.add_argument('--include_mr_polfil', action='store_true')
    parser.add_argument('--no_convert_ptr', action='store_true')
    parser.add_argument('--keep_coordinates', action='store_true')
    args = parser.parse_args()

    if args.source == 'hmi_sharp':
        if args.t_start is None:
            parser.error('--t_start is required for --source hmi_sharp')
        return download_hmi_sharp(
            download_dir=args.download_dir,
            email=args.email,
            t_start=_parse_time(args.t_start),
            t_end=_parse_time(args.t_end),
            noaa_num=args.noaa_num,
            sharp_num=args.sharp_num,
            cadence=args.cadence,
            segments=args.segments or DEFAULT_SHARP_SEGMENTS,
            series=args.series or 'sharp_cea_720s',
        )

    if args.source == 'hmi_synoptic':
        if args.carrington_rotation is None and args.t_start is None:
            parser.error('Either --carrington_rotation or --t_start is required for --source hmi_synoptic')
        return download_hmi_synoptic(
            download_dir=args.download_dir,
            email=args.email,
            carrington_rotation=args.carrington_rotation,
            carrington_rotation_end=args.carrington_rotation_end,
            t_start=_parse_time(args.t_start),
            t_end=_parse_time(args.t_end),
            segments=args.segments or DEFAULT_VECTOR_SEGMENTS,
            series=args.series or 'b_synoptic',
            include_mr_polfil=args.include_mr_polfil,
            synoptic_product=args.synoptic_product,
        )

    if args.t_start is None:
        parser.error('--t_start is required for --source hmi_full_disk')
    return download_hmi_full_disk(
        download_dir=args.download_dir,
        email=args.email,
        t_start=_parse_time(args.t_start),
        t_end=_parse_time(args.t_end),
        cadence=args.cadence,
        series=args.series or 'B_720s',
        segments=args.segments or 'field,inclination,azimuth,disambig',
        convert_ptr=not args.no_convert_ptr,
        keep_coordinates=args.keep_coordinates,
    )


if __name__ == '__main__':
    main()
