import os

import drms


def download_HARP(harpnum, time, dir, client, series='sharp_cea_720s', segments='Br, Bp, Bt, Br_err, Bp_err, Bt_err'):
    ds = 'hmi.%s[%d][%s]{%s}' % (
        series, harpnum, time.isoformat('_', timespec='seconds'), segments)
    donwload_ds(ds, dir, client)


def download_SHARP_series(download_dir, email, t_start, t_end=None, noaa_num=None, sharp_num=None,
                          cadence='720s', segments='Br, Bp, Bt, Br_err, Bp_err, Bt_err', series='sharp_cea_720s'):
    assert sharp_num is not None or noaa_num is not None, 'Either sharp_num or noaa_num must be provided'
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
    return donwload_ds(ds, download_dir, client)


def donwload_ds(ds, dir, client, process=None):
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


def download_euv(start_time, dir, client, end_time=None, cadence='60s', channel='131'):
    if end_time is None:
        ds = f'aia.lev1_euv_12s[{start_time.isoformat("_", timespec="seconds")}][{channel}]{{image}}'
    else:
        ds = f'aia.lev1_euv_12s[{start_time.isoformat("_", timespec="seconds")} / {(end_time - start_time).total_seconds()}s@{cadence}][{channel}]{{image}}'
    euv_files = donwload_ds(ds, dir, client).download
    return euv_files
