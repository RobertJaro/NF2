import os


def download_HARP(harpnum, time, dir, client, series='sharp_cea_720s', download_error=True):
    segments = 'Br, Bp, Bt, Br_err, Bp_err, Bt_err' if download_error else 'Br, Bp, Bt'
    ds = 'hmi.%s[%d][%s]{%s}' % (
    series, harpnum, time.isoformat('_', timespec='seconds'), segments)
    donwload_ds(ds, dir, client)

def download_HARP_series(harpnum, t_start, duration, download_dir, client, series='sharp_cea_720s', download_error=True):
    segments = 'Br, Bp, Bt, Br_err, Bp_err, Bt_err' if download_error else 'Br, Bp, Bt'
    ds = 'hmi.%s[%d][%s/%s]{%s}' % \
         (series, harpnum, t_start.isoformat('_', timespec='seconds'), duration, segments)
    donwload_ds(ds, download_dir, client)

def donwload_ds(ds, dir, client):
    os.makedirs(dir, exist_ok=True)
    r = client.export(ds, protocol='fits')
    r.wait()
    download_result = r.download(dir)
    return download_result


def find_HARP(start_time, noaa_nums, client):
    ar_mapping = client.query('hmi.Mharp_720s[][%sZ]' % start_time.isoformat('_', timespec='seconds'),
                              key=['NOAA_AR', 'HARPNUM'])
    if len(ar_mapping) == 0:
        return None
    for noaa_num in noaa_nums:
        harpnum = ar_mapping[ar_mapping['NOAA_AR'] == int(noaa_num)]['HARPNUM']
        if len(harpnum) > 0:
            return harpnum.iloc[0]
    return None


def download_euv(start_time, end_time, dir, client):
    ds = f'aia.lev1_euv_12s[{start_time.isoformat("_", timespec="seconds")} / {(end_time - start_time).total_seconds()}s@60s][94]{{image}}'
    euv_files = donwload_ds(ds, dir, client).download
    return euv_files