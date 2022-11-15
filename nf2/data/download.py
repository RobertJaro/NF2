import os


def download_HARP(harpnum, time, dir, client, series='sharp_cea_720s'):
    ds = 'hmi.%s[%d][%s]{Br, Bp, Bt, Br_err, Bp_err, Bt_err}' % (
    series, harpnum, time.isoformat('_', timespec='seconds'))
    donwload_ds(ds, dir, client)


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
