import os
import shutil

def download_HARP(harpnum, time, dir, client):
    ds = 'hmi.sharp_cea_720s[%d][%s]{Br, Bp, Bt, Br_err, Bp_err, Bt_err}' % (harpnum,
                                                                             time.isoformat('_', timespec='seconds'))
    donwload_ds(ds, dir, client)

def donwload_ds(ds, dir, client):
    os.makedirs(dir, exist_ok=True)
    r = client.export(ds, method='url-tar', protocol='fits')
    r.wait()
    download_result = r.download(dir)
    for f in download_result.download:
        shutil.unpack_archive(f, dir)
        os.remove(f)


def find_HARP(start_time, noaa_nums, client):
    noaa_nums = [float(num) for num in noaa_nums[1:-1].split(',')]
    ar_mapping = client.query('hmi.Mharp_720s[][%sZ]' % start_time.to_pydatetime().isoformat('_', timespec='seconds'),
                              key=['NOAA_AR', 'HARPNUM'])
    if len(ar_mapping) == 0:
        return None
    for noaa_num in noaa_nums:
        harpnum = ar_mapping[ar_mapping['NOAA_AR'] == noaa_num]['HARPNUM']
        if len(harpnum) > 0:
            return harpnum.iloc[0]
    return None