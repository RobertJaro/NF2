import datetime
import os
import sys

import drms
import numpy as np
import pandas as pd
from tqdm import tqdm

from nf2.data.download import download_HARP, find_HARP

download_dir = sys.argv[1]
flare_list_path = sys.argv[2]
email = sys.argv[2]

flare_offset = datetime.timedelta(hours=1)
os.makedirs(download_dir, exist_ok=True)

goes_list = pd.read_csv(flare_list_path, parse_dates=['start_time', 'peak_time', 'end_time'])
flares = goes_list[(goes_list['primary_verified'] == True)]

flares = flares[flares['goes_class'].str.contains('X') | flares['goes_class'].str.contains('M')]
flares = flares[~pd.isna(flares['noaa_active_region']) & ~pd.isna(flares['x_hpc'])]
flares = flares[np.abs(flares['x_hpc']) < 500]
ars = flares.groupby('noaa_active_region').first()

client = drms.Client(email=email, verbose=True)
for start_time, noaa_nums in tqdm(zip(ars['start_time'], ars['candidate_ars']), total=len(ars)):
    harpnum = find_HARP(start_time, noaa_nums, client)
    if harpnum is None:
        print('Invalid', start_time, noaa_nums)
        continue
    time = (start_time.to_pydatetime() - flare_offset)
    print('Downloading %d at %s' % (harpnum, time.isoformat('_', timespec='seconds')))
    download_HARP(harpnum, time, download_dir, client)

flares = goes_list[(goes_list['primary_verified'] == True)]
flares = flares[flares['goes_class'].str.contains('X')]
flares = flares[~pd.isna(flares['noaa_active_region']) & ~pd.isna(flares['x_hpc'])]
ars = flares.groupby('noaa_active_region').first()
for start_time, noaa_nums in tqdm(zip(ars['start_time'], ars['candidate_ars']), total=len(ars)):
    harpnum = find_HARP(start_time, noaa_nums, client)
    print(noaa_nums, harpnum)