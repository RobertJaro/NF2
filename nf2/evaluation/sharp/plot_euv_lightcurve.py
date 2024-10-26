import argparse
import glob
import os
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
from sunpy.map import Map
from tqdm import tqdm

from nf2.evaluation.sharp.convert_series import load_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--pkl_path', type=str, help='path to the directory with the converted pkl files.')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--euv_path', type=str, help='path to the EUV files.', required=False, default=None)
    args = parser.parse_args()

    output = load_results(args.pkl_path)

    euv_files = np.array(sorted(glob.glob(args.euv_path)))
    euv_dates = np.array([parse(fits.getheader(f, 1)['DATE-OBS']) for f in euv_files])

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    times = output['times']
    wcs = output['wcs']

    euv_lightcurve = []
    cond = (times > datetime(2024, 5, 9, 15)) & (times < datetime(2024, 5, 10, 12))
    times = times[cond]
    wcs = wcs[cond]

    for i, time in tqdm(enumerate(times), total=len(times)):
        euv_file = euv_files[np.argmin(np.abs(euv_dates - time))]
        euv_map = Map(euv_file)
        exposure = euv_map.exposure_time.to_value(u.s)
        euv_map = euv_map.reproject_to(wcs[i][0])
        euv_lightcurve.append(euv_map.data.mean() / exposure)

    date_format = DateFormatter('%d-%H:%M')

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(date_format)

    fig.autofmt_xdate()

    ax.plot(times, euv_lightcurve)
    ax.set_ylabel('SDO/AIA 131 $\AA$')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'euv_lightcurve.jpg'), dpi=300)
    plt.close()


# if __name__ == '__main__':
#     main()
