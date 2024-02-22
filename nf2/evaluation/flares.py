import argparse
import glob
import os
from datetime import timedelta
from multiprocessing import Pool

import drms
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.visualization import ImageNormalize, AsinhStretch
from dateutil.parser import parse
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a
from tqdm import tqdm

from nf2.data.download import download_euv
from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.unpack import load_cube, load_B_map


def _calculate_free_energy(nf2_file, z, batch_size):
    b = load_cube(nf2_file, progress=False, z=z, batch_size=batch_size)
    free_me = get_free_mag_energy(b, progress=False)
    return free_me


class _F:
    def __init__(self, ref_wcs):
        self.ref_wcs = ref_wcs

    def func(self, file):
        return Map(file).reproject_to(self.ref_wcs)


def get_integrated_euv_map(euv_files, ref_wcs):
    with Pool(os.cpu_count()) as p:
        reprojected_maps = p.map(_F(ref_wcs).func, euv_files)
    integrated_euv = np.array([m.data for m in reprojected_maps]).sum(0)
    euv_map = Map(integrated_euv, mag_map.meta)
    return euv_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('nf2_path', type=str, help='path to the directory of the NF2 files')
    parser.add_argument('--result_path', type=str, help='path to the output directory', required=False, default=None)
    parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)
    parser.add_argument('--email', type=str, help='email for the DRMS client', required=True)
    parser.add_argument('--flare_classes', nargs='+', type=str, required=False, default=['X', 'M'])
    args = parser.parse_args()

    nf2_files = sorted(glob.glob(os.path.join(args.nf2_path, '*.nf2')))
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    batch_size = int(1e5 * torch.cuda.device_count())

    # load simulation scaling
    Mm_per_pix = .72  # torch.load(nf2_files[0])['Mm_per_pix']
    z_pixels = int(np.ceil(20 / (Mm_per_pix)))  # 20 Mm --> pixels

    # adjust scale to strides
    Mm_per_pix *= args.strides
    cm_per_pix = (Mm_per_pix * 1e8)

    dates = [parse(os.path.basename(nf2_file).split('.')[0][:-4].replace('_', 'T')) for nf2_file in nf2_files]

    flares = Fido.search(a.Time(min(dates), max(dates)),
                         a.hek.EventType("FL"),
                         a.hek.OBS.Observatory == "GOES")["hek"]
    flares = [f for f in flares if f["fl_goescls"][0] in args.flare_classes]

    client = drms.Client(email=args.email, verbose=True)

    for flare in tqdm(flares, desc='Analyse flares'):
        save_path = os.path.join(result_path,
                                 f'flare_{flare["event_peaktime"].datetime.isoformat("_", timespec="seconds")}_{flare["fl_goescls"]}.jpg')
        if os.path.exists(save_path):
            continue

        start_time = flare["event_starttime"].datetime
        end_time = flare["event_endtime"].datetime
        euv_files = download_euv(start_time, end_time, result_path, client=client)

        filter_dates = (np.array(dates) > (start_time - timedelta(minutes=12))) & \
                       (np.array(dates) < (end_time + timedelta(minutes=12)))
        flare_nf2_files = np.array(nf2_files)[filter_dates]

        free_energy_0 = _calculate_free_energy(flare_nf2_files[0], z_pixels, batch_size)
        free_energy_1 = _calculate_free_energy(flare_nf2_files[-1], z_pixels, batch_size)
        released_energy = -np.clip(free_energy_0 - free_energy_1, a_min=None, a_max=0)
        released_energy_map = released_energy.sum(2) * cm_per_pix

        mag_map = load_B_map(flare_nf2_files[0])
        euv_map = get_integrated_euv_map(euv_files, mag_map.wcs)

        plt.figure(figsize=(15, 3))

        plt.subplot(131, projection=mag_map)
        mag_map.plot()
        plt.title('$B_z$')

        plt.subplot(132, projection=euv_map)
        plt.imshow(euv_map.data, origin='lower', cmap='sdoaia94', norm=ImageNormalize(stretch=AsinhStretch(0.005)))
        plt.title('Integrated EUV SDO/AIA 94 $\AA$')
        plt.xlabel('Carrington Longitude')
        plt.ylabel(' ')

        plt.subplot(133, projection=mag_map)
        im = plt.imshow(released_energy_map.T, origin='lower', cmap='jet')
        plt.title(f'Released energy [{released_energy.sum() * cm_per_pix ** 3:.2e} erg]')
        plt.xlabel('Carrington Longitude')
        plt.ylabel(' ')
        plt.colorbar(mappable=im, label='erg/cm$^2$')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        [os.remove(f) for f in euv_files]
