# python
import argparse
import glob
import os

import numpy as np
from astropy import units as u
from tqdm import tqdm

from nf2.evaluation.output_metrics import energy
from nf2.loader.muram import MURaMSimulation
from nf2.potential.potential_field import get_fft_potential_field


class _Loader:

    def __init__(self, simulation, tau=0.1, max_height=50 * u.Mm):
        self.simulation = simulation
        self.pix_height = simulation.get_average_height(tau)
        self.max_height = max_height

    def load(self, iteration):
        snapshot = self.simulation.iterations[iteration]
        scale = list(self.simulation.ds)

        # load max height
        max_height_pix = int((self.max_height / scale[2]).to_value(u.pixel))
        # adjust scale
        dz = int((scale[0] // scale[2]).to_value(u.dimensionless_unscaled))
        scale[2] = scale[0]
        # stack magnetic field
        bx = snapshot.Bx[:, :, self.pix_height:max_height_pix:dz] * np.sqrt(4 * np.pi)
        by = snapshot.By[:, :, self.pix_height:max_height_pix:dz] * np.sqrt(4 * np.pi)
        bz = snapshot.Bz[:, :, self.pix_height:max_height_pix:dz] * np.sqrt(4 * np.pi)
        b = np.stack([bx, by, bz], axis=-1) * u.G
        # compute energies
        b_energy = energy(b)['energy']
        total_energy = b_energy.sum() * (scale[0] * scale[1] * scale[2]) * u.pix ** 3
        potential_field = get_fft_potential_field(bz[:, :, 0], bz.shape[2], scale=1) * u.G
        potential_energy = energy(potential_field)['energy']
        free_energy = (b_energy - potential_energy)
        total_free_energy = free_energy.sum() * (scale[0] * scale[1] * scale[2]) * u.pix ** 3

        # energy profiles
        profile_energy = b_energy.sum((0, 1)) * (scale[0] * scale[1]) * u.pix ** 2
        profile_free_energy = free_energy.sum((0, 1)) * (scale[0] * scale[1]) * u.pix ** 2

        return {'iteration': iteration,
                'time[s]': snapshot.time.to_value(u.s), 'total_energy[erg]': total_energy.to_value(u.erg),
                'total_free_energy[erg]': total_free_energy.to_value(u.erg),
                'profile_energy[erg/cm]': profile_energy.to_value(u.erg / u.cm),
                'profile_free_energy[erg/cm]': profile_free_energy.to_value(u.erg / u.cm),
                'scale[cm/pix]': scale[2].to_value(u.cm / u.pix)}


def main():
    parser = argparse.ArgumentParser(description='Compute total and free magnetic energy from a MURaM snapshot.')
    parser.add_argument('--data_path', type=str, required=True, help='path to the MURaM simulation.')
    parser.add_argument('--out_path', type=str, default='.', help='output path')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    simulation = MURaMSimulation(args.data_path)

    loader = _Loader(simulation)

    times = np.array(simulation.times)
    iterations = list(simulation.iterations.keys())

    # select times every 12 minutes
    selected_times = np.arange(times[0], times[-1], 12 * 60)
    # select iterations closest to selected times
    selected_iterations = []
    for t in selected_times:
        closest_time = np.argmin(np.abs(times - t))
        selected_iterations.append(iterations[closest_time])
    selected_iterations = sorted(list(set(selected_iterations)))

    # filter out existing files
    existing_iterations = [int(os.path.basename(f).split('.')[0]) for f in glob.glob(os.path.join(out_path, '*.npz'))]
    print(f'Skipping {len(existing_iterations)} existing files')
    selected_iterations = [i for i in selected_iterations if i not in existing_iterations]

    # with Pool(16) as p:
    #     for result in tqdm(p.imap(loader.load, iterations), total=len(iterations)):
    for iteration in tqdm(selected_iterations):
        result = loader.load(iteration)
        iteration = result['iteration']
        result_path = os.path.join(out_path, f'{iteration:07d}.npz')
        np.savez(result_path, **result)


if __name__ == '__main__':
    main()
