import argparse
import os.path
from threading import Thread

import h5py

from nf2.evaluation.output import CartesianOutput


class _SaveFileTask(Thread):

    def __init__(self, out_path, output, nf2_out):
        super().__init__()
        self.out_path = out_path
        self.output = output
        self.nf2_out = nf2_out

    def run(self):
        f = h5py.File(self.out_path, 'w')
        f.create_dataset('B', data=self.output['b'], dtype='f4', compression='gzip')
        f.attrs['INFO'] = f'Magnetic vector field (B_x, B_y, B_z): {self.output["b"].shape} (x, y, z, 3)'
        f.attrs['Mm_per_pixel'] = self.output['Mm_per_pixel']
        f.attrs['type'] = self.nf2_out.data_config['type']
        f.attrs['wcs'] = self.nf2_out.wcs[0].to_header_string()
        f.attrs['time'] = self.nf2_out.time.isoformat('T', timespec='seconds')
        f.close()
        print('File saved:', self.out_path)


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.hdf5')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, **kwargs)

    # save file in background
    task = _SaveFileTask(out_path, output, nf2_out)
    task.start()
    return task


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target HDF5 file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path
    height_range = args.height_range

    convert(nf2_path, out_path, Mm_per_pixel, height_range)


if __name__ == '__main__':
    main()
