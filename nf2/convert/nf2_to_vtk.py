import argparse
import os.path
from threading import Thread

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.vtk import save_vtk


class _SaveFileTask(Thread):

    def __init__(self, out_path, output, metrics):
        super().__init__()
        self.out_path = out_path
        self.output = output
        self.metrics = metrics if metrics is not None else []

    def run(self):
        Mm_per_pixel = self.output['Mm_per_pixel']

        # split output into vectors and scalars
        vectors = {k: v for k, v in self.output['metrics'].items() if len(v.shape) == 4 and v.shape[-1] == 3}
        scalars = {k: v for k, v in self.output['metrics'].items() if len(v.shape) == 3}

        vectors['b'] = self.output['b']

        save_vtk(self.out_path, vectors=vectors, scalars=scalars, Mm_per_pix=Mm_per_pixel)


def convert(nf2_path, out_path=None, Mm_per_pixel=None, height_range=None, metrics=None, **kwargs):
    out_path = out_path if out_path is not None \
        else os.path.join(os.path.dirname(nf2_path), nf2_path.split(os.sep)[-2] + '.vtk')

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=Mm_per_pixel, height_range=height_range, metrics=metrics, **kwargs)

    # save file in background
    task = _SaveFileTask(out_path, output, metrics)
    task.start()


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--Mm_per_pixel', type=float, help='spatial resolution (0.36 for original HMI)', required=False,
                        default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)
    parser.add_argument('--metrics', type=str, nargs='*', help='metrics to be computed', required=False, default=['j'])

    args = parser.parse_args()
    nf2_path = args.nf2_path

    Mm_per_pixel = args.Mm_per_pixel
    out_path = args.out_path
    height_range = args.height_range
    metrics = args.metrics

    convert(nf2_path, out_path, Mm_per_pixel, height_range, metrics=metrics, progress=True)


if __name__ == '__main__':
    main()
