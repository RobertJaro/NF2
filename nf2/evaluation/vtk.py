from tvtk.api import tvtk, write_data
import argparse

import numpy as np
import torch


from nf2.evaluation.unpack import load_cube


def save_vtk(vec, path, name, Mm_per_pix=720e-3):
    # Unpack
    dim = vec.shape[:-1]
    # Generate the grid
    pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
    # reorder the points and vectors in agreement with VTK
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3)
    pts = pts.reshape((-1, 3))
    vectors = vec.transpose(2, 1, 0, 3)
    vectors = vectors.reshape((-1, 3))
    sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
    sg.point_data.vectors = vectors
    sg.point_data.vectors.name = name
    write_data(sg, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('vtk_path', type=str, help='path to the target VTK file')

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    b = load_cube(args.nf2_path, device, progress=True)
    save_vtk(b, args.vtk_path, 'B')