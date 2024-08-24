import numpy as np
from tvtk.api import tvtk, write_data


def save_vtk(path, coords=None, vectors={}, scalars={}, Mm_per_pix=720e-3):
    """Save numpy array as VTK file

    :param vectors: numpy array of the vector field (x, y, z, c)
    :param path: path to the target VTK file
    :param name: label of the vector field (e.g., B)
    :param Mm_per_pix: pixel size in Mm. 360e-3 for original HMI resolution. (default bin2 pixel scale)
    """
    # Unpack
    if len(vectors) > 0:
        dim = list(vectors.values())[0].shape[:-1]
    elif len(scalars) > 0:
        dim = list(scalars.values())[0].shape
    else:
        raise ValueError('No data to save')

    if coords is None:
        # Generate the grid
        pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
        # reorder the points and vectors in agreement with VTK
        pts = pts.transpose(2, 1, 0, 3)
        pts = pts.reshape((-1, 3))
    else:
        pts = coords
        # reorder the points and vectors in agreement with VTK
        pts = pts.transpose(2, 1, 0, 3)
        pts = pts.reshape((-1, 3))


    sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
    i = 0
    for v_name, v in vectors.items():
        v = v.transpose(2, 1, 0, 3)
        v = v.reshape((-1, 3))
        sg.point_data.add_array(v)
        sg.point_data.get_array(i).name = v_name
        sg.point_data.update()
        i += 1
    for s_name, s in scalars.items():
        s = s.transpose(2, 1, 0)
        s = s.reshape((-1))
        sg.point_data.add_array(s)
        sg.point_data.get_array(i).name = s_name
        sg.point_data.update()
        i += 1

    write_data(sg, path)
