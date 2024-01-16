import numpy as np
import pfsspy
import torch
from torch import nn

from nf2.data.util import vector_cartesian_to_spherical, spherical_to_cartesian
from nf2.evaluation.unpack import load_coords


# transform NF2 solution to PFSS output
class NF2Output(pfsspy.Output):

    def __init__(self, nf2_path, input: pfsspy.Input):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        state = torch.load(nf2_path, map_location=device)
        model = nn.DataParallel(state['model'])

        coords = np.stack(np.meshgrid(
            input.grid.pg,
            np.arccos(-input.grid.sg),
            np.exp(input.grid.rg), indexing='ij'), -1)
        cube_shape = coords.shape[:-1]

        input_coords = coords.reshape(-1, 3)
        input_coords = np.stack([input_coords[:, 2], input_coords[:, 1], input_coords[:, 0]], -1)
        input_cartesian_coords = spherical_to_cartesian(input_coords)

        b = load_coords(model, 1, state['b_norm'], input_cartesian_coords, device, progress=True,
                        compute_currents=False)
        # input_coords[:, 1] -= np.pi / 2
        b = vector_cartesian_to_spherical(b, input_coords)
        b[..., 1] *= -1
        b = b.reshape(cube_shape + (3,))
        # b[np.linalg.norm(b, axis=-1) < 1] = 0
        self.b = b

        super().__init__(b[..., 0], b[..., 1], b[..., 2], grid=input.grid, input_map=input.map)

    @property
    def bg(self):
        b = np.stack([self.b[..., 2], self.b[..., 1], self.b[..., 0]], -1) * self.bunit
        b[0] = b[-1]  # assure periodicity - should be small anyways but the check is not working
        return b
