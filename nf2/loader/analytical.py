import os

import numpy as np
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.analytical_field import get_analytic_b_field
from nf2.data.dataset import BatchesDataset, RandomCoordinateDataset, CubeDataset
from nf2.data.loader import prep_b_data


class AnalyticDataModule(LightningDataModule):

    def __init__(self, case, height, spatial_norm, b_norm, work_directory, batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None, boundary={"type": "full"}, **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = spatial_norm
        self.height = height
        self.b_norm = b_norm
        self.meta_data = None

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        tau_surfaces = boundary['tau_surfaces'] if boundary['type'] == 'tau' else None
        if case == 1:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi / 4, tau_surfaces=tau_surfaces)
        elif case == 2:
            b_cube = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi * 0.15, tau_surfaces=tau_surfaces,
                                          resolution=[80, 80, 72])
        else:
            raise Exception(f'Invalid CASE {case}. Available cases are: [1, 2]')

        for i in range(b_cube.shape[2]):
            fig, axs = plt.subplots(1, 3, figsize=(6, 2))
            min_max = np.abs(b_cube[..., i, 0]).max()
            axs[0].imshow(b_cube[..., i, 0].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[1].imshow(b_cube[..., i, 1].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            axs[2].imshow(b_cube[..., i, 2].transpose(), vmin=-min_max, vmax=min_max, cmap='gray', origin='lower')
            wandb.log({"Overview": fig})
            plt.close('all')

        if boundary['type'] == "full":
            coord_cube = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1)
            values = [b_cube[:, :, :1].reshape((-1, 3)), b_cube[:, :, -1:].reshape((-1, 3)),
                      b_cube[:, :1, :].reshape((-1, 3)), b_cube[:, -1:, :].reshape((-1, 3)),
                      b_cube[:1, :, :].reshape((-1, 3)), b_cube[-1:, :, :].reshape((-1, 3)), ]
            coords = [coord_cube[:, :, :1].reshape((-1, 3)), coord_cube[:, :, -1:].reshape((-1, 3)),
                      coord_cube[:, :1, :].reshape((-1, 3)), coord_cube[:, -1:, :].reshape((-1, 3)),
                      coord_cube[:1, :, :].reshape((-1, 3)), coord_cube[-1:, :, :].reshape((-1, 3)), ]
            #
            coords = np.concatenate(coords).astype(np.float32)
            values = np.concatenate(values).astype(np.float32)
        elif boundary['type'] == "potential":
            b_cube = b_cube[:, :, 0]
            coords, values, err = prep_b_data(b_cube, np.zeros_like(b_cube), height,
                                              potential_boundary=True,
                                              potential_strides=boundary['strides'])
        elif boundary['type'] == "open":
            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            height_slices = boundary['height_slices'] if 'height_slices' in boundary else 0
            b_cube = b_cube[:, :, height_slices]
            coords = coords[:, :, height_slices]
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
        elif boundary['type'] == "tau":

            # load dataset
            coords = np.stack(np.mgrid[:b_cube.shape[0], :b_cube.shape[1], :b_cube.shape[2]], -1).astype(np.float32)
            ranges = np.zeros((*coords.shape[:3], 2), dtype=np.float32)

            height_mapping = {"z": [h / 2 for h in tau_surfaces],
                              "z_min": [0] * len(tau_surfaces),
                              "z_max": [h for h in tau_surfaces]}
            self.height_mapping = height_mapping
            for i, (z, z_min, z_max) in enumerate(
                    zip(height_mapping["z"], height_mapping["z_min"], height_mapping["z_max"])):
                ranges[:, :, i, 0] = z_min
                ranges[:, :, i, 1] = z_max
                coords[:, :, i, 2] = z
            #
            if boundary['use_LOS']:
                b_cube[:, :, 1:, 0] = None
                b_cube[:, :, 1:, 1] = None
            # flatten data
            coords = coords.reshape((-1, 3)).astype(np.float32)
            values = b_cube.reshape((-1, 3)).astype(np.float32)
            ranges = ranges.reshape((-1, 2)).astype(np.float32)
        else:
            raise Exception(f'Invalid boundary condition: {boundary}. Available options: ["full", "potential", "open"]')

        # normalize data
        values = values / b_norm
        # apply spatial normalization
        coords = coords / spatial_norm

        cube_shape = [*b_cube.shape[:2], height]
        self.cube_shape = cube_shape

        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        values = values[r]
        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords)
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values)
        # create data loaders
        batches_path = {'coords': coords_npy_path, 'values': values_npy_path}
        if boundary['type'] == "tau":  # add ranges
            ranges /= spatial_norm
            ranges = ranges[r]
            ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
            np.save(ranges_npy_path, ranges)
            batches_path['height_ranges'] = ranges_npy_path

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomCoordinateDataset(cube_shape, spatial_norm, random_batch_size)
        self.cube_dataset = CubeDataset(cube_shape, spatial_norm, batch_size=boundary_batch_size)
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=self.iterations))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        return [cube_loader, boundary_loader]
