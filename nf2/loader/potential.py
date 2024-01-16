import os

import numpy as np
import pfsspy
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BatchesDataset, RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset
from nf2.data.util import vector_spherical_to_cartesian, spherical_to_cartesian, cartesian_to_spherical_matrix


class PotentialTestDataModule(LightningDataModule):

    def __init__(self, synoptic_files, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 height_mapping={'z': [0]}, boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.height_mapping = height_mapping
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)


        source_surface_height = boundary['source_surface_height'] if 'source_surface_height' in boundary else 2.5
        resample = boundary['resample'] if 'resample' in boundary else [360 * 2, 180 * 2]
        sampling_points = boundary['sampling_points'] if 'sampling_points' in boundary else 100
        assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

        # PFSS extrapolation
        potential_r_map = Map(synoptic_files['Br'])
        potential_r_map = potential_r_map.resample(resample * u.pix)
        potential_r_map.data[np.isnan(potential_r_map.data)] = 0
        pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
        pfss_out = pfsspy.pfss(pfss_in)

        # load B field
        ref_coords = all_coordinates_from_map(potential_r_map)
        spherical_boundary_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=1 * u.solRad, frame=ref_coords.frame)
        potential_shape = spherical_boundary_coords.shape # required workaround for pfsspy spherical reshape
        spherical_boundary_values = pfss_out.get_bvec(spherical_boundary_coords.reshape((-1,)))
        spherical_boundary_values = spherical_boundary_values.reshape((*potential_shape, 3)).value
        spherical_boundary_values[..., 1] *= -1 # flip B_theta
        spherical_boundary_values = np.stack([spherical_boundary_values[..., 0],
                                              spherical_boundary_values[..., 1],
                                              spherical_boundary_values[..., 2]]).T

        # load coordinates
        spherical_boundary_coords = np.stack([
            spherical_boundary_coords.radius.value,
            np.pi / 2 + spherical_boundary_coords.lat.to(u.rad).value,
            spherical_boundary_coords.lon.to(u.rad).value]).T

        # convert to spherical coordinates
        boundary_values = vector_spherical_to_cartesian(spherical_boundary_values, spherical_boundary_coords)
        boundary_coords = spherical_to_cartesian(spherical_boundary_coords)
        boundary_transform = cartesian_to_spherical_matrix(spherical_boundary_coords)

        b_spherical_slices = [spherical_boundary_values]
        b_slices = [boundary_values]
        error_slices = [np.zeros_like(boundary_values)]
        coords = [boundary_coords]
        spherical_coords = [spherical_boundary_coords]
        transform = [boundary_transform]

        dataset_kwargs = {}

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview": fig})
                plt.close('all')
            for b in b_spherical_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].imshow(b[..., 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[2].imshow(b[..., 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                wandb.log({"Overview Spherical": fig})
                plt.close('all')
            for c in coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_spherical_slices]).astype(np.float32)
        errors = np.concatenate([e.reshape((-1, 3)) for e in error_slices]).astype(np.float32)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            errors = errors[~nan_mask]

        # normalize data
        values = values / b_norm
        errors = errors / b_norm

        self.cube_shape = {'type': 'spherical', 'height': height}

        # check data
        assert len(coords) == len(transform) == len(values) == len(errors), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        errors = errors[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        err_npy_path = os.path.join(work_directory, 'errors.npy')
        np.save(err_npy_path, errors.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        'errors': err_npy_path
                        }

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings)
                                for settings in plot_settings if settings['type'] == 'slices'}
        self.batches_path = batches_path

    def clear(self):
        [os.remove(f) for f in self.batches_path.values()]

    def train_dataloader(self):
        data_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        random_loader = DataLoader(self.random_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                   sampler=RandomSampler(self.dataset, replacement=True, num_samples=len(self.dataset)))
        return {'boundary': data_loader, 'random': random_loader}

    def val_dataloader(self):
        cube_loader = DataLoader(self.cube_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                 shuffle=False)
        boundary_loader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False)
        slices_loaders = [DataLoader(ds, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                     shuffle=False) for ds in self.slices_datasets.values()]
        return boundary_loader, cube_loader, *slices_loaders
