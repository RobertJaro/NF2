import os
from copy import copy

import numpy as np
import pfsspy
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BatchesDataset, RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset
from nf2.data.util import spherical_to_cartesian, vector_spherical_to_cartesian, cartesian_to_spherical_matrix


class SphericalDataModule(LightningDataModule):

    def __init__(self, synoptic_files, full_disk_files, height, b_norm, work_directory,
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

        # load synchronic map
        synoptic_r_map = Map(synoptic_files['Br'])
        synoptic_t_map = Map(synoptic_files['Bt'])
        synoptic_p_map = Map(synoptic_files['Bp'])

        synchronic_spherical_coords = all_coordinates_from_map(synoptic_r_map)
        synchronic_spherical_coords = np.stack([
            synchronic_spherical_coords.radius.value,
            np.pi / 2 + synchronic_spherical_coords.lat.to(u.rad).value,
            synchronic_spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        synchronic_coords = spherical_to_cartesian(synchronic_spherical_coords)

        synchronic_b_spherical = np.stack([synoptic_r_map.data, -synoptic_t_map.data, synoptic_p_map.data]).transpose()
        synchronic_b = vector_spherical_to_cartesian(synchronic_b_spherical, synchronic_spherical_coords)
        synchronic_transform = cartesian_to_spherical_matrix(synchronic_spherical_coords)

        # load full disk map
        full_disk_r_map = Map(full_disk_files['Br'])
        full_disk_t_map = Map(full_disk_files['Bt'])
        full_disk_p_map = Map(full_disk_files['Bp'])

        full_disk_spherical_coords = all_coordinates_from_map(full_disk_r_map)
        full_disk_spherical_coords = full_disk_spherical_coords.transform_to(frames.HeliographicCarrington)
        full_disk_spherical_coords = np.stack([
            full_disk_spherical_coords.radius.to(u.solRad).value,
            np.pi / 2 + full_disk_spherical_coords.lat.to(u.rad).value,
            full_disk_spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        full_disk_coords = spherical_to_cartesian(full_disk_spherical_coords)
        full_disk_transform = cartesian_to_spherical_matrix(full_disk_spherical_coords)

        full_disk_b_spherical = np.stack(
            [full_disk_r_map.data, -full_disk_t_map.data, full_disk_p_map.data]).transpose()
        full_disk_b = vector_spherical_to_cartesian(full_disk_b_spherical, full_disk_spherical_coords)

        if 'Br_err' in full_disk_files and 'Bt_err' in full_disk_files and 'Bp_err' in full_disk_files:
            full_disk_r_error_map = Map(full_disk_files['Br_err'])
            full_disk_t_error_map = Map(full_disk_files['Bt_err'])
            full_disk_p_error_map = Map(full_disk_files['Bp_err'])
            full_disk_b_error = np.stack([full_disk_r_error_map.data,
                                          full_disk_t_error_map.data,
                                          full_disk_p_error_map.data]).transpose()
        else:
            full_disk_b_error = np.zeros_like(full_disk_b)

        # mask overlap
        synoptic_r_map.meta['date-obs'] = full_disk_r_map.meta['date-obs']  # set constant background
        reprojected_map = full_disk_r_map.reproject_to(synoptic_r_map.wcs)
        mask = ~np.isnan(reprojected_map.data).T
        synchronic_b_spherical[mask] = np.nan
        synchronic_b[mask] = np.nan

        b_spherical_slices = [synchronic_b_spherical, full_disk_b_spherical]
        b_slices = [synchronic_b, full_disk_b]
        error_slices = [np.zeros_like(synchronic_b), full_disk_b_error]
        coords = [synchronic_coords, full_disk_coords]
        spherical_coords = [synchronic_spherical_coords, full_disk_spherical_coords]
        transform = [synchronic_transform, full_disk_transform]

        if boundary['type'] == 'potential':
            source_surface_height = boundary['source_surface_height'] if 'source_surface_height' in boundary else 2.5
            resample = boundary['resample'] if 'resample' in boundary else [360, 180]
            sampling_points = boundary['sampling_points'] if 'sampling_points' in boundary else 100
            assert source_surface_height >= height, 'Source surface height must be greater than height (set source_surface_height to >height)'

            # PFSS extrapolation
            potential_r_map = Map(boundary['Br'])
            potential_r_map = potential_r_map.resample(resample * u.pix)
            pfss_in = pfsspy.Input(potential_r_map, sampling_points, source_surface_height)
            pfss_out = pfsspy.pfss(pfss_in)

            # load B field
            ref_coords = all_coordinates_from_map(potential_r_map)
            spherical_boundary_coords = SkyCoord(lon=ref_coords.lon, lat=ref_coords.lat, radius=height * u.solRad, frame=ref_coords.frame)
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

            b_spherical_slices += [spherical_boundary_values]
            b_slices += [boundary_values]
            error_slices += [np.zeros_like(boundary_values)]
            coords += [boundary_coords]
            spherical_coords += [spherical_boundary_coords]
            transform += [boundary_transform]

        dataset_kwargs = {}
        if slice:
            if slice['frame'] == 'helioprojective':
                bottom_left = SkyCoord(slice['Tx'][0] * u.arcsec, slice['Ty'][0] * u.arcsec,
                                       frame=full_disk_r_map.coordinate_frame)
                top_right = SkyCoord(slice['Tx'][1] * u.arcsec, slice['Ty'][1] * u.arcsec,
                                     frame=full_disk_r_map.coordinate_frame)
                bottom_left = bottom_left.transform_to(frames.HeliographicCarrington)
                top_right = top_right.transform_to(frames.HeliographicCarrington)
                slice_lon = np.array([bottom_left.lon.to(u.rad).value, top_right.lon.to(u.rad).value])
                slice_lat = np.array([bottom_left.lat.to(u.rad).value, top_right.lat.to(u.rad).value]) + np.pi / 2
            elif slice['frame'] == 'heliographic_carrington':
                slice_lon = slice['longitude']
                slice_lat = slice['latitude']
            else:
                raise ValueError(f"Unknown slice type '{slice['type']}'")
            # set values outside lat lon range to nan
            for b, c in zip(b_spherical_slices, spherical_coords):
                mask =  (c[..., 1] > slice_lat[0]) &\
                        (c[..., 1] < slice_lat[1]) &\
                        (c[..., 2] > slice_lon[0]) &\
                        (c[..., 2] < slice_lon[1])
                b[~mask] = np.nan
                c[~mask] = np.nan
            dataset_kwargs['latitude_range'] = slice_lat
            dataset_kwargs['longitude_range'] = slice_lon
            self.sampling_range = [[1, height], slice_lat, slice_lon]

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

        # load dataset
        # assert len(height_mapping['z']) == b_slices.shape[2], 'Invalid height mapping configuration: z must have the same length as the number of slices'
        # for i, h in enumerate(height_mapping['z']):
        #     coords[:, :, i, 2] = h
        # ranges = np.zeros((*coords.shape[:-1], 2))
        # use_height_range = 'z_max' in height_mapping
        # if use_height_range:
        #     z1 = height_mapping['z_max']
        #     # set to lower boundary if not specified
        #     z0 = height_mapping['z_min'] if 'z_min' in height_mapping else np.zeros_like(z1)
        #     assert len(z0) == len(z1) == len(height_mapping['z']), \
        #         'Invalid height mapping configuration: z_min, z_max and z must have the same length'
        #     for i, (h_min, h_max) in enumerate(zip(z0, z1)):
        #         ranges[:, :, i, 0] = h_min
        #         ranges[:, :, i, 1] = h_max

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transform]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_spherical_slices]).astype(np.float32)
        errors = np.concatenate([e.reshape((-1, 3)) for e in error_slices]).astype(np.float32)
        # ranges = ranges.reshape((-1, 2)).astype(np.float32)


        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]
            errors = errors[~nan_mask]
            # ranges = ranges[~nan_mask]

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
        # ranges = ranges[r]

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

        # add height ranges if provided
        # if use_height_range:
        #     ranges_npy_path = os.path.join(work_directory, 'ranges.npy')
        #     np.save(ranges_npy_path, ranges)
        #     batches_path['height_ranges'] = ranges_npy_path


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


class SphericalSeriesDataModule(SphericalDataModule):
    def __init__(self, full_disk_files, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.full_disk_files = copy(full_disk_files)
        self.current_files = self.full_disk_files[0]

        super().__init__(full_disk_files=self.full_disk_files[0], *self.args, **self.kwargs)

    def train_dataloader(self):
        if len(self.full_disk_files) == 0:
            return None
        # re-initialize
        print(f"Load next file: {os.path.basename(self.full_disk_files[0]['Br'])}")
        self.current_files = self.full_disk_files[0]
        super().__init__(full_disk_files=self.full_disk_files[0], *self.args, **self.kwargs)
        del self.full_disk_files[0]
        return super().train_dataloader()
