import os

import numpy as np
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler

from nf2.data.dataset import BatchesDataset, RandomSphericalCoordinateDataset, SphereDataset, SphereSlicesDataset
from nf2.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix
from nf2.train.model import image_to_spherical_matrix


class AzimuthDataModule(LightningDataModule):

    def __init__(self, B_data, height, b_norm, work_directory,
                 batch_size={"boundary": 1e4, "random": 2e4},
                 iterations=1e5, num_workers=None,
                 boundary={"type": "open"},
                 validation_resolution=256,
                 meta_data=None, plot_overview=True, slice=None,
                 plot_settings=[],
                 **kwargs):
        super().__init__()

        # data parameters
        self.spatial_norm = None
        self.height = height
        self.b_norm = b_norm
        self.meta_data = meta_data
        assert boundary['type'] in ['open', 'potential'], 'Unknown boundary type. Implemented types are: open, potential'

        # train parameters
        self.iterations = int(iterations)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        os.makedirs(work_directory, exist_ok=True)

        # load synchronic map
        B_field_map = Map(B_data['B_field'])
        B_az_map = Map(B_data['B_azimuth'])
        B_disamb_map = Map(B_data['B_disambig']) if 'B_disambig' in B_data else None
        B_in_map = Map(B_data['B_inclination'])

        fld = B_field_map.data
        inc = np.deg2rad(B_in_map.data)
        azi = np.deg2rad(B_az_map.data)

        # disambiguate
        if B_disamb_map is not None:
            amb = B_disamb_map.data
            amb_weak = 2
            condition = (amb.astype((int)) >> amb_weak).astype(bool)
            azi[condition] += np.pi

        azi -= np.pi

        spherical_coords = all_coordinates_from_map(B_field_map).transform_to(frames.HeliographicCarrington)
        radius = spherical_coords.radius
        spherical_coords = np.stack([
            radius.value if radius.isscalar else radius.to(u.solRad).value,
            np.pi / 2 + spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        coords = spherical_to_cartesian(spherical_coords)

        pAng = -np.deg2rad(B_field_map.meta['CROTA2'])
        latc, lonc = np.deg2rad(B_field_map.meta['CRLT_OBS']), np.deg2rad(B_field_map.meta['CRLN_OBS'])
        map_coords = all_coordinates_from_map(B_field_map).transform_to(frames.HeliographicCarrington)
        lat, lon = map_coords.lat.to(u.rad).value.transpose(), map_coords.lon.to(u.rad).value.transpose()

        cs_matrix = cartesian_to_spherical_matrix(spherical_coords)
        is_matrix = image_to_spherical_matrix(lon, lat, latc, lonc, pAng=pAng)
        si_matrix = np.linalg.inv(is_matrix)
        transform = np.matmul(si_matrix, cs_matrix)

        b_los = fld * np.cos(inc)
        b_trv = np.abs(fld * np.sin(inc))
        b = np.stack([b_los, b_trv, azi]).transpose()

        # load observer cartesian coordinates
        obs_coords = np.array([
            B_field_map.observer_coordinate.radius.to(u.solRad).value,
            np.pi / 2 + latc,
            lonc,
        ])
        obs_coords = spherical_to_cartesian(obs_coords)

        # define height mapping range
        if 'height_mapping' in B_data:
            height_mapping = B_data['height_mapping']
            min_coords = coords - obs_coords[None, None, :]

            # solve quadratic equation --> find points at min solar radii
            rays_d = coords - obs_coords[None, None, :]
            rays_o = obs_coords[None, None, :]
            a = rays_d.pow(2).sum(-1)
            b = (2 * rays_o * rays_d).sum(-1)
            c = rays_o.pow(2).sum(-1) - height_mapping["min"] ** 2
            dist_far = (-b - np.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

            a = rays_d.pow(2).sum(-1)
            b = (2 * rays_o * rays_d).sum(-1)
            c = rays_o.pow(2).sum(-1) - height_mapping["max"] ** 2
            dist_near = (-b - np.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

            height_range = np.stack([dist_near, dist_far], axis=-1)

        b_slices = [b]
        coords_slices = [coords]
        spherical_coords_slices = [spherical_coords]
        transforms = [transform]
        # observer_slices = [np.ones_like(coords) * obs_coords[None, :]]
        # height_range_slices = [height_range]

        dataset_kwargs = {'latitude_range': [np.nanmin(spherical_coords[..., 1]), np.nanmax(spherical_coords[..., 1])],
                          'longitude_range': [np.nanmin(spherical_coords[..., 2]), np.nanmax(spherical_coords[..., 2])],}
        if slice:
            if slice['frame'] == 'helioprojective':
                bottom_left = SkyCoord(slice['Tx'][0] * u.arcsec, slice['Ty'][0] * u.arcsec,
                                       frame=B_field_map.coordinate_frame)
                top_right = SkyCoord(slice['Tx'][1] * u.arcsec, slice['Ty'][1] * u.arcsec,
                                     frame=B_field_map.coordinate_frame)
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
            for b, c, sc in zip(b_slices, coords_slices, spherical_coords_slices):
                mask = (sc[..., 1] > slice_lat[0]) & \
                       (sc[..., 1] < slice_lat[1]) & \
                       (sc[..., 2] > slice_lon[0]) & \
                       (sc[..., 2] < slice_lon[1])
                b[~mask] = np.nan
                c[~mask] = np.nan
            dataset_kwargs['latitude_range'] = slice_lat
            dataset_kwargs['longitude_range'] = slice_lon
            self.sampling_range = [[1, height], slice_lat, slice_lon]

        if plot_overview:
            for b in b_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(b[..., 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
                axs[0].set_title('$B_{LOS}$')
                axs[0].set_xlabel('Longitude [rad]'), axs[0].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[G]')
                im = axs[1].imshow(b[..., 1].transpose(), vmin=0, vmax=b_norm, cmap='gray', origin='lower')
                axs[1].set_title('$B_{trv}$')
                axs[1].set_xlabel('Longitude [rad]'), axs[1].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[G]')
                axs[2].imshow(b[..., 2].transpose(), vmin=-np.pi, vmax=np.pi, cmap='gray', origin='lower')
                axs[2].set_title('$\phi_B$')
                axs[2].set_xlabel('Longitude [rad]'), axs[2].set_ylabel('Latitude [rad]')
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='[rad]')
                fig.tight_layout()
                wandb.log({"Overview": fig})
                plt.close('all')
            for c in coords_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                fig.tight_layout()
                wandb.log({"Coordinates": fig})
                plt.close('all')
            for c in spherical_coords_slices:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im = axs[0].imshow(c[..., 0].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[0])
                im = axs[1].imshow(c[..., 1].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[1])
                im = axs[2].imshow(c[..., 2].transpose(), origin='lower')
                fig.colorbar(im, ax=axs[2])
                fig.tight_layout()
                wandb.log({"Spherical Coordinates": fig})
                plt.close('all')

        # flatten data
        coords = np.concatenate([c.reshape((-1, 3)) for c in coords_slices]).astype(np.float32)
        transform = np.concatenate([t.reshape((-1, 3, 3)) for t in transforms]).astype(np.float32)
        values = np.concatenate([b.reshape((-1, 3)) for b in b_slices]).astype(np.float32)
        # observer = np.concatenate([o.reshape((-1, 3)) for o in observer_slices]).astype(np.float32)
        # height_range = np.concatenate([h.reshape((-1, 2)) for h in height_range_slices]).astype(np.float32)

        # filter nan entries
        nan_mask = np.all(np.isnan(values), -1) | np.any(np.isnan(coords), -1)
        if nan_mask.sum() > 0:
            print(f'Filtering {nan_mask.sum()} nan entries')
            coords = coords[~nan_mask]
            transform = transform[~nan_mask]
            values = values[~nan_mask]

        # normalize data
        values[..., 0] /= b_norm
        values[..., 1] /= b_norm

        self.cube_shape = {'type': 'spherical', 'height': height, 'shapes': [b.shape[:2] for b in b_slices]}

        # check data
        assert len(coords) == len(transform) == len(values), 'Data length mismatch'
        # prep dataset
        # shuffle data
        r = np.random.permutation(coords.shape[0])
        coords = coords[r]
        transform = transform[r]
        values = values[r]
        # observer = observer[r]
        # height_range = height_range[r]

        # store data to disk
        coords_npy_path = os.path.join(work_directory, 'coords.npy')
        np.save(coords_npy_path, coords.astype(np.float32))
        transform_npy_path = os.path.join(work_directory, 'transform.npy')
        np.save(transform_npy_path, transform.astype(np.float32))
        values_npy_path = os.path.join(work_directory, 'values.npy')
        np.save(values_npy_path, values.astype(np.float32))
        # observer_npy_path = os.path.join(work_directory, 'observer.npy')
        # np.save(observer_npy_path, observer.astype(np.float32))
        # height_range_npy_path = os.path.join(work_directory, 'height_range.npy')
        # np.save(height_range_npy_path, height_range.astype(np.float32))

        batches_path = {'coords': coords_npy_path,
                        'values': values_npy_path,
                        'transform': transform_npy_path,
                        # 'observer': observer_npy_path,
                        # 'height_range': height_range_npy_path
                        }

        boundary_batch_size = int(batch_size['boundary']) if isinstance(batch_size, dict) else int(batch_size)
        random_batch_size = int(batch_size['random']) if isinstance(batch_size, dict) else int(batch_size)

        # create data loaders
        self.dataset = BatchesDataset(batches_path, boundary_batch_size)
        self.random_dataset = RandomSphericalCoordinateDataset([1, height], random_batch_size, **dataset_kwargs)
        self.cube_dataset = SphereDataset([1, height], batch_size=boundary_batch_size, resolution=validation_resolution, **dataset_kwargs)
        # update plot settings with dataset kwargs
        plot_settings = [{**dataset_kwargs, **settings} for settings in plot_settings]
        self.slices_datasets = {settings['name']: SphereSlicesDataset(**settings, batch_size=boundary_batch_size)
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
