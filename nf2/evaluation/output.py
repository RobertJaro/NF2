import numpy as np
import torch
from astropy import units as u, constants
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from nf2.data.util import spherical_to_cartesian, cartesian_to_spherical, vector_cartesian_to_spherical
from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy
from nf2.train.model import VectorPotentialModel, calculate_current


class BaseOutput:

    def __init__(self, checkpoint, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = torch.load(checkpoint, map_location=device)
        model = self.state['model']
        self._requires_grad = isinstance(model, VectorPotentialModel)
        self.model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        self.spatial_norm = 1
        self.device = device
        self.c = constants.c

    @property
    def G_per_dB(self):
        return self.state['data']['G_per_dB'] * u.G

    @property
    def m_per_ds(self):
        return (self.state['data']['Mm_per_ds'] * u.Mm).to(u.m)

    def load_coords(self, coords, batch_size=int(2 ** 12), progress=False, compute_currents=False):
        def _load(coords):
            # normalize and to tensor
            coords = torch.tensor(coords / self.spatial_norm, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            cube = []
            j_cube = []
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it) if progress else it
            for k in it:
                self.model.zero_grad()
                coord = coords[k * batch_size: (k + 1) * batch_size]
                coord = coord.to(self.device)
                coord.requires_grad = True
                result = self.model(coord, compute_jacobian=False)
                b_batch = result['b']
                if compute_currents:
                    j_batch = calculate_current(b_batch, coord)
                    j_cube += [j_batch.detach().cpu()]
                cube += [b_batch.detach().cpu()]

            cube = torch.cat(cube)
            cube = cube.reshape(*coords_shape).numpy()
            b = cube * self.G_per_dB

            model_out = {'B': b}
            if compute_currents:
                j_cube = torch.cat(j_cube)
                j_cube = j_cube.reshape(*coords_shape).numpy()
                j = j_cube * self.G_per_dB / self.m_per_ds * self.c / (4 * np.pi)
                model_out['J'] = j
            return model_out

        if compute_currents or self._requires_grad:
            return _load(coords)
        else:
            with torch.no_grad():
                return _load(coords)


class CartesianOutput(BaseOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state['data']['type'] == 'cartesian', 'Requires spherical NF2 data!'

        self.coord_range = self.state['data']['coord_range']
        self.coord_range = self.coord_range[0] if isinstance(self.coord_range, list) else self.coord_range
        self.max_height = self.state['data']['max_height']
        self.ds_per_pixel = self.state['data']['ds_per_pixel']
        self.ds_per_pixel = self.ds_per_pixel[0] if isinstance(self.ds_per_pixel, list) else self.ds_per_pixel
        self.Mm_per_ds = self.state['data']['Mm_per_ds']
        self.Mm_per_pixel = self.ds_per_pixel * self.Mm_per_ds
        self.wcs = self.state['data']['wcs'] if 'wcs' in self.state['data'] else None

    def load_cube(self, height_range=None, Mm_per_pixel=None, **kwargs):
        x_min, x_max = self.coord_range[0]
        y_min, y_max = self.coord_range[1]
        z_min, z_max = (0, self.max_height / self.Mm_per_ds) if height_range is None else (h / self.Mm_per_ds for h in height_range)

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        pixel_per_ds = self.Mm_per_ds / Mm_per_pixel

        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds)),
                        np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds)),
                        np.linspace(z_min, z_max, int((z_max - z_min) * pixel_per_ds)), indexing='ij'), -1)

        cube_shape = coords.shape[:-1]

        model_out = self.load_coords(coords.reshape(-1, 3), compute_currents=True, **kwargs)

        b = model_out['B'].reshape(cube_shape + (3,))
        j = model_out['J'].reshape(cube_shape + (3,))

        return {'B': b, 'J': j, 'coords': coords, 'Mm_per_pixel': Mm_per_pixel}

    def load_maps(self, **kwargs):
        model_out = self.load_cube(**kwargs)

        j_map = np.linalg.norm(model_out['J'], axis=-1).sum(axis=-1)
        b_map = np.linalg.norm(model_out['B'], axis=-1).sum(axis=-1)
        energy_map = energy(model_out['B']).sum(axis=-1)
        free_energy_map = get_free_mag_energy(model_out['B']).sum(axis=-1)

        return {'B': Map(b_map, wcs=self.wcs),
                'J': Map(j_map, wcs=self.wcs),
                'Energy': Map(energy_map, wcs=self.wcs),
                'FreeEnergy': Map(free_energy_map, wcs=self.wcs)}


class HeightTransformOutput(CartesianOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 'transform_module' in self.state, 'Requires transform module!'
        self.transform_module = self.state['transform_module']

        self.coord_range_list = self.state['data']['coord_range']
        self.height_mapping_list = self.state['data']['height_mapping']
        self.ds_per_pixel_list = self.state['data']['ds_per_pixel']

    def load_height_mapping(self, **kwargs):
        for coord_range, height_mapping, ds_per_pixel in zip(self.coord_range_list, self.height_mapping_list,
                                                             self.ds_per_pixel_list):
            x_min, x_max = coord_range[0]
            y_min, y_max = coord_range[1]
            z = height_mapping['z'] / self.Mm_per_ds

            pixel_per_ds = 1 / ds_per_pixel
            coords = np.stack(
                np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds)),
                            np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds)),
                            z, indexing='ij'), -1)
            height_range = np.zeros((*coords.shape[:-1], 2), dtype=np.float32)
            height_range[..., 0] = height_mapping['z_min'] / self.Mm_per_ds
            height_range[..., 1] = height_mapping['z_max'] / self.Mm_per_ds

            model_out = self.load_transformed_coords(coords, height_range, **kwargs)
            yield {'coords': model_out['coords'] * self.Mm_per_ds * u.Mm,
                   'original_coords': coords * self.Mm_per_ds * u.Mm,
                   'height_range': height_range * self.Mm_per_ds * u.Mm}

    def load_transformed_coords(self, coords, height_range, batch_size=int(2 ** 12), progress=False):
        def _load(coords, height_range):
            # normalize and to tensor
            coords = torch.tensor(coords / self.spatial_norm, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            height_range = torch.tensor(height_range, dtype=torch.float32)
            height_range = height_range.reshape((-1, 2))

            cube = []
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it) if progress else it
            for k in it:
                self.transform_module.zero_grad()
                coord = coords[k * batch_size: (k + 1) * batch_size]
                coord = coord.to(self.device)

                height_range_batch = height_range[k * batch_size: (k + 1) * batch_size]
                height_range_batch = height_range_batch.to(self.device)

                transformed_coords = self.transform_module.transform_coords(coord, height_range_batch)

                cube += [transformed_coords.detach().cpu()]

            cube = torch.cat(cube)
            cube = cube.reshape(*coords_shape).numpy()

            return {'coords': cube}

        with torch.no_grad():
            return _load(coords, height_range)


class SphericalOutput(BaseOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state['data']['type'] == 'spherical', 'Requires spherical NF2 data!'

        self.radius_range = self.state['data']['radius_range']

    def load_spherical(self, radius_range: u.Quantity = None,
                       latitude_range: u.Quantity = (-np.pi / 2, np.pi / 2) * u.rad,
                       longitude_range: u.Quantity = (0, 2 * np.pi),
                       sampling=[100, 180, 360], **kwargs):
        radius_range = radius_range if radius_range is not None else self.radius_range
        latitude_range += np.pi / 2 * u.rad  # transform coordinate frame
        spherical_coords = np.stack(
            np.meshgrid(
                np.linspace(radius_range[0].to_value(u.solRad), radius_range[1].to_value(u.solRad), sampling[0]),
                np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), sampling[1]),
                np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), sampling[2]),
                indexing='ij'), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)

        model_out = self.load_coords(cartesian_coords, compute_currents=True, **kwargs)
        return {'B': model_out['B'], 'coords': cartesian_coords}

    def load(self,
             radius_range: u.Quantity = None,
             latitude_range: u.Quantity = (0, np.pi) * u.rad,
             longitude_range: u.Quantity = (0, 2 * np.pi),
             resolution: u.Quantity = 64 * u.pix / u.solRad, **kwargs):
        radius_range = radius_range if radius_range is not None else self.radius_range
        spherical_bounds = np.stack(
            np.meshgrid(np.linspace(radius_range[0].to_value(u.solRad), radius_range[1].to_value(u.solRad), 50),
                        np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), 50),
                        np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), 50),
                        indexing='ij'), -1)

        cartesian_bounds = spherical_to_cartesian(spherical_bounds)
        x_min, x_max = cartesian_bounds[..., 0].min(), cartesian_bounds[..., 0].max()
        y_min, y_max = cartesian_bounds[..., 1].min(), cartesian_bounds[..., 1].max()
        z_min, z_max = cartesian_bounds[..., 2].min(), cartesian_bounds[..., 2].max()

        res = resolution.to_value(u.pix / u.solRad)
        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * res)),
                        np.linspace(y_min, y_max, int((y_max - y_min) * res)),
                        np.linspace(z_min, z_max, int((z_max - z_min) * res)), indexing='ij'), -1)
        # flipped z axis
        spherical_coords = cartesian_to_spherical(coords)
        lat_coord = (spherical_coords[..., 1] % np.pi)
        lon_coord = (spherical_coords[..., 2] % (2 * np.pi))
        rad_coord = spherical_coords[..., 0]

        min_lat, max_lat = latitude_range[0].to_value(u.rad) % np.pi, latitude_range[1].to_value(u.rad) % np.pi
        min_lon, max_lon = longitude_range[0].to_value(u.rad) % (2 * np.pi), longitude_range[1].to_value(u.rad) % (
                    2 * np.pi)

        # only evaluate coordinates in simulation volume
        if min_lat == max_lat:
            lat_cond = np.ones_like(lat_coord, dtype=bool)
        else:
            lat_cond = (lat_coord >= min_lat) & (lat_coord < max_lat)
        if min_lon == max_lon:
            lon_cond = np.ones_like(lon_coord, dtype=bool)
        else:
            lon_cond = (lon_coord >= min_lon) & (lon_coord < max_lon)
        rad_cond = (rad_coord >= radius_range[0].to_value(u.solRad)) & (rad_coord < radius_range[1].to_value(u.solRad))
        condition =  rad_cond & lat_cond  & lon_cond
        sub_coords = coords[condition]

        cube_shape = coords.shape[:-1]
        model_out = self.load_coords(sub_coords, compute_currents=True, **kwargs)

        sub_b = model_out['B']
        b = np.zeros(cube_shape + (3,)) * sub_b.unit
        b[condition] = sub_b
        b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

        sub_j = model_out['J']
        j = np.zeros(cube_shape + (3,)) * sub_j.unit
        j[condition] = sub_j

        return {'B': b, 'B_rtp': b_spherical, 'J': j, 'coords': coords, 'spherical_coords': spherical_coords}

    def load_spherical_coords(self, spherical_coords: SkyCoord):
        cartesian_coords = self._skycoords_to_cartesian(spherical_coords)

        return self.load_coords(cartesian_coords)

    def _skycoords_to_cartesian(self, spherical_coords):
        spherical_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
        r = spherical_coords.radius
        r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
        spherical_coords = np.stack([
            r.to(u.solRad).value,
            np.pi / 2 + spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        return cartesian_coords
