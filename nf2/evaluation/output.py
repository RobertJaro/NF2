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
from nf2.train.model import VectorPotentialModel, calculate_current_from_jacobian

def current_density(jac_matrix, **kwargs):
    j = calculate_current_from_jacobian(jac_matrix, f=np)
    return j.to(u.G / u.Mm)

def b_nabla_bz(b, jac_matrix, **kwargs):
    # compute B * nabla * Bz
    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]
    dBx_dx = jac_matrix[..., 0, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dx = jac_matrix[..., 1, 0]
    dBy_dy = jac_matrix[..., 1, 1]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dx = jac_matrix[..., 2, 0]
    dBz_dy = jac_matrix[..., 2, 1]
    dBz_dz = jac_matrix[..., 2, 2]

    norm_B = np.linalg.norm(b, axis=-1)

    dnormB_dx = - norm_B ** -3 * (bx * dBx_dx + by * dBy_dx + bz * dBz_dx)
    dnormB_dy = - norm_B ** -3 * (bx * dBx_dy + by * dBy_dy + bz * dBz_dy)
    dnormB_dz = - norm_B ** -3 * (bx * dBx_dz + by * dBy_dz + bz * dBz_dz)

    b_nabla_bz = (bx * dBz_dx + by * dBz_dy + bz * dBz_dz) / norm_B ** 2 + \
                 (bz / norm_B) * (bx * dnormB_dx + by * dnormB_dy + bz * dnormB_dz)
    return b_nabla_bz

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

    def load_coords(self, coords, batch_size=int(2 ** 12), progress=False, compute_jacobian=True, metrics={'j' : current_density}):
        def _load(coords):
            # normalize and to tensor
            coords = torch.tensor(coords / self.spatial_norm, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            model_out = {}
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it) if progress else it
            for k in it:
                self.model.zero_grad()
                coord = coords[k * batch_size: (k + 1) * batch_size]
                coord = coord.to(self.device)
                coord.requires_grad = True
                result = self.model(coord, compute_jacobian=compute_jacobian)
                for k, v in result.items():
                    if k not in model_out:
                        model_out[k] = []
                    model_out[k] += [v.detach().cpu()]

            model_out = {k: torch.cat(v) for k, v in model_out.items()}
            model_out = {k: v.reshape(*coords_shape[:-1], *v.shape[1:]).numpy() for k, v in model_out.items()}

            model_out['b'] = model_out['b'] * self.G_per_dB
            return model_out

        if compute_jacobian or self._requires_grad:
            model_out = _load(coords)
            jac_matrix = model_out['jac_matrix']
            jac_matrix = jac_matrix * self.G_per_dB / self.m_per_ds
            model_out['jac_matrix'] = jac_matrix

            for k, f in metrics.items():
                model_out[k] = f(**model_out)

            return model_out
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
        z_min, z_max = (0, self.max_height / self.Mm_per_ds) if height_range is None else (h / self.Mm_per_ds for h in
                                                                                           height_range)

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        pixel_per_ds = self.Mm_per_ds / Mm_per_pixel

        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds) + 1),
                        np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds) + 1),
                        np.linspace(z_min, z_max, int((z_max - z_min) * pixel_per_ds) + 1), indexing='ij'), -1)

        model_out = self.load_coords(coords, **kwargs)

        return {**model_out, 'coords': coords, 'Mm_per_pixel': Mm_per_pixel}

    def load_boundary(self, Mm_per_pixel=None, **kwargs):
        x_min, x_max = self.coord_range[0]
        y_min, y_max = self.coord_range[1]

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        pixel_per_ds = self.Mm_per_ds / Mm_per_pixel

        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds + 1)),
                        np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds + 1)),
                        np.zeros((1,), dtype=np.float32), indexing='ij'), -1)
        coords = coords[:, :, 0]

        model_out = self.load_coords(coords, **kwargs)

        return {**model_out, 'coords': coords, 'Mm_per_pixel': Mm_per_pixel}

    def load_maps(self, **kwargs):
        model_out = self.load_cube(**kwargs)

        j_map = np.linalg.norm(model_out['j'], axis=-1).sum(axis=-1)
        b_map = np.linalg.norm(model_out['b'], axis=-1).sum(axis=-1)
        energy_map = energy(model_out['b']).sum(axis=-1)
        free_energy_map = get_free_mag_energy(model_out['b']).sum(axis=-1)

        return {'b': Map(b_map, wcs=self.wcs),
                'j': Map(j_map, wcs=self.wcs),
                'energy': Map(energy_map, wcs=self.wcs),
                'free_energy': Map(free_energy_map, wcs=self.wcs)}


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
        self.spatial_norm = (self.m_per_ds / (1 * u.solRad)).to_value(u.dimensionless_unscaled)

    def load_spherical(self, radius_range: u.Quantity = None,
                       latitude_range: u.Quantity = (0, np.pi) * u.rad,
                       longitude_range: u.Quantity = (0, 2 * np.pi) * u.rad,
                       sampling=[100, 180, 360], **kwargs):
        radius_range = radius_range if radius_range is not None else self.radius_range
        spherical_coords = np.stack(
            np.meshgrid(
                np.linspace(radius_range[0].to_value(u.solRad), radius_range[1].to_value(u.solRad), sampling[0]),
                np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), sampling[1]),
                np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), sampling[2]),
                indexing='ij'), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)

        model_out = self.load_coords(cartesian_coords, **kwargs)
        return {**model_out, 'coords': cartesian_coords, 'spherical_coords': spherical_coords}

    def load(self,
             radius_range: u.Quantity = None,
             latitude_range: u.Quantity = (0, np.pi) * u.rad,
             longitude_range: u.Quantity = (0, 2 * np.pi),
             resolution: u.Quantity = 64 * u.pix / u.solRad, nan_value=0, **kwargs):
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

        min_lat, max_lat = latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad)
        min_lon, max_lon = (longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad))

        # only evaluate coordinates in simulation volume
        if min_lat == max_lat:
            lat_cond = np.ones_like(lat_coord, dtype=bool)
        else:
            lat_cond = (lat_coord >= min_lat) & (lat_coord < max_lat)
        if min_lon == max_lon:
            lon_cond = np.ones_like(lon_coord, dtype=bool)
        else:
            lon_cond = (lon_coord >= min_lon) & (lon_coord < max_lon)
            if max_lon > 2 * np.pi:
                lon_cond = lon_cond | ((lon_coord < max_lon - 2 * np.pi) & (lon_coord >= 0))
        rad_cond = (rad_coord >= radius_range[0].to_value(u.solRad)) & (rad_coord < radius_range[1].to_value(u.solRad))
        condition = rad_cond & lat_cond & lon_cond
        sub_coords = coords[condition]

        cube_shape = coords.shape[:-1]
        model_out = self.load_coords(sub_coords, **kwargs)

        sub_b = model_out['b']
        b = np.zeros(cube_shape + (3,)) * sub_b.unit * nan_value
        b[condition] = sub_b
        b_spherical = vector_cartesian_to_spherical(b, spherical_coords)

        sub_j = model_out['j']
        j = np.zeros(cube_shape + (3,)) * sub_j.unit * nan_value
        j[condition] = sub_j

        sub_jac = model_out['jac_matrix']
        jac_matrix = np.zeros(cube_shape + (3, 3)) * sub_jac.unit * nan_value
        jac_matrix[condition] = sub_jac

        return {'b': b, 'b_rtp': b_spherical, 'j': j, 'jac_matrix': jac_matrix, 'coords': coords,
                'spherical_coords': spherical_coords}

    def load_spherical_coords(self, spherical_coords: SkyCoord):
        cartesian_coords, spherical_coords = self._skycoords_to_cartesian(spherical_coords)
        model_out = self.load_coords(cartesian_coords)
        model_out['spherical_coords'] = spherical_coords
        model_out['coords'] = cartesian_coords
        return model_out

    def _skycoords_to_cartesian(self, spherical_coords):
        spherical_coords = spherical_coords.transform_to(frames.HeliographicCarrington)
        r = spherical_coords.radius
        r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
        spherical_coords = np.stack([
            r.to(u.solRad).value,
            np.pi / 2 - spherical_coords.lat.to(u.rad).value,
            spherical_coords.lon.to(u.rad).value,
        ]).transpose()
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        return cartesian_coords, spherical_coords
