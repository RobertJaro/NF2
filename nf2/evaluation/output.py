import numpy as np
import torch
from astropy import units as u, constants
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from sunpy.coordinates import frames
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from nf2.data.util import spherical_to_cartesian, cartesian_to_spherical, vector_cartesian_to_spherical
from nf2.evaluation.energy import get_free_mag_energy
from nf2.evaluation.metric import energy
from nf2.evaluation.output_metrics import metric_mapping, normalize_metric_names
from nf2.train.model import VectorPotentialModel
from nf2.train.transform import HeightRangeTransformModel, AzimuthTransformModel, HeightTransformModel


JACOBIAN_METRICS = {"j", "alpha", "b_nabla_bz", "energy_gradient", "spherical_energy_gradient"}


class BaseOutput:
    """Base evaluator for NF2 checkpoints.

    Users normally construct geometry-specific helpers through :func:`nf2.load`
    instead of instantiating this class directly.
    """

    def __init__(self, checkpoint, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.state = checkpoint if isinstance(checkpoint, dict) else torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        model = self.state['model']
        model = model.to(device)
        self._requires_grad = isinstance(model, VectorPotentialModel) or getattr(model, 'requires_grad_forward', False)
        self.model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        self.device = device
        self.c = constants.c

    @property
    def Gauss_per_dB(self):
        return self.state['data']['Gauss_per_dB'] * u.G

    @property
    def m_per_ds(self):
        return (self.state['data']['Mm_per_ds'] * u.Mm).to(u.m)

    def _sample_tensor(self, coords, compute_jacobian=False):
        """Evaluate the loaded NF2 model without leaving the torch device.

        Inputs and outputs use normalized model units.  This is the common
        low-level path used by both ``load_coords`` and the field-line tracer;
        checkpoint loading and model/device ownership remain in ``BaseOutput``.
        """
        if torch.is_tensor(coords):
            coord = coords.to(device=self.device, dtype=torch.float32)
        else:
            coord = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        coord = coord.reshape(-1, 3).detach()
        requires_grad = self._requires_grad or compute_jacobian
        coord.requires_grad_(requires_grad)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            result = self.model(coord, compute_jacobian=compute_jacobian)
        return {key: value.detach() for key, value in result.items()}

    def _trace_geometry(self):
        raise NotImplementedError

    def _default_trace_config(self):
        from nf2.evaluation.tracing import TraceConfig

        return TraceConfig()

    def trace_field_lines(self, start_coords, metrics=(), trace_config=None, q_method=None):
        """Trace a full batch of model-coordinate seed points in both directions."""
        from nf2.evaluation.tracing import BatchedFieldLineTracer, TraceConfig

        if trace_config is None:
            trace_config = self._default_trace_config()
        elif isinstance(trace_config, dict):
            trace_config = dict(trace_config)
            Mm_per_ds = self.m_per_ds.to_value(u.Mm)
            if "step_size_Mm" in trace_config:
                trace_config["step_size"] = trace_config.pop("step_size_Mm") / Mm_per_ds
            if "min_step_size_Mm" in trace_config:
                trace_config["min_step_size"] = trace_config.pop("min_step_size_Mm") / Mm_per_ds
            if "max_step_size_Mm" in trace_config:
                trace_config["max_step_size"] = trace_config.pop("max_step_size_Mm") / Mm_per_ds
            if "max_length_Mm" in trace_config:
                trace_config["max_length"] = trace_config.pop("max_length_Mm") / Mm_per_ds
            if "q_epsilon_Mm" in trace_config:
                trace_config["q_epsilon"] = trace_config.pop("q_epsilon_Mm") / Mm_per_ds
            if "min_field_G" in trace_config:
                trace_config["min_field_strength"] = trace_config.pop("min_field_G") / self.Gauss_per_dB.to_value(u.G)
            defaults = self._default_trace_config()
            default_values = dict(defaults.__dict__)
            # Let TraceConfig derive adaptive bounds from an overridden initial
            # step unless the caller supplied explicit bounds.
            if "min_step_size" not in trace_config:
                default_values["min_step_size"] = None
            if "max_step_size" not in trace_config:
                default_values["max_step_size"] = None
            trace_config = TraceConfig(**{**default_values, **trace_config})
        tracer = BatchedFieldLineTracer(self, self._trace_geometry(), trace_config)
        return tracer.trace(start_coords, metrics=metrics, q_method=q_method)

    def compute_fieldline_metrics(self, coords, metrics, trace_config=None, q_method=None):
        """Compute field-line quantities at arbitrary normalized model coordinates."""
        requested = set(normalize_metric_names(metrics))
        trace_metrics = set()
        if "squashing_factor" in requested:
            trace_metrics.add("squashing_factor")
        if "twist_number" in requested or "twist" in requested:
            trace_metrics.add("twist_number")
        if "fieldline_length" in requested:
            trace_metrics.add("fieldline_length")
        if "integrated_current_density" in requested:
            trace_metrics.add("integrated_current_density")
        if "fieldline_geometry" in requested:
            trace_metrics.add("fieldline_geometry")
        traced = self.trace_field_lines(coords, trace_metrics, trace_config=trace_config, q_method=q_method)

        output = {}
        if "squashing_factor" in requested:
            output.update({key: traced[key] for key in (
                "squashing_factor", "log10_q", "q_valid", "q_condition_number"
            )})
        if "twist_number" in requested or "twist" in requested:
            output["twist_number"] = traced["twist_number"] * u.dimensionless_unscaled
        if "fieldline_length" in requested:
            output["fieldline_length"] = (traced["fieldline_length"] * self.m_per_ds).to(u.Mm)
        if "integrated_current_density" in requested:
            scale = self.Gauss_per_dB * self.c / (4 * np.pi)
            output["integrated_current_density"] = traced["integrated_current_density"] * scale
        if "fieldline_geometry" in requested:
            output["open"] = traced["open"]
            output["closed"] = traced["closed"]
            output["open_polarity"] = traced["open_polarity"]
            output["footpoint_separation"] = (traced["footpoint_separation"] * self.m_per_ds).to(u.Mm)
            if isinstance(self, CartesianOutput):
                output["apex_height"] = (traced["apex"] * self.m_per_ds).to(u.Mm)
            else:
                output["apex_radius"] = (traced["apex"] * self.m_per_ds).to(u.solRad)
        return output

    def load_coords(self, coords, batch_size=int(2 ** 12), progress=False, compute_jacobian=True, metrics=None,
                    trace_config=None, q_method=None):
        """Evaluate the neural field at normalized model coordinates.

        Parameters
        ----------
        coords:
            Array with final dimension ``(x, y, z)`` in model coordinates.
        batch_size:
            Number of coordinates evaluated per model call.
        progress:
            Show a progress bar.
        compute_jacobian:
            Include the magnetic-field Jacobian in the output.
        metrics:
            Optional metric names from ``nf2.evaluation.output_metrics``.
        """
        batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size
        metrics = normalize_metric_names(metrics)
        fieldline_metric_names = {
            "squashing_factor", "twist_number", "twist", "fieldline_length",
            "integrated_current_density", "fieldline_geometry",
        }
        fieldline_metrics = [name for name in metrics if name in fieldline_metric_names]
        point_metrics = [name for name in metrics if name not in fieldline_metric_names]
        unknown_metrics = [name for name in point_metrics if name not in metric_mapping]
        if unknown_metrics:
            valid_options = ', '.join(sorted(metric_mapping))
            raise ValueError(f"Unknown output metric '{unknown_metrics[0]}'. Valid options: {valid_options}")
        # Exporters can skip a whole-volume Jacobian for field-line-only and
        # algebraic metrics. Derivative-based point metrics always override it.
        compute_jacobian = compute_jacobian or bool(JACOBIAN_METRICS.intersection(point_metrics))

        def _load(coords):
            # normalize and to tensor
            coords = torch.tensor(coords, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            model_out = {}
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it, desc='Load NF2') if progress else it
            for k in it:
                coord = coords[k * batch_size: (k + 1) * batch_size]
                result = self._sample_tensor(coord, compute_jacobian=compute_jacobian)
                for k, v in result.items():
                    if k not in model_out:
                        model_out[k] = []
                    model_out[k] += [v.detach().cpu()]

            model_out = {k: torch.cat(v) for k, v in model_out.items()}
            model_out = {k: v.reshape(*coords_shape[:-1], *v.shape[1:]).numpy() for k, v in model_out.items()}

            model_out['b'] = model_out['b'] * self.Gauss_per_dB
            if 'a' in model_out:
                model_out['a'] = model_out['a'] * self.Gauss_per_dB * self.m_per_ds
            if 'p' in model_out:
                model_out['p'] = model_out['p'] * self.Gauss_per_dB ** 2
            return model_out

        if self._requires_grad or compute_jacobian:
            model_out = _load(coords)

            if compute_jacobian:
                jac_matrix = model_out['jac_matrix']
                jac_matrix = jac_matrix * self.Gauss_per_dB / self.m_per_ds
                model_out['jac_matrix'] = jac_matrix
        else:
            with torch.no_grad():
                model_out = _load(coords)

        state = {**model_out, 'coords': coords}
        metrics_out = {}
        for key in point_metrics:
            metric_out = metric_mapping[key](**state)
            metrics_out.update(metric_out)
            state.update(metric_out)

        if fieldline_metrics:
            effective_trace_config = trace_config
            if progress:
                if effective_trace_config is None:
                    effective_trace_config = {"progress": True}
                elif isinstance(effective_trace_config, dict):
                    effective_trace_config = {**effective_trace_config, "progress": True}
                else:
                    from dataclasses import replace

                    effective_trace_config = replace(effective_trace_config, progress=True)
            fieldline_out = self.compute_fieldline_metrics(
                coords, fieldline_metrics, trace_config=effective_trace_config, q_method=q_method
            )
            metrics_out.update(fieldline_out)

        model_out['metrics'] = metrics_out
        return model_out


class CartesianOutput(BaseOutput):
    """Evaluate Cartesian NF2 extrapolation checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state['data']['type'] == 'cartesian', 'Requires cartesian NF2 data!'

        self.coord_range = self.state['data']['coord_range']
        self.coord_range = self.coord_range[0] if isinstance(self.coord_range, list) else self.coord_range
        self.max_height = self.state['data']['max_height']
        self.ds_per_pixel = self.state['data']['ds_per_pixel']
        self.ds_per_pixel = self.ds_per_pixel[0] if isinstance(self.ds_per_pixel, list) else self.ds_per_pixel
        self.Mm_per_ds = self.state['data']['Mm_per_ds']
        self.Mm_per_pixel = self.ds_per_pixel * self.Mm_per_ds
        self.wcs = [wcs for wcs in self.state['data']['wcs'] if wcs is not None] if 'wcs' in self.state[
            'data'] else None
        self.time = None if self.wcs is None or len(self.wcs) == 0 else parse(self.wcs[0].wcs.dateobs)
        self.data_config = self.state['data']

    def _trace_geometry(self):
        from nf2.evaluation.tracing import CartesianTraceGeometry

        return CartesianTraceGeometry([
            self.coord_range[0], self.coord_range[1], (0, self.max_height / self.Mm_per_ds)
        ])

    def _default_trace_config(self):
        from nf2.evaluation.tracing import TraceConfig

        step_size = float(self.ds_per_pixel) / 4
        return TraceConfig(step_size=step_size)

    def load_cube(self, height_range=None, x_range=None, y_range=None, Mm_per_pixel=None, **kwargs):
        """Load a regularly sampled Cartesian volume.

        Ranges are specified in megameters. Additional keyword arguments are
        forwarded to :meth:`BaseOutput.load_coords`.
        """
        x_min, x_max = self.coord_range[0] if x_range is None else np.array(x_range) / self.Mm_per_ds
        y_min, y_max = self.coord_range[1] if y_range is None else np.array(y_range) / self.Mm_per_ds
        z_min, z_max = (0, self.max_height / self.Mm_per_ds) if height_range is None \
            else (h / self.Mm_per_ds for h in height_range)

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        ds_per_pixel = Mm_per_pixel / self.Mm_per_ds

        n_x_pix = np.round((x_max - x_min) / ds_per_pixel).astype(int)
        n_y_pix = np.round((y_max - y_min) / ds_per_pixel).astype(int)
        n_z_pix = np.round((z_max - z_min) / ds_per_pixel).astype(int)

        coords = np.stack(np.mgrid[:n_x_pix, :n_y_pix, :n_z_pix], -1)
        coords = coords * ds_per_pixel + np.array([x_min, y_min, z_min]).reshape((1, 1, 1, 3))

        model_out = self.load_coords(coords, **kwargs)

        coords_Mm = coords / ds_per_pixel * Mm_per_pixel
        return {**model_out, 'coords': coords_Mm, 'Mm_per_pixel': Mm_per_pixel}

    def load_slice(self, z=0 * u.Mm, Mm_per_pixel=None, **kwargs):
        """Load one horizontal Cartesian slice at height ``z``."""
        x_min, x_max = self.coord_range[0]
        y_min, y_max = self.coord_range[1]

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        ds_per_pixel = Mm_per_pixel / self.Mm_per_ds

        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, np.round((x_max - x_min) / ds_per_pixel + 1).astype(int)),
                        np.linspace(y_min, y_max, np.round((y_max - y_min) / ds_per_pixel + 1).astype(int)),
                        np.ones((1,), dtype=np.float32) * z.to_value(u.Mm) / self.Mm_per_ds, indexing='ij'), -1)
        coords = coords[:, :, 0]

        model_out = self.load_coords(coords, **kwargs)

        return {**model_out, 'coords': coords * self.Mm_per_ds, 'Mm_per_pixel': Mm_per_pixel}

    def load_maps(self, **kwargs):
        """Load SunPy maps for integrated field strength, current, and energy."""
        model_out = self.load_cube(**kwargs)

        j_map = np.linalg.norm(model_out['j'], axis=-1).sum(axis=-1)
        b_map = np.linalg.norm(model_out['b'], axis=-1).sum(axis=-1)
        energy_map = energy(model_out['b']).sum(axis=-1)
        free_energy_map = get_free_mag_energy(model_out['b']).sum(axis=-1)

        return {'b': Map(b_map, wcs=self.wcs),
                'j': Map(j_map, wcs=self.wcs),
                'energy': Map(energy_map, wcs=self.wcs),
                'free_energy': Map(free_energy_map, wcs=self.wcs)}

    def trace_bottom(self, Mm_per_pixel=None, **kwargs):
        x_min, x_max = self.coord_range[0]
        y_min, y_max = self.coord_range[1]

        Mm_per_pixel = self.Mm_per_pixel if Mm_per_pixel is None else Mm_per_pixel
        pixel_per_ds = self.Mm_per_ds / Mm_per_pixel

        coords = np.stack(
            np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds + 1)),
                        np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds + 1)),
                        np.zeros((1,), dtype=np.float32), indexing='ij'), -1)
        trace_config = dict(kwargs.pop("trace_config", {}) or {})
        trace_config.setdefault("store_path", False)
        return self.trace(coords, trace_config=trace_config, **kwargs)

    def trace(self, start_coords, metrics=(), trace_config=None, q_method=None, **kwargs):
        """Trace Cartesian field lines using the tensor-native batched tracer."""
        return self.trace_field_lines(start_coords, metrics, trace_config=trace_config, q_method=q_method)


class HeightTransformOutput(CartesianOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = self.state['transforms']
        height_transforms = [t for t in self.transforms
                             if isinstance(t, HeightRangeTransformModel) or isinstance(t, HeightTransformModel)]

        assert len(height_transforms) == 1, 'Requires transform module!'
        self.transform_module = height_transforms[0]

        self.coord_range_list = self.state['data']['coord_range']
        self.height_mapping_list = self.state['data']['height_mapping']
        self.ds_per_pixel_list = self.state['data']['ds_per_pixel']

    def load_height_mapping(self, Mm_per_pixel=None, **kwargs):
        mapping_out = []
        for coord_range, height_mapping, ds_per_pixel in zip(self.coord_range_list, self.height_mapping_list,
                                                             self.ds_per_pixel_list):
            if height_mapping is None:
                continue
            x_min, x_max = coord_range[0]
            y_min, y_max = coord_range[1]
            z = height_mapping['z'] / self.Mm_per_ds

            pixel_per_ds = self.Mm_per_ds / Mm_per_pixel if Mm_per_pixel is not None else 1 / ds_per_pixel
            coords = np.stack(
                np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds)),
                            np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds)),
                            z, indexing='ij'), -1)
            in_tensors = {'coords': coords}
            if 'z_min' in height_mapping and 'z_max' in height_mapping:
                height_range = np.zeros((*coords.shape[:-1], 2), dtype=np.float32)
                height_range[..., 0] = height_mapping['z_min'] / self.Mm_per_ds
                height_range[..., 1] = height_mapping['z_max'] / self.Mm_per_ds
                in_tensors['height_range'] = height_range

            in_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in in_tensors.items()}
            model_out = self.load_transformed_coords(in_tensors, **kwargs)
            entry = {'height': z * self.Mm_per_ds, 'coords': model_out['coords'] * self.Mm_per_ds * u.Mm,
                     'original_coords': coords * self.Mm_per_ds * u.Mm,
                     'Mm_per_pixel': self.Mm_per_ds / pixel_per_ds}
            mapping_out.append(entry)
        return mapping_out

    @torch.no_grad()
    def load_transformed_coords(self, in_tensors, batch_size=int(2 ** 12), progress=False):
        cube_shape = list(in_tensors.values())[0].shape[:-1]
        flattened_tensors = {k: v.reshape((-1, v.shape[-1])) for k, v in in_tensors.items()}

        cube = {}
        it = range(int(np.ceil(list(flattened_tensors.values())[0].shape[0] / batch_size)))
        it = tqdm(it) if progress else it
        for i in it:
            self.transform_module.zero_grad()
            batch = {k: v[i * batch_size: (i + 1) * batch_size].to(self.device) for k, v in flattened_tensors.items()}

            transformed_coords = self.transform_module(batch)

            for k, v in transformed_coords.items():
                if k not in cube:
                    cube[k] = []
                cube[k] += [v.detach().cpu()]

        cube = {k: torch.cat(v).reshape(*cube_shape, -1).numpy() for k, v in cube.items()}

        return cube


class DisambiguationOutput(CartesianOutput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = self.state['transforms']
        disambiguation_transforms = [t for t in self.transforms if isinstance(t, AzimuthTransformModel)]

        assert len(disambiguation_transforms) == 1, 'Requires transform module!'
        self.transform_module = disambiguation_transforms[0]

        self.coord_range_list = self.state['data']['coord_range']
        self.height_mapping_list = self.state['data']['height_mapping']
        self.ds_per_pixel_list = self.state['data']['ds_per_pixel']

    def load_slice(self, z=0 * u.Mm, coord_range=None, **kwargs):
        disambiguation = []

        for coord_range, ds_per_pixel in zip(self.coord_range_list, self.ds_per_pixel_list):
            x_min, x_max = coord_range[0]
            y_min, y_max = coord_range[1]
            z = z.to_value(u.Mm) / self.Mm_per_ds

            pixel_per_ds = 1 / ds_per_pixel
            coords = np.stack(
                np.meshgrid(np.linspace(x_min, x_max, int((x_max - x_min) * pixel_per_ds)),
                            np.linspace(y_min, y_max, int((y_max - y_min) * pixel_per_ds)),
                            z, indexing='ij'), -1)

            model_out = self.load_transformed_coords(coords, **kwargs)
            entry = {'coords': coords * self.Mm_per_ds * u.Mm, 'flip': model_out['flip']}
            disambiguation.append(entry)
        return disambiguation

    def load_transformed_coords(self, coords, batch_size=int(2 ** 12), progress=False):
        def _load(coords):
            # normalize and to tensor
            coords = torch.tensor(coords, dtype=torch.float32)
            coords_shape = coords.shape
            coords = coords.reshape((-1, 3))

            cube = {}
            it = range(int(np.ceil(coords.shape[0] / batch_size)))
            it = tqdm(it) if progress else it
            for k in it:
                self.transform_module.zero_grad()
                coord = coords[k * batch_size: (k + 1) * batch_size]
                coord = coord.to(self.device)

                transformed_coords = self.transform_module({'coords': coord})

                for k, v in transformed_coords.items():
                    if k not in cube:
                        cube[k] = []
                    cube[k] += [v.detach().cpu()]

            cube = {k: torch.cat(v).reshape(*coords_shape[:-1]).numpy() for k, v in cube.items()}

            return cube

        with torch.no_grad():
            return _load(coords)


class SphericalOutput(BaseOutput):
    """Evaluate spherical NF2 extrapolation checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state['data']['type'] == 'spherical', 'Requires spherical NF2 data!'

        self.radius_range = self.state['data']['radius_range']
        if not hasattr(self.radius_range, 'unit'):
            self.radius_range = self.radius_range * u.solRad

    def _trace_geometry(self):
        from nf2.evaluation.tracing import SphericalTraceGeometry

        scale = (1 * u.solRad / self.m_per_ds).to_value(u.dimensionless_unscaled)
        radii = self.radius_range.to_value(u.solRad) * scale
        return SphericalTraceGeometry(radii)

    def _default_trace_config(self):
        from nf2.evaluation.tracing import TraceConfig

        step_size = (1 * u.Mm / self.m_per_ds).to_value(u.dimensionless_unscaled)
        return TraceConfig(step_size=float(step_size))

    def trace(self, start_coords, metrics=(), trace_config=None, q_method=None, **kwargs):
        """Trace spherical-domain field lines in Cartesian model coordinates."""
        return self.trace_field_lines(start_coords, metrics, trace_config=trace_config, q_method=q_method)

    def load_spherical(self, radius_range: u.Quantity = None,
                       latitude_range: u.Quantity = (-np.pi / 2, np.pi / 2) * u.rad,
                       longitude_range: u.Quantity = (0, 2 * np.pi) * u.rad,
                       sampling=[100, 180, 360], sin_latitude=False, **kwargs):
        """Load a regularly sampled spherical volume.

        ``sampling`` is ordered as radius, latitude, longitude. Ranges should
        use Astropy units.
        """
        radius_range = radius_range if radius_range is not None else self.radius_range
        colatitude_range = sorted(np.pi / 2 * u.rad - latitude_range)
        if sin_latitude:
            latitude_bounds = np.sort(u.Quantity(latitude_range).to_value(u.rad))
            sin_latitude_edges = np.linspace(
                np.sin(latitude_bounds[1]), np.sin(latitude_bounds[0]), sampling[1] + 1
            )
            sin_latitude_values = (sin_latitude_edges[:-1] + sin_latitude_edges[1:]) / 2
            colatitude_values = np.pi / 2 - np.arcsin(sin_latitude_values)
        else:
            colatitude_values = np.linspace(
                colatitude_range[0].to_value(u.rad), colatitude_range[1].to_value(u.rad), sampling[1]
            )
        spherical_coords = np.stack(
            np.meshgrid(
                np.linspace(radius_range[0].to_value(u.solRad), radius_range[1].to_value(u.solRad), sampling[0]),
                colatitude_values,
                np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), sampling[2]),
                indexing='ij'), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        scaled_coords = cartesian_coords * (1 * u.solRad / self.m_per_ds).to_value(u.dimensionless_unscaled)
        model_out = self.load_coords(scaled_coords, **kwargs)
        model_out['b_rtp'] = vector_cartesian_to_spherical(model_out['b'], spherical_coords)
        return {**model_out, 'coords': cartesian_coords, 'spherical_coords': spherical_coords}

    def load_spherical_layer(self, radius=1 * u.solRad,
                             latitude_range=(-np.pi / 2, np.pi / 2) * u.rad,
                             longitude_range=(0, 2 * np.pi) * u.rad,
                             sampling=(180, 360), sin_latitude=False, **kwargs):
        """Evaluate one spherical layer, including optional field-line metrics."""
        result = self.load_spherical(
            radius_range=u.Quantity([radius.to_value(u.solRad), radius.to_value(u.solRad)], u.solRad),
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            sampling=[1, int(sampling[0]), int(sampling[1])],
            sin_latitude=sin_latitude,
            **kwargs,
        )
        squeezed = {}
        for key, value in result.items():
            if key == "metrics":
                squeezed[key] = {name: values[0] for name, values in value.items()}
            elif hasattr(value, "shape") and value.shape[:1] == (1,):
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed

    def load(self,
             radius_range: u.Quantity = None,
             latitude_range: u.Quantity = (-np.pi / 2, np.pi / 2) * u.rad,
             longitude_range: u.Quantity = (0, 2 * np.pi),
             resolution: u.Quantity = 64 * u.pix / u.solRad, nan_value=0, **kwargs):
        """Load a Cartesian cube covering a spherical shell selection."""
        radius_range = radius_range if radius_range is not None else self.radius_range

        # convert latitude to colatitude
        latitude_range = sorted(np.pi / 2 * u.rad - latitude_range)

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
        colatitude_coord = spherical_coords[..., 1]
        lon_coord = (spherical_coords[..., 2] % (2 * np.pi))
        rad_coord = spherical_coords[..., 0]

        min_colatitude, max_colatitude = latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad)
        min_lon, max_lon = (longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad))

        # only evaluate coordinates in simulation volume
        if min_colatitude == max_colatitude:
            lat_cond = np.ones_like(colatitude_coord, dtype=bool)
        else:
            lat_cond = (colatitude_coord >= min_colatitude) & (colatitude_coord < max_colatitude)
        if min_lon == max_lon:
            lon_cond = np.ones_like(lon_coord, dtype=bool)
        else:
            lon_cond = (lon_coord >= min_lon) & (lon_coord < max_lon)
            if max_lon > 2 * np.pi:
                lon_cond = lon_cond | ((lon_coord < max_lon - 2 * np.pi) & (lon_coord >= 0))
        rad_cond = (rad_coord >= radius_range[0].to_value(u.solRad)) & (rad_coord < radius_range[1].to_value(u.solRad))
        condition = rad_cond & lat_cond & lon_cond

        scaled_coords = coords * (1 * u.solRad / self.m_per_ds).to_value(u.dimensionless_unscaled)
        sub_coords = scaled_coords[condition]

        cube_shape = scaled_coords.shape[:-1]
        model_out = self.load_coords(sub_coords, **kwargs)

        spherical_out = {'spherical_coords': spherical_coords, 'coords': coords, 'metrics': {}}
        metrics = model_out.pop('metrics')
        for k, sub_v in metrics.items():
            volume = np.ones(cube_shape + sub_v.shape[1:]) * nan_value
            if hasattr(sub_v, 'unit'):  # preserve units
                volume = volume * sub_v.unit
            volume[condition] = sub_v
            spherical_out['metrics'][k] = volume

        for k, sub_v in model_out.items():
            volume = np.ones(cube_shape + sub_v.shape[1:]) * nan_value
            if hasattr(sub_v, 'unit'):  # preserve units
                volume = volume * sub_v.unit
            volume[condition] = sub_v
            spherical_out[k] = volume

        spherical_out['b_rtp'] = vector_cartesian_to_spherical(spherical_out['b'], spherical_coords)

        return spherical_out

    def load_spherical_coords(self, spherical_coords: SkyCoord, **kwargs):
        """Evaluate the model at explicit SkyCoord positions."""
        cartesian_coords, spherical_coords = self._skycoords_to_cartesian(spherical_coords)
        scaled_coords = cartesian_coords * (1 * u.solRad / self.m_per_ds).to_value(u.dimensionless_unscaled)
        model_out = self.load_coords(scaled_coords, **kwargs)
        model_out['b_rtp'] = vector_cartesian_to_spherical(model_out['b'], spherical_coords)
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
        ], -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        return cartesian_coords, spherical_coords
