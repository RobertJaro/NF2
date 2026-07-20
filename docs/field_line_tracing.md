# Field-Line Tracing

NF2 can trace magnetic field lines directly through a trained neural field in an
`extrapolation_result.nf2` file. The tracer evaluates the model at every
integration stage; it does not interpolate a previously exported field cube.
This preserves the continuous NF2 representation and lets the same trace
produce topology, geometry, twist, length, and integrated-current diagnostics.

This guide covers:

- Cartesian and spherical tracing domains;
- forward/backward integration and termination;
- Euler and Runge--Kutta methods;
- the squashing factor $Q$ and twist number $T_w$;
- field-line length, integrated current density, and connectivity;
- layers, explicit seed points, paths, full volumes, and exports;
- performance, convergence, and invalid-result handling.

## Quick Start From An NF2 File

Load the checkpoint through the public loader. It selects `CartesianOutput` or
`SphericalOutput` from the checkpoint metadata and loads the model only once.

```python
import nf2

out = nf2.load("extrapolation_result.nf2")
print(type(out).__name__)
```

Request field-line metrics from a sampled layer:

```python
from astropy import units as u

trace_config = {
    "method": "rkf45",
    "step_size_Mm": 0.25,
    "batch_size": 2**14,
    "progress": True,
}

if isinstance(out, nf2.CartesianOutput):
    layer = out.load_slice(
        z=0 * u.Mm,
        Mm_per_pixel=1.0,
        metrics=[
            "squashing_factor",
            "twist_number",
            "fieldline_length",
            "integrated_current_density",
            "fieldline_geometry",
        ],
        trace_config=trace_config,
    )
else:
    layer = out.load_spherical_layer(
        radius=1.0 * u.solRad,
        sampling=(180, 360),
        metrics=[
            "squashing_factor",
            "twist_number",
            "fieldline_length",
            "integrated_current_density",
            "fieldline_geometry",
        ],
        trace_config=trace_config,
    )

metrics = layer["metrics"]
q = metrics["squashing_factor"]
twist = metrics["twist_number"]
length = metrics["fieldline_length"]
```

All requested quantities share the same central field-line trace. Requesting
length, twist, current, and geometry together does not trace four independent
central lines. A perturbed multi-point stencil is used only when $Q$ is
requested with `q_method="perturbed"`.

## What Is Integrated

For a seed position $\mathbf{x}_0$, NF2 integrates in both magnetic-field
directions:

$$
\frac{d\mathbf{x}_{\pm}}{ds} = \pm \hat{\mathbf{b}}(\mathbf{x}_{\pm}),
\qquad
\hat{\mathbf{b}} = \frac{\mathbf{B}}{|\mathbf{B}|}.
$$

Here $s$ is arc length in normalized NF2 model coordinates. The backward
half-line is reversed and joined to the forward half-line, producing one line
ordered as:

```text
backward endpoint -> seed -> forward endpoint
```

Forward and backward work items are interleaved in the same model batches. A
persistent queue replaces terminated work items with remaining seeds, so short
lines do not leave most of a batch idle while long lines finish.

### Integration Methods

`TraceConfig.method` accepts:

| Method | Model stages per step | Typical use |
| --- | ---: | --- |
| `euler` or `rk1` | 1 | Fast preview and debugging. |
| `rk2` | 2 | Faster exploratory maps. |
| `rk3` | 3 | Intermediate accuracy/cost. |
| `rk4` | 4 | Fixed-step reference calculation. |
| `rkf45` | 6 per attempt | Default; adaptive fifth-order solution with embedded fourth-order error control. |

The same method advances coordinates and all requested line integrals. Tangent
$Q$ also advances the infinitesimal deformation matrix with the selected
Runge--Kutta scheme. For `rkf45`, coordinates and every requested Jacobian-based
metric participate in the acceptance test; `rtol` and `atol` control the error,
while `min_step_size` and `max_step_size` bound the adaptive step.

Always test convergence by reducing `step_size_Mm` and comparing the resulting
maps. Thin high-$Q$ structures and field lines tangent to a boundary generally
need smaller steps than smooth length or connectivity maps.

## Tracing Domains And Boundaries

The model is sampled in Cartesian coordinates for both geometries. Spherical
seed grids are constructed from radius, colatitude, and longitude and converted
to Cartesian model coordinates before tracing. Perturbed-Q offsets use a local
surface-tangent basis embedded in Cartesian space; they are not finite
increments in theta and longitude. This avoids coordinate singularities at the
poles.

### Cartesian Geometry

The Cartesian domain is an axis-aligned box:

- horizontal bounds come from the checkpoint's `coord_range`;
- the lower boundary is `z_min`, normally the photosphere at $z=0$;
- side faces and `z_max` are outer boundaries;
- `apex_height` is the maximum Cartesian $z$ reached by the combined line.

A line is classified as:

- **closed** when both endpoints reach the lower boundary;
- **open** when exactly one endpoint reaches the lower boundary and the other
  reaches a side or top boundary.

The Cartesian boundary ids and names are included in raw trace results. The
standard names are `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max`.

### Spherical Geometry

The spherical domain is a shell defined by the checkpoint's `radius_range`:

- the inner sphere is the photospheric/lower boundary;
- the outer sphere is the open/source boundary;
- `apex_radius` is the maximum radius reached by the combined line.

A spherical line is:

- **closed** when both endpoints reach the inner sphere;
- **open** when one endpoint reaches the inner sphere and the other reaches the
  outer sphere.

`load_spherical()` and `load_spherical_layer()` return both Cartesian magnetic
components in `b` and $(B_r, B_\theta, B_\phi)$ in `b_rtp`. Theta is
colatitude, not latitude.

`footpoint_separation` currently reports the straight Cartesian distance
between the two traced endpoints. In a spherical shell this is a chord distance,
not a great-circle surface distance.

## Field-Line Metrics

Pass these names through `metrics=`:

| Metric request | Returned values | Meaning |
| --- | --- | --- |
| `squashing_factor` | `squashing_factor`, `log10_q`, `q_valid`, `q_condition_number` | Boundary-to-boundary mapping distortion. |
| `twist_number` | `twist_number` | $T_w=\int\alpha\,dl/(4\pi)$. |
| `fieldline_length` | `fieldline_length` | Forward plus backward arc length. |
| `integrated_current_density` | `integrated_current_density` | Vector $\int\mathbf{J}\,dl$. |
| `fieldline_geometry` | `open`, `closed`, `open_polarity`, `footpoint_separation`, `apex_height` or `apex_radius` | Connectivity and geometry. |

Loader metric outputs use Astropy quantities where units apply. Length and
Cartesian separations are returned in Mm, spherical apex radius in solar radii,
and twist and $Q$ are dimensionless. `open_polarity` is `+1` or `-1` at the
inner endpoint of an open line and `0` for lines without a valid open
classification.

### Twist Number

The force-free parameter sampled along the line is

$$
\alpha = \frac{(\nabla\times\mathbf{B})\cdot\mathbf{B}}{|\mathbf{B}|^2}.
$$

NF2 integrates

$$
T_w = \frac{1}{4\pi}\int_L \alpha\,dl
$$

over both half-lines. Computing twist requires the magnetic-field Jacobian at
every Runge--Kutta stage.

### Integrated Current Density

NF2 computes the current density in Gaussian units,

$$
\mathbf{J}=\frac{c}{4\pi}\nabla\times\mathbf{B},
$$

and returns the vector line integral

$$
\int_L \mathbf{J}\,dl.
$$

This is a vector integral, not the integral of $|\mathbf{J}|$. For a scalar
visualization, explicitly plot its magnitude:

```python
import numpy as np

integrated_j = layer["metrics"]["integrated_current_density"]
integrated_j_norm = np.linalg.norm(integrated_j, axis=-1)
```

Current and twist use the same sampled Jacobian when requested together.

## Squashing Factor

$Q$ measures how strongly neighboring field-line footpoint mappings are
stretched. NF2 provides tangent-map and perturbed-point implementations.

### Tangent-Map Method

The default `q_method="tangent"` transports two infinitesimal transverse
vectors with each central field line. Let $D$ be the three-dimensional
deformation matrix. Along each direction it obeys

$$
\frac{dD}{ds}=\pm\nabla\hat{\mathbf{b}}\,D.
$$

At an endpoint, the deformation is corrected from a fixed integration-time
surface to the actual crossed boundary. With boundary normal $\mathbf{n}$ and
flow direction $\mathbf{v}=\pm\hat{\mathbf{b}}$, the event correction is

$$
P=I-\frac{\mathbf{v}\mathbf{n}^{T}}{\mathbf{n}\cdot\mathbf{v}}.
$$

After projection into two-dimensional tangent bases at the seed and endpoints,
the backward and forward maps are $M_-$ and $M_+$. The complete
boundary-to-boundary map is

$$
M=M_+M_-^{-1},
$$

and

$$
Q=\frac{\|M\|_F^2}{|\det M|}.
$$

Numerically valid values satisfy $Q\geq2$. Tangent $Q$ traces only the central
seed line, but it requires the full magnetic-field Jacobian and deformation
matrix during every integration stage.

### Perturbed-Point Method

Set `q_method="perturbed"` to estimate the endpoint derivatives with centered
finite differences. NF2 builds a stable transverse basis $(\mathbf{e}_1,
\mathbf{e}_2)$ and uses the point set

$$
\mathbf{x}_0,\quad
\mathbf{x}_0\pm\epsilon\mathbf{e}_1,\quad
\mathbf{x}_0\pm\epsilon\mathbf{e}_2.
$$

The four offsets form the centered stencil. Their forward and backward
endpoints must land on the same respective boundaries as the central line.
This method avoids Jacobians along the stencil lines, but traces multiple lines
per seed and is normally slower. Use it as a reference or when comparing the
tangent-map implementation.

`q_epsilon_Mm` controls the physical offset. It should be small compared with
the resolved magnetic structure but large enough to avoid cancellation. When
it is omitted, the normalized offset defaults to one quarter of the integration
step.

### Invalid Q

Use `q_valid` when plotting or aggregating Q. Q is invalid when:

- either central half-line does not reach a boundary;
- the endpoint flow is nearly tangent to the boundary;
- the backward tangent map is singular or exceeds `q_condition_limit`;
- a perturbed stencil changes boundary connectivity;
- the endpoint map contains non-finite values.

Do not silently replace invalid Q with a finite value. A useful display is:

```python
log_q = layer["metrics"]["log10_q"].copy()
log_q[~layer["metrics"]["q_valid"]] = np.nan
```

## Trace Configuration

Use a dictionary with physical aliases through an output object:

```python
trace_config = {
    "method": "rkf45",
    "step_size_Mm": 0.25,
    "rtol": 1e-5,
    "atol": 1e-7,
    "min_step_size_Mm": 0.001,
    "max_step_size_Mm": 1.0,
    "max_steps": 10_000,
    "max_length_Mm": None,
    "min_field_G": 1e-6,
    "batch_size": 2**14,
    "progress": True,
    "store_path": False,
    "q_method": "tangent",
    "q_epsilon_Mm": 0.05,
    "q_condition_limit": 1e8,
}
```

| Option | Description |
| --- | --- |
| `method` | `euler`/`rk1`, `rk2`, `rk3`, `rk4`, or adaptive `rkf45`. |
| `step_size_Mm` | Fixed step, or initial RKF45 step, passed through an output helper. |
| `step_size` | Step in normalized model coordinates. |
| `rtol`, `atol` | Relative and absolute RKF45 error tolerances. |
| `min_step_size_Mm`, `max_step_size_Mm` | Optional physical RKF45 step bounds. |
| `max_steps` | Maximum accepted steps for each half-line. |
| `max_length_Mm` | Maximum length of each half-line in Mm. The combined line can be twice this value. |
| `min_field_G` | Stop when a sampled field falls below this physical strength. |
| `boundary_tolerance` | Normalized tolerance for accepting seed points near a boundary. |
| `batch_size` | Number of original seeds kept live; up to twice this many half-lines are evaluated together. |
| `store_path` | Return the complete padded trajectory. This is memory intensive. |
| `progress` | Show completed combined lines and `lines/s`. |
| `q_method` | `tangent` or `perturbed`. |
| `q_epsilon_Mm` | Physical finite-difference offset for perturbed Q. |
| `q_condition_limit` | Maximum accepted tangent-map condition number. |

`TraceConfig` itself uses normalized model coordinates. Physical keys ending in
`_Mm` or `_G` are converted only when a configuration dictionary is passed
through `trace_field_lines()`, `compute_fieldline_metrics()`, or a loader.

## Cartesian Examples

### One Horizontal Layer

```python
from astropy import units as u
import nf2

out = nf2.load("cartesian/extrapolation_result.nf2")
layer = out.load_slice(
    z=10 * u.Mm,
    Mm_per_pixel=0.72,
    metrics=["squashing_factor", "twist_number", "fieldline_geometry"],
    trace_config={"method": "rk4", "step_size_Mm": 0.18, "progress": True},
)

log_q = layer["metrics"]["log10_q"]
apex = layer["metrics"]["apex_height"]
```

### Explicit Cartesian Points

Explicit `trace()` and `compute_fieldline_metrics()` coordinates are normalized
model coordinates, not Mm:

```python
import numpy as np

points_Mm = np.array([
    [20.0, 30.0, 0.0],
    [40.0, 50.0, 10.0],
])
points_model = points_Mm / out.Mm_per_ds

metrics = out.compute_fieldline_metrics(
    points_model,
    metrics=["fieldline_length", "twist_number", "fieldline_geometry"],
    trace_config={"step_size_Mm": 0.25, "progress": True},
)
```

### Complete Paths

```python
lines = out.trace(
    points_model,
    metrics=["fieldline_length", "fieldline_geometry"],
    trace_config={
        "method": "rk4",
        "step_size_Mm": 0.25,
        "store_path": True,
        "progress": True,
    },
)

# Shape: (padded_path_length, number_of_points, 3).
paths_model = lines["path"]
paths_Mm = paths_model * out.Mm_per_ds
```

NaN padding marks positions beyond the end of each ragged path.

## Spherical Examples

### Bottom-Boundary Topology

```python
from astropy import units as u
import nf2

out = nf2.load("spherical/extrapolation_result.nf2")
bottom = out.load_spherical_layer(
    radius=out.radius_range[0],
    sampling=(180, 360),
    metrics=[
        "twist_number",
        "integrated_current_density",
        "fieldline_geometry",
    ],
    trace_config={"method": "rk2", "step_size_Mm": 0.25, "progress": True},
)

open_regions = bottom["metrics"]["open"]
closed_regions = bottom["metrics"]["closed"]
apex_radius = bottom["metrics"]["apex_radius"]
```

### S-Web Layer

Trace a coronal seed layer rather than the bottom boundary when visualizing the
S-web:

```python
s_web = out.load_spherical_layer(
    radius=1.4 * u.solRad,
    sampling=(180, 360),
    metrics=["squashing_factor"],
    trace_config={
        "method": "rk4",
        "step_size_Mm": 0.25,
        "q_method": "tangent",
        "progress": True,
    },
)

log_q = s_web["metrics"]["log10_q"]
br = s_web["b_rtp"][..., 0]
```

Spherical arrays are sampled in `(colatitude, longitude)` order after the fixed
radius dimension is removed. For a latitude plot with `origin="lower"`, flip
the first axis:

Set `sin_latitude=True` on `load_spherical()` or `load_spherical_layer()` to
sample uniformly in sine latitude instead of latitude. This provides an
equal-area angular grid and should be paired with sine-latitude plot edges.

```python
signed_log_q = np.flip(np.sign(br.value) * log_q, axis=0)
```

### Explicit Spherical Points

`load_spherical_coords()` accepts `SkyCoord`, but raw tracing still uses
Cartesian model coordinates:

```python
from nf2.data.util import spherical_to_cartesian

# Columns are radius [R_sun], colatitude [rad], longitude [rad].
points_rtp = np.array([
    [1.0, np.pi / 2, 0.0],
    [1.4, np.pi / 3, np.pi],
])
points_cartesian_Rsun = spherical_to_cartesian(points_rtp)
scale = (1 * u.solRad / out.m_per_ds).to_value(u.dimensionless_unscaled)
points_model = points_cartesian_Rsun * scale

metrics = out.compute_fieldline_metrics(
    points_model,
    metrics=["squashing_factor", "fieldline_geometry"],
    trace_config={"step_size_Mm": 0.25},
)
```

## Full Volumes And Exports

Any field-line metric can be requested for a layer, defined points, or a full
sampled volume. For a Cartesian checkpoint:

```python
cube = out.load_cube(
    Mm_per_pixel=2.0,
    height_range=[0, 80],
    metrics=["fieldline_length", "fieldline_geometry"],
    trace_config={"step_size_Mm": 0.5, "batch_size": 2**14, "progress": True},
)
```

For a spherical checkpoint, use a regular spherical grid:

```python
spherical_volume = out.load_spherical(
    radius_range=[1.0, 1.5] * u.solRad,
    sampling=(20, 90, 180),
    metrics=["fieldline_length", "fieldline_geometry"],
    trace_config={"step_size_Mm": 0.5, "batch_size": 2**14, "progress": True},
)
```

Full-volume Q and Jacobian-based metrics can be expensive. Start with a coarse
layer, verify boundaries and convergence, then increase resolution.

The unified exporter accepts the same metric names:

```bash
nf2-export extrapolation_result.nf2 \
  --format hdf5 \
  --out topology.hdf5 \
  --metrics squashing_factor twist_number fieldline_length \
            integrated_current_density fieldline_geometry \
  --trace-method rk4 \
  --trace-step-Mm 0.25 \
  --trace-batch-size 16384 \
  --q-method tangent
```

See [Exporting](exporting.md) for format-specific limitations and all CLI
options.

## Termination And Status

Each half-line terminates with one status:

| Status | Meaning |
| --- | --- |
| `boundary` | Reached a Cartesian face or spherical shell boundary. |
| `weak_field` | Field magnitude fell below `min_field_strength`. |
| `non_finite` | Model or integration produced a non-finite coordinate. |
| `max_steps` | Reached the per-half-line step limit. |
| `max_length` | Reached the per-half-line length limit. |
| `invalid_start` | Seed was outside the tracing domain. |

Raw `trace()` results include `forward_status`, `backward_status`,
`status_codes`, boundary ids, and `boundary_names`. Geometry and Q are valid
only when both halves terminate on boundaries. A line that stops at
`max_steps`, `max_length`, or weak field still has a partial length and partial
integrals, but it is not classified as a complete open or closed line.

## Performance And Memory

- Keep `store_path=False` for maps. Full path recording copies ragged path
  samples to CPU and is intended for selected seeds.
- Increase `batch_size` until GPU memory or throughput stops improving. The
  live queue can contain up to `2 * batch_size` half-lines.
- `rk4` makes four model calls per step; `rk2` makes two; `rkf45` makes six per
  attempt and adapts the number of accepted steps to the local error.
- Twist and integrated current require a Jacobian at each stage.
- Tangent Q shares that Jacobian when requested with twist or current.
- Perturbed Q traces four displaced stencil lines only when `squashing_factor` and
  `q_method="perturbed"` are both selected.
- Tracing tensors remain on the accelerator until public results are converted
  to NumPy. `store_path=True` and progress reporting introduce host transfers or
  synchronization.
- Multi-GPU systems currently use PyTorch `DataParallel` for model sampling.
  Queue state and gathered results remain on the primary GPU.

For a reproducible production map, report the NF2 checkpoint, seed layer,
sampling, integration method, step size, limits, Q method, and Q epsilon.

## Notebook

The runnable [field-line tracing notebook](../examples/notebooks/field_line_tracing.ipynb)
starts from an existing `.nf2` file, automatically selects Cartesian or
spherical loading, computes all supported field-line diagnostics, plots a
layer, and traces selected complete paths.
