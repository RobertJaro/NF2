# Coordinates and Units

This page is important for avoiding geometry mistakes.

## Cartesian Convention

Cartesian workflows operate in local `(x, y, z)` coordinates.

- `x`, `y`: horizontal spatial axes
- `z`: height above the lower boundary

## Spherical Convention

Internally, spherical transforms use:

- `r`: radius
- `theta`: colatitude
- `phi`: longitude

That means:

- `theta = 0` at the north pole
- `theta = pi / 2` at the equator
- `theta = pi` at the south pole

## User-Facing Latitude Inputs

Configs and export commands usually accept latitude ranges in the more familiar latitude convention.

These are converted at the boundary to the internal colatitude convention.

## Vector Components

Spherical vectors follow the internal `(r, theta, phi)` basis.

When converting to Cartesian:

- radial stays radial
- `theta` is the colatitude direction
- `phi` is the longitudinal direction

## Units

Spatial coordinates are normalized internally, but configs and exports usually refer to:

- `Mm`
- `solRad`
- `pix / solRad` for spherical export resolution

Magnetic fields are typically stored or reported in:

- `G`

## Important Note

The spherical loader and output layers were explicitly aligned during the refactor so that all internal transforms now use the same colatitude convention.
