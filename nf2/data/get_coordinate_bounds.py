import argparse

import numpy as np
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
import astropy.units as u

# arguments
parser = argparse.ArgumentParser(description='Determines the Carrington coordinate bounds of a FITS file.')

parser.add_argument('file', type=str, help='FITS file to get the coordinate bounds of')
parser.add_argument('--max_longitude', type=float, default=0.45 * np.pi, help='Buffer to add to the coordinate bounds')
parser.add_argument('--max_latitude', type=float, default=0.4 * np.pi, help='Buffer to add to the coordinate bounds')
args = parser.parse_args()

s_map = Map(args.file)
center_coord = s_map.center.transform_to(frames.HeliographicCarrington)
center_lat = np.pi / 2 * u.rad - center_coord.lat
center_lon = center_coord.lon

lat_bounds = np.array([center_lat.to_value(u.rad) - args.max_latitude,
                       center_lat.to_value(u.rad) + args.max_latitude])

lon_bounds = np.array([center_lon.to_value(u.rad) - args.max_longitude,
                       center_lon.to_value(u.rad) + args.max_longitude])
if lon_bounds[0] < 0:
    lon_bounds += 2 * np.pi

print(f'Center: lat {center_lat.to(u.rad):.3f}, lon {center_lon.to(u.rad):.3f}')
print(f'Longitude bounds: {lon_bounds[0]:.3f}, {lon_bounds[1]:.3f}')
print(f'Latitude bounds: {lat_bounds[0]:.3f}, {lat_bounds[1]:.3f}')
