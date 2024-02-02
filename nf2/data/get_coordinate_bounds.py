import argparse

import numpy as np
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

# arguments
parser = argparse.ArgumentParser(description='Determines the Carrington coordinate bounds of a FITS file.')

parser.add_argument('file', type=str, help='FITS file to get the coordinate bounds of')
parser.add_argument('--max_longitude', type=float, default=2/5 * np.pi, help='Buffer to add to the coordinate bounds')

args = parser.parse_args()

s_map = Map(args.file)
center_coord = s_map.center.transform_to(frames.HeliographicCarrington)

bounds = np.array([center_coord.lon.to_value('rad') - args.max_longitude,
              center_coord.lon.to_value('rad') + args.max_longitude])

print(f'Longitude bounds: {bounds[0]} to {bounds[1]}')