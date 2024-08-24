import argparse

from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map

parser = argparse.ArgumentParser()
parser.add_argument('--fits_file', type=str, required=True)
parser.add_argument('--Tx', type=float, required=True)
parser.add_argument('--Ty', type=float, required=True)

args = parser.parse_args()

s_map = Map(args.fits_file)
hpc_coords = SkyCoord(args.Tx * u.arcsec, args.Ty * u.arcsec, frame=s_map.coordinate_frame)
hcc_coords = hpc_coords.transform_to(frames.HeliographicCarrington)

print(f'x: {hcc_coords.lon.value}, y: {hcc_coords.lat.value}')
