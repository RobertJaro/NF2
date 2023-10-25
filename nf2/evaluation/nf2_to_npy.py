import argparse

import numpy as np
import torch

from nf2.evaluation.unpack import load_cube

parser = argparse.ArgumentParser(description='Convert NF2 file to npy.')
parser.add_argument('nf2_path', type=str, help='path to the source NF2 file')
parser.add_argument('npy_path', type=str, help='path to the target numpy file')
parser.add_argument('--strides', type=int, help='downsampling of the volume', required=False, default=1)

args = parser.parse_args()
nf2_path = args.nf2_path
strides = args.strides
npy_path = args.npy_path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

b = load_cube(nf2_path, device, progress=True, strides=strides)

np.save(npy_path, b)


def main(): # workaround for entry_points
    pass