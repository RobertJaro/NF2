import argparse

from nf2.evaluation.output import CartesianOutput

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SHARP snapshot.')
    parser.add_argument('--output', type=str, help='output path.')
    args = parser.parse_args()

    sharp_out = CartesianOutput('/glade/work/rjarolim/nf2/sst/sharp_13392_ambiguous_v02/sharp_13392_ambiguous_v02.vtk')
