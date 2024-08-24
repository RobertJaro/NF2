# create a new file based on a template file
import argparse
import os
from string import Template


def main():
    parser = argparse.ArgumentParser(
        description='Create a new config file for synoptic extrapolations based on a template file')
    parser.add_argument('--template_file', help='template file')
    parser.add_argument('--carrington_rotation', nargs='+', help='carrington rotations of the synoptic maps')
    parser.add_argument('--output_dir', help='output directory', required=False, default='.')
    parser.add_argument('--overwrite', help='overwrite existing files', action='store_true')
    args = parser.parse_args()

    with open(args.template_file, 'r') as template_file:
        src = Template(template_file.read())
        for id in args.carrington_rotation:
            output_file = os.path.join(args.output_dir, f'{id}.json')
            if os.path.exists(output_file) and not args.overwrite:
                print(f'File {output_file} already exists. Skipping.')
                continue
            with open(output_file, 'w') as target_file:
                target_file.write(src.substitute(carrington_rotation=id))


if __name__ == '__main__':
    main()
