import argparse

from nf2.core.runner import run_series
from nf2.train.util import load_yaml_config

run = run_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file for the simulation')
    args, overwrite_args = parser.parse_known_args()

    yaml_config_file = args.config
    config = load_yaml_config(yaml_config_file, overwrite_args)

    run_series(**config)


if __name__ == '__main__':
    main()
