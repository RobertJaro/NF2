import argparse

from nf2.core.runner import run
from nf2.train.util import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="canonical YAML config for the analytical run")
    args, overwrite_args = parser.parse_known_args()

    config = load_yaml_config(args.config, overwrite_args)
    run(**config)


if __name__ == "__main__":
    main()
