"""Dispatch the public NF2 command to its module, importing it on demand."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    "extrapolate": "nf2.extrapolate",
    "extrapolate-series": "nf2.extrapolate_series",
    "export": "nf2.export",
    "metrics": "nf2.metrics",
    "download": "nf2.data.download",
    "noaa-to-sharp": "nf2.data.noaa_to_sharp",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="nf2", description="Run and evaluate NF2 extrapolations.")
    parser.add_argument("command", choices=COMMANDS)
    command = parser.parse_args(argv[:1]).command
    import_module(COMMANDS[command]).main(argv[1:], prog=f"nf2 {command}")
