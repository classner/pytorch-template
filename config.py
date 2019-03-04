#!/usr/bin/env python
"""Configuration values."""
from os import path

import click

EXP_DATA_FP = path.abspath(path.join(path.dirname(__file__), "data"))


@click.command()
@click.argument("key", type=click.STRING)
def cli(key):
    """Print a configuration value."""
    if key in globals().keys():
        print(globals()[key])
    else:
        raise Exception(
            "Requested configuration key not available! Available keys: "
            + str([key_name for key_name in globals().keys() if key_name.isupper()])
            + "."
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
