import os
import click

from sotabenchapi import config
from sotabenchapi import consts


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    envvar="SOTABENCH_CONFIG",
    help="Path to the alternative configuration file.",
)
@click.option(
    "--profile",
    default="default",
    envvar="SOTABENCH_PROFILE",
    help="Configuration file profile.",
)
@click.pass_context
def cli(ctx, config_path, profile):
    """sotabench command line client."""
    if config is None:
        config_path = os.path.expanduser(consts.DEFAULT_CONFIG_PATH)
    ctx.obj = config.Config(config_path, profile)
