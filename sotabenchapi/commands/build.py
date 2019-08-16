import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, check_repo


@cli.group("build")
def build_cli():
    """Build operations."""
    pass


@build_cli.command("start")
@click.argument("repository", required="True")
@click.pass_obj
@handle_errors()
def build_start(config: Config, repository):
    """Start build."""
    repository = check_repo(repository)
    client = Client(config)
    click.secho(client.build_start(repository=repository))


@build_cli.command("get")
@click.argument("repository", required=True)
@click.argument("run_number", type=int, required=True)
@click.pass_obj
@handle_errors()
def build_get(config: Config, repository: str, run_number: int):
    """Get build details."""
    repository = check_repo(repository)
    client = Client(config)
    click.secho(client.build_get(repository=repository, run_number=run_number))
