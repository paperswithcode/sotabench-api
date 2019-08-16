import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, check_repo, table


@cli.group("repo")
def repo_cli():
    """Repository operations."""
    pass


@repo_cli.command("list")
@click.option(
    "-o", "--owner", help="Filter by repository owner.", default=None
)
@click.pass_obj
@handle_errors()
def repo_list(config: Config, owner):
    """List repositories."""
    client = Client(config)
    table(client.repository_list(username=owner))


@repo_cli.command("get")
@click.argument("repository", required=True)
@click.pass_obj
@handle_errors()
def repo_get(config: Config, repository: str):
    """Get repository.

    Repository name must be in ``owner/project`` format.
    """
    repository = check_repo(repository)
    client = Client(config)
    table(client.repository_get(repository=repository))
