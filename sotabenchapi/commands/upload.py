import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, check_repo, table


part_size_type = click.IntRange(min=5)
part_size_type.name = "integer"


@cli.command("upload")
@click.argument(
    "dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("-r", "--repository", required=True, help="Repository slug.")
@click.option("-l", "--library", required=True, help="Library name.")
@click.option(
    "-p",
    "--part-size",
    type=part_size_type,
    default=None,
    help=(
        "Set the part size in MB (min 5MB). If not provided the part size "
        "will be calculated based on the file size."
    ),
)
@click.pass_obj
@handle_errors(m404="Library not found.")
def upload(
    config: Config, dataset: str, repository: str, library: str, part_size: int
):
    """Upload dataset for a repository."""
    client = Client(config)
    client.upload(
        dataset=dataset,
        repository=repository,
        library=library,
        part_size=part_size,
    )
