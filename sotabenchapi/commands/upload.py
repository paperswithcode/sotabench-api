import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, check_repo


part_size_type = click.IntRange(min=5)
part_size_type.name = "integer"


@cli.command("upload")
@click.argument(
    "dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("-r", "--repository", required=True, help="Repository slug.")
@click.option(
    "-p",
    "--path",
    required=False,
    default=None,
    help="Path in .data folder where the dataset should be downloaded. "
         "Default: `basename(dataset)`",
)
@click.option(
    "-s",
    "--part-size",
    type=part_size_type,
    default=None,
    help=(
        "Set the part size in MB (min 5MB). If not provided the part size "
        "will be calculated based on the file size."
    ),
)
@click.pass_obj
@handle_errors(m404="Repository not found.")
def upload(
    config: Config, dataset: str, repository: str, path: str, part_size: int
):
    """Upload dataset for a repository."""
    client = Client(config)
    client.upload(
        dataset=dataset,
        repository=check_repo(repository),
        path=path,
        part_size=part_size,
    )
