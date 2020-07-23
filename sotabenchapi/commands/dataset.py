import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, check_repo, table


@cli.group("dataset")
def dataset_cli():
    """Dataset operations (upload, management)."""
    pass


part_size_type = click.IntRange(min=5)
part_size_type.name = "integer"


@dataset_cli.command("list")
@click.argument("repository", required=True)
@click.pass_obj
@handle_errors(m404="Repository not found.")
def dataset_list(config: Config, repository: str):
    """List all uploaded datasets for a repository.

    Repository name must be in ``owner/project`` format.
    """
    repository = check_repo(repository)
    client = Client(config)
    table(client.dataset_list(repository=repository))


@dataset_cli.command("upload")
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
def dataset_upload(
    config: Config, dataset: str, repository: str, path: str, part_size: int
):
    """Upload dataset for a repository."""
    client = Client(config)
    client.dataset_upload(
        dataset=dataset,
        repository=check_repo(repository),
        path=path,
        part_size=part_size,
    )


@dataset_cli.command("get")
@click.argument("repository", required=True)
@click.argument("dataset", required=True)
@click.pass_obj
@handle_errors(m404="Either the repository or the dataset is not found.")
def dataset_get(config: Config, repository: str, dataset: str):
    """Get dataset details.

    Repository name must be in ``owner/project`` format.
    """
    repository = check_repo(repository)
    client = Client(config)
    table(client.dataset_get(repository=repository, dataset=dataset))


@dataset_cli.command("delete")
@click.argument("repository", required=True)
@click.argument("dataset", required=True)
@click.pass_obj
@handle_errors(m404="Either the repository or the dataset is not found.")
def dataset_delete(config: Config, repository: str, dataset: str):
    """Delete dataset.

    Repository name must be in ``owner/project`` format.
    """
    repository = check_repo(repository)
    client = Client(config)
    result = client.dataset_delete(repository=repository, dataset=dataset)
    if result["status"] == "OK":
        click.secho("Dataset successfully deleted.", fg="green")
    else:
        click.secho("An error occurred please try again later..", fg="red")
