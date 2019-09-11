import click

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors, table


@cli.group("benchmark")
def benchmark_cli():
    """Benchmark operations."""
    pass


@benchmark_cli.command("list")
@click.pass_obj
@handle_errors()
def benchmark_list(config: Config):
    """List users benchmarks."""
    client = Client(config)
    table(client.benchmark_list())


@benchmark_cli.command("get")
@click.argument("benchmark", required=True)
@click.pass_obj
@handle_errors()
def benchmark_get(config: Config, benchmark: str):
    """Get benchmark.

    Provide benchmark slug as an argument.
    """
    client = Client(config)
    table(client.benchmark_get(benchmark=benchmark))


@benchmark_cli.command("upload")
@click.argument("benchmark", required=True)
@click.argument(
    "dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.pass_obj
@handle_errors()
def upload(config: Config, benchmark: str, dataset: str):
    """Upload dataset for a benchmark."""
    client = Client(config)
    client.benchmark_upload(benchmark=benchmark, dataset=dataset)
