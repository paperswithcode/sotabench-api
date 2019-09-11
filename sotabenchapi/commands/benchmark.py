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
@click.argument(
    "dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("-b", "--benchmark", required=True, help="Benchmark slug.")
@click.option("-l", "--library", required=True, help="Library name.")
@click.pass_obj
@handle_errors(m404="Benchmark library not found.")
def upload(config: Config, dataset: str, benchmark: str, library: str):
    """Upload dataset for a benchmark."""
    client = Client(config)
    client.benchmark_upload(
        dataset=dataset, benchmark=benchmark, library=library
    )
