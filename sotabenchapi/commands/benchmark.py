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
