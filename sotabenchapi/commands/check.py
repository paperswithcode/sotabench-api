import os
import sys
import click
import subprocess
from pathlib import Path

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli


@cli.command("check")
@click.pass_obj
def check(config: Config):
    """Check if the benchmarking setup is correct."""
    cwd = Path(os.getcwd()).absolute()

    if not (cwd / "sotabench.py").is_file():
        click.secho("sotabench.py is missing.", fg="red")
        sys.exit(1)

    if not (cwd / "requirements.txt").is_file():
        click.secho("requirements.txt is missing.", fg="red")
        sys.exit(1)

    process = subprocess.Popen(
        [sys.executable, "sotabench.py"],
        env={"SOTABENCH_CHECK": config.sotabench_check or "full"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()

    if process.returncode != 0:
        click.secho("Could not run sotabench.py.", fg="red")
        stdout = process.stdout.read().decode("utf-8").strip()
        stderr = process.stderr.read().decode("utf-8").strip()
        if stdout:
            click.secho(f"\nStdout:", fg="cyan")
            click.secho(stdout)
        if stderr:
            click.secho(f"\nStderr:", fg="cyan")
            click.secho(stderr, fg="red")

    # Check hashes
    hashes = ["foo", "bar"]
    client = Client(config)
    result = client.check_run_hashes(hashes)
    for h, exist in result.items():
        click.secho(
            f"Hash: `{h}` - {'exists' if exist else 'does not exist.'}"
        )
