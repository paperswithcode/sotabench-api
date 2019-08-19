import os
import sys
import click
import subprocess
from pathlib import Path

from sotabenchapi.config import Config
from sotabenchapi.client import Client
from sotabenchapi.commands.cli import cli
from sotabenchapi.commands.utils import handle_errors


@cli.command("check")
@click.pass_obj
@click.option("-p", "--params",
              is_flag=True,
              default=False,
              help="Checks the parameters, such as model names and arxiv paper ids, "
                   "to ensure they are correct. Does not perform evaluation, "
                   "but is a pure API call with the inputs. You can use this command to check "
                   "input string validity before submission.")
@handle_errors()
def check(config: Config, params: bool=False):
    """Check if the benchmarking setup is correct."""
    cwd = Path(os.getcwd()).absolute()

    if not (cwd / "sotabench.py").is_file():
        click.secho("sotabench.py is missing.", fg="red")
        sys.exit(1)

    if not (cwd / "requirements.txt").is_file():
        click.secho("requirements.txt is missing.", fg="red")
        sys.exit(1)

    check_var = config.sotabench_check or "full"

    if params is True:
        check_var = "params"

    process = subprocess.Popen(
        [sys.executable, "sotabench.py"],
        env={"SOTABENCH_CHECK": check_var},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()

    if process.returncode != 0:  # catch running errors
        click.secho("Could not run sotabench.py.", fg="red")
        stdout = process.stdout.read().decode("utf-8").strip()
        stderr = process.stderr.read().decode("utf-8").strip()
        if stdout:
            click.secho(f"\nStdout:", fg="cyan")
            click.secho(stdout)
        if stderr:
            click.secho(f"\nStderr:", fg="cyan")
            click.secho(stderr, fg="red")
    else:  # catch other stdout
        stdout = process.stdout.read().decode("utf-8").strip()

        if stdout:
            click.secho(stdout)
