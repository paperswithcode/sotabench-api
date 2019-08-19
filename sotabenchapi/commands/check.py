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
@click.option("-cm", "--checkmeta",
              is_flag=True,
              default=False,
              help="Checks the metadata, such as model names and arxiv paper ids, "
                   "to ensure they are correct. Does not perform evaluation, "
                   "but is a pure API call with the inputs. You can use this command to check "
                   "input validity before submission ")
@click.option("-cp", "--checkparams",
              is_flag=True,
              default=False,
              help="Checks all parameters, including the model, by performing one batch of evaluation "
                   "and ensuring things run. Afterwards will also check metadata - same checks as checkmeta")
@handle_errors()
def check(config: Config, checkparams: bool=False, checkmeta: bool = False):
    """Check if the benchmarking setup is correct."""
    cwd = Path(os.getcwd()).absolute()

    if not (cwd / "sotabench.py").is_file():
        click.secho("sotabench.py is missing.", fg="red")
        sys.exit(1)

    if not (cwd / "requirements.txt").is_file():
        click.secho("requirements.txt is missing.", fg="red")
        sys.exit(1)

    if checkparams is True:
        check_var = "params"
    elif checkmeta is True:
        check_var = "meta"
    else:
        check_var = "full"

    process = subprocess.Popen(
        [sys.executable, "sotabench.py"],
        env={"SOTABENCH_CHECK": config.sotabench_check or check_var},
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
