import sys
import functools

import click
from tabulate import tabulate

from sotabenchapi import errors


def handle_errors(m404=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors.HttpClientError as e:
                if m404 and e.status_code == 404:
                    click.secho(m404, fg="red")
                else:
                    click.secho(e.message, fg="red")

        return wrapper

    return decorator


def check_repo(repository: str):
    parts = repository.split("/")
    if len(parts) != 2:
        click.secho(f"Invalid repository name: ", fg="red", nl=False)
        click.secho(repository)
        click.secho(
            "Repository name must be in owner/project format.", fg="cyan"
        )
        sys.exit(1)
    return repository.strip("/")


def table(data):
    """Show data as a table."""
    if not isinstance(data, (list, tuple)):
        data = [data]
    click.secho(tabulate(data, headers="keys", tablefmt="fancy_grid"))
