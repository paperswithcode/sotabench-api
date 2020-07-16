import hashlib
from typing import Optional, Union
from datetime import datetime, timezone

import click

from sotabenchapi.uploader.consts import MB


def get_sha256(file, size, chunk_size=MB, label="") -> str:
    """Return a sha256 hexdigest of a file.

    Tested different chunk sizes, and I got the fastest calculation with the
    1MB chunk size.

    Args:
        file: File like object.
        size (int): File size.
        chunk_size (int): Chunk size while reading the file.
        label (str): Progress bar label.
    """
    sha = hashlib.sha256()

    with click.progressbar(length=size, label=label) as bar:
        while True:
            buf = file.read(chunk_size)
            if not buf:
                break
            sha.update(buf)
            bar.update(chunk_size)
    return sha.hexdigest()


def utcnow() -> datetime:
    """Return tz aware UTC now."""
    return datetime.utcnow().astimezone(timezone.utc)


# Format used in serialization and deserialization from json.
timestamp_format = "%Y.%m.%dT%H:%M:%S"


def strftime(dt: Optional[datetime]):
    """Format datetime as string."""
    if dt is None:
        return None
    return dt.strftime(timestamp_format)


# Unsafe timestamp can either be None (when no timestamp is provided),
# string (when we deserialized json) or datetime object.
UnsafeTimestamp = Optional[Union[str, datetime]]

# Safe timestamp is either None (when no timestamp is provided or a timezone
# aware datetime
SafeTimestamp = Optional[datetime]


def safe_timestamp(dt: UnsafeTimestamp) -> SafeTimestamp:
    """Returns tz aware UTC SafeTimestamp from UnsafeTimestamp.

    It can receive either str, datetime or None. If it receives None or
    datetime it will return them, since both are valid in object serialization.

    None represents missing object and datetime is a valid datetime.

    If it receives a string it's probably a product of json deserialization
    and it should be parsed to a valid datetime object.

    Args:
        dt: None, string serialized datetime or datetime object.
    """
    # If it's None return None
    if dt is None:
        return None

    # If it's a string, it's from deserialized json so strptime it.
    if isinstance(dt, str):
        dt = datetime.strptime(dt, timestamp_format)

    # Return timezone aware UTC datetime.
    return dt.astimezone(timezone.utc)
