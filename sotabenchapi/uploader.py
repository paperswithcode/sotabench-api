import io
import os
import hashlib

import click

from sotabenchapi.http import HttpClient


KB = 1024
MB = KB * KB


def get_md5(filename: str, chunk_size=MB) -> str:
    """Return a md5 hexdigest of a file.

    Tested different chunk sizes, and I got the fastest calculation with the
    1MB chunk size.

    Args:
        filename (str): Path to the file.
        chunk_size (int): Chunk size while reading the file.
    """
    size = os.stat(filename).st_size
    md5 = hashlib.md5()

    with io.open(filename, "rb") as f:
        with click.progressbar(length=size, label="Calculating MD5") as bar:
            while True:
                buf = f.read(chunk_size)
                if not buf:
                    break
                md5.update(buf)
                bar.update(chunk_size)
    return md5.hexdigest()


def upload(http: HttpClient, filename: str, benchmark: str, library: str):
    size = os.stat(filename).st_size
    md5 = get_md5(filename)
    result = http.post(
        f"/upload/{benchmark}/{md5}/start/",
        data={
            "name": os.path.basename(filename),
            "size": size,
            "library": library,
        },
    )
    print(result)
