import io
import os
import logging
from typing import Optional

import click
import requests

from sotabenchapi.http import HttpClient
from sotabenchapi.uploader.utils import get_md5
from sotabenchapi.uploader.models import Part, Upload


logger = logging.getLogger(__name__)


class Buffer(io.BytesIO):
    def __init__(self, buffer, label=None):
        self.size = len(buffer)
        if label is None:
            self.bar = None
        else:
            self.bar = click.progressbar(length=self.size, label=label)
        super().__init__(buffer)

    def read(self, n=-1):
        chunk = super().read(n)
        if self.bar is not None:
            self.bar.update(len(chunk))
        return chunk

    def reset(self, label=None):
        self.seek(0)
        if label is None:
            self.bar = None
        else:
            self.bar = click.progressbar(length=self.size, label=label)


def multipart_upload(
    http: HttpClient,
    filename: str,
    benchmark: str,
    library: str,
    part_size: Optional[int] = None,
):
    size = os.stat(filename).st_size
    file = io.open(filename, "rb")
    try:
        md5 = get_md5(file, size=size, label="Calculating file MD5")
        file.seek(0)

        upload = Upload.from_dict(
            http.post(
                "/upload/start/",
                data={
                    "benchmark": benchmark,
                    "library": library,
                    "name": os.path.basename(filename),
                    "size": size,
                    "md5": md5,
                    "part_size": part_size,
                },
            )
        )
        while True:
            part = Part.from_dict(
                http.post(
                    "/upload/part/reserve/", data={"upload_id": upload.id}
                )
            )
            if part is None:
                # No more parts to upload, we finished
                break
            offset = (part.no - 1) * part.size
            file.seek(offset)

            # buffer = io.BytesIO(file.read(part.size))
            buffer = Buffer(file.read(part.size))
            part.md5 = get_md5(
                buffer,
                size=part.size,
                label=f"Calculating MD5 for part #{part.no}",
            )
            part = Part.from_dict(
                http.post("/upload/part/start/", data=part.to_dict())
            )
            buffer.reset(label=f"Uploading part #{part.no}")
            try:
                result = requests.put(part.presigned_url, data=buffer)
                part.etag = result.headers.get("ETag", "")
                part.state = Part.State.finished
            except Exception as e:
                logger.exception("Failed to upload: %s", e)
                part.state = Part.State.error
            http.post("/upload/part/end/", data=part.to_dict())

        http.post("/upload/end/", data={"upload_id": upload.id})
        click.secho("\nUpload successfully finished.", fg="cyan")
    finally:
        file.close()
