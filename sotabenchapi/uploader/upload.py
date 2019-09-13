import io
import os
import logging

import requests

from sotabenchapi.http import HttpClient
from sotabenchapi.uploader.utils import get_md5
from sotabenchapi.uploader.models import Part


logger = logging.getLogger(__name__)


def upload(http: HttpClient, filename: str, benchmark: str, library: str):
    size = os.stat(filename).st_size
    file = io.open(filename, "rb")
    try:
        md5 = get_md5(file, size=size, label="Calculating file MD5")
        file.seek(0)

        result = http.post(
            "/upload/start/",
            data={
                "benchmark": benchmark,
                "library": library,
                "name": os.path.basename(filename),
                "size": size,
                "md5": md5,
            },
        )
        print(result)
        upload_id = result["id"]
        while True:
            result = http.post(
                "/upload/part/reserve/", data={"upload_id": upload_id}
            )
            if result == {}:
                # No more parts to upload, we finished
                return
            print(result)
            part = Part.from_dict(result)
            offset = (part.no - 1) * part.size
            file.seek(offset)
            buffer = io.BytesIO(file.read(part.size))
            part.md5 = get_md5(
                buffer,
                size=part.size,
                label=f"Calculating part #{part.no} MD5",
            )
            buffer.seek(0)
            result = http.post("/upload/part/start/", data=part.to_dict())
            print(result)
            part = Part.from_dict(result)
            try:
                result = requests.put(
                    part.presigned_url,
                    data=buffer,
                    headers={
                        "Content-Length": str(part.size),
                        "Content-MD5": part.md5,
                        "Host": "sotabench.s3.amazonaws.com",
                    },
                )
                print("-----------------")
                print(repr(result.text))
                print(result.headers)
                print("-----------------")
            except Exception as e:
                logger.exception("Failed to upload: %s", e)
                part.state = "error"
                http.post("/upload/part/end/", data=part.to_dict())

    finally:
        file.close()
