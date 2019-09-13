import io
import os
import requests

from sotabenchapi.http import HttpClient
from sotabenchapi.uploader.utils import get_md5
from sotabenchapi.uploader.models import Part, Slice


def upload(http: HttpClient, filename: str, benchmark: str, library: str):
    size = os.stat(filename).st_size
    with io.open(filename, "rb") as f:
        md5 = get_md5(f, size=size, label="Calculating file MD5")

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
        with Slice(filename, offset=offset, length=part.size) as s:
            part.md5 = get_md5(
                s, size=part.size, label=f"Calculating part #{part.no} MD5"
            )
        result = http.post("/upload/part/start/", data=part.to_dict())
        print(result)
        part = Part.from_dict(result)
        with Slice(filename, offset=offset, length=part.size) as s:
            result = requests.put(part.presigned_url, data=s)
            print(result.json())
