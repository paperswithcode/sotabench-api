import io
import sys
import uuid
import mmap


from sotabenchapi.uploader.utils import utcnow, strftime, safe_timestamp


class Part:
    def __init__(
        self,
        upload,
        no,
        size,
        state,
        md5=None,
        etag=None,
        start_time=None,
        end_time=None,
        presigned_url=None,
    ):
        self.upload = upload
        self.no = no
        self.size = size
        self.state = state
        self.md5 = md5
        self.etag = etag
        self.start_time = start_time or utcnow()
        self.end_time = end_time
        self.presigned_url = presigned_url

    def to_dict(self) -> dict:
        return {
            "upload": str(self.upload),
            "no": self.no,
            "size": str(self.size),
            "md5": self.md5,
            "etag": self.etag,
            "state": self.state,
            "start_time": strftime(self.start_time),
            "end_time": strftime(self.end_time),
            "presigned_url": self.presigned_url,
        }

    @classmethod
    def from_dict(cls, d) -> "Part":
        return cls(
            upload=uuid.UUID(d["upload"]),
            no=int(d["no"]),
            size=int(d["size"]),
            md5=d["md5"],
            etag=d["etag"],
            state=d["state"],
            start_time=safe_timestamp(d["start_time"]),
            end_time=safe_timestamp(d["end_time"]),
            presigned_url=d["presigned_url"],
        )


def memory_map(file):
    if "win" in sys.platform and not sys.platform == "darwin":
        return mmap.mmap(file.fileno(), 0, None, mmap.ACCESS_READ)
    else:
        return mmap.mmap(file.fileno(), 0, mmap.MAP_PRIVATE, mmap.ACCESS_READ)


class Slice:
    def __init__(self, filename, length, offset=0):
        self.filename = filename
        self.length = length
        self.offset = offset
        self.end = offset + length
        self.pos = offset
        self.f = io.open(filename, "rb")
        self.mmap = memory_map(self.f)
        # self.bio = io.BytesIO(self.mmap[self.offset:self.end])
        self.m = self.mmap[self.offset : self.end]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reset(self):
        self.f.seek(self.offset, 0)

    def read(self, size):
        return self.bio.read(size)

    def readline(self, size=-1):
        return self.bio.readline(size)

    def readlines(self, hint=-1):
        return self.bio.readlines(hint)

    def __iter__(self):
        return self.bio.__iter__()

    def close(self):
        self.mmap.close()
        self.f.close()
