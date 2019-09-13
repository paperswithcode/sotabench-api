import uuid


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
