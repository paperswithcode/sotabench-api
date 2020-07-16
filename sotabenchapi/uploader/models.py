import enum
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

from sotabenchapi.uploader.utils import utcnow, strftime, safe_timestamp


class UploadState(enum.Enum):
    exists = "exists"
    queued = "queued"
    started = "started"
    finished = "finished"
    error = "error"


@dataclass
class Upload:
    State = UploadState

    id: str
    sha256: str
    size: int
    part_size: int
    part_number: int
    state: UploadState

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "size": str(self.size),
            "part_size": str(self.part_size),
            "part_number": self.part_number,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Optional["Upload"]:
        if len(d) == 0:
            return None

        return cls(
            id=d["id"],
            sha256=d["sha256"],
            size=int(d["size"]),
            part_size=int(d["part_size"]),
            part_number=d["part_number"],
            state=UploadState(d["state"]),
        )


@dataclass
class Part:
    State = UploadState

    upload: str
    no: int
    size: int
    state: UploadState
    sha256: Optional[str] = None
    etag: Optional[str] = None
    started_time: Optional[datetime] = None
    finished_time: Optional[datetime] = None
    presigned_url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "upload": self.upload,
            "no": self.no,
            "size": str(self.size),
            "sha256": self.sha256,
            "etag": self.etag,
            "state": self.state.value,
            "started_time": strftime(self.started_time),
            "finished_time": strftime(self.finished_time),
            "presigned_url": self.presigned_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Optional["Part"]:
        if len(d) == 0:
            return None

        return cls(
            upload=d["upload"],
            no=int(d["no"]),
            size=int(d["size"]),
            sha256=d["sha256"],
            etag=d["etag"],
            state=UploadState(d["state"]),
            started_time=safe_timestamp(d["started_time"]) or utcnow(),
            finished_time=safe_timestamp(d["finished_time"]),
            presigned_url=d["presigned_url"],
        )
