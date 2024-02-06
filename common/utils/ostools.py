from pathlib import Path
from typing import Literal
from glob import glob
from datetime import datetime
from dataclasses import dataclass
from common.utils.timetools import timediff
from common.utils import strtools


def ls(arg: str | Path,
       sortby: Literal['path', 'name', 'created', 'modified', 'accessed', 'size'] = 'path',
       kind: Literal['file', 'dir', 'both'] = 'both',
       allow_zero_size: bool = True
       ) -> list[Path]:

    file_infos = [FileInfo(pth) for pth in glob(str(arg))]

    if kind != 'both':
        file_infos = [file_info for file_info in file_infos if file_info.path.is_file() == (kind == 'file')]

    if not allow_zero_size:
        file_infos = [file_info for file_info in file_infos if file_info.size > 0]

    file_infos.sort(key=lambda x: x[sortby])

    return [file_info.path for file_info in file_infos]


@dataclass
class FileInfo:

    path: Path | str
    size: int = None
    created: datetime = None
    modified: datetime = None
    inode_change: datetime = None
    accessed: datetime = None

    def __post_init__(self):
        self.path = Path(self.path)
        stat = self.path.stat()
        self.size = stat.st_size
        self.created = datetime.fromtimestamp(stat.st_birthtime)
        self.modified = datetime.fromtimestamp(stat.st_mtime)
        self.inode_change = datetime.fromtimestamp(stat.st_ctime)
        self.accessed = datetime.fromtimestamp(stat.st_atime)

    def time_ago(self, attrib, unit: str = 's') -> float:
        return timediff.convert(datetime.now() - getattr(self, attrib), unit)

    def days_ago_created(self) -> float:
        return self.time_ago('created', unit='d')

    def days_ago_modified(self) -> float:
        return self.time_ago('modified', unit='d')

    @property
    def name(self) -> str:
        return self.path.name

    def __getitem__(self, item):
        return getattr(self, item)

    def display(self):
        print(strtools.attribs_string(self))
