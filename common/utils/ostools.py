from pathlib import Path
from typing import Literal
from glob import glob
import os
from datetime import datetime


def ls(arg: str | Path,
       sortby: Literal['name', 'path', 'create', 'modify', 'access', 'size'] = 'path',
       kind: Literal['file', 'dir', 'both'] = 'both',
       aspath: bool = False):

    paths_list = glob(str(arg))

    if kind != 'both':
        paths_list = [pth for pth in paths_list if os.path.isfile(pth) == (kind == 'file')]

    if sortby == 'path':
        paths_list.sort()
    elif sortby == 'name':
        paths_list.sort(key=os.path.basename)
    else:
        paths_list.sort(key=lambda x: stats(x)[sortby])

    if aspath:
        paths_list = [Path(pth) for pth in paths_list]

    return paths_list


def stats(pth: str | Path | list):
    if isinstance(pth, list):
        return [stats(p) for p in pth]
    s = os.stat(pth)
    ret = {'create': datetime.fromtimestamp(s.st_birthtime),
           'modify': datetime.fromtimestamp(s.st_mtime),
           'inode_change': datetime.fromtimestamp(s.st_ctime),
           'access': datetime.fromtimestamp(s.st_atime),
           'size': s.st_size}
    return ret
