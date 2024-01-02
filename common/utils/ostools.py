from pathlib import Path
from typing import Literal
from glob import glob
import os
from datetime import datetime
from datetime import timedelta


def ls(arg: str | Path,
       sortby: Literal['name', 'path', 'create', 'modify', 'access', 'size'] = 'path',
       kind: Literal['file', 'dir', 'both'] = 'both',
       created_days_ago: float = None,
       modified_days_ago: float = None,
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

    created_dt = timedelta(days=created_days_ago) if created_days_ago is not None else None
    modified_dt = timedelta(days=modified_days_ago) if modified_days_ago is not None else None
    if created_dt is not None or modified_dt is not None:
        now_t = datetime.now()
        paths_list_ = []
        for pth in paths_list:
            pth_stats = stats(pth)
            if created_dt is not None and (now_t - pth_stats['create']) > created_dt:
                continue
            if modified_dt is not None and (now_t - pth_stats['modify']) > modified_dt:
                continue
            paths_list_.append(pth)
        paths_list = paths_list_

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
