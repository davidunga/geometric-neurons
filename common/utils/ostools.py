from pathlib import Path
from typing import Literal
from glob import glob
import os


def ls(arg: str | Path,
       sortby: Literal['name', 'path', 'time'] = 'path',
       kind: Literal['file', 'dir', 'both'] = 'both'):

    paths_list = glob(str(arg))

    if kind != 'both':
        paths_list = [pth for pth in paths_list if os.path.isfile(pth) == (kind == 'file')]

    if sortby == 'time':
        paths_list.sort(key=os.path.getmtime)
    elif sortby == 'path':
        paths_list.sort()
    elif sortby == 'name':
        paths_list.sort(key=os.path.basename)

    return paths_list
