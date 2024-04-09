import os
from collections.abc import MutableMapping
from pathlib import Path

from file_io import FileIO, PickleIO, PandasCsvIO, DynamicFileIO


class FileStore(MutableMapping):
    """
    File based storage with dictionary-like behavior.

    Examples:

        # single file type (pickle):

        fs = FileStore.PickleStore('path/to/store')
        fs['key'] = ..          # dump to 'path/to/store/key.pkl'
        value = fs['key']       # load from 'path/to/store/key.pkl'
        del fs['key']           # delete 'path/to/store/key.pkl'

        # multiple file types (default)
        # keys must be of the form 'keyName.ext' -

        fs = FileStore('path/to/store')
        fs['key1.pkl'] = ..   # dump to 'path/to/store/key1.pkl'
        fs['key2.csv'] = ..   # dump to 'path/to/store/key2.csv'

    """

    def __init__(self, path: Path | str, io: FileIO | dict[str, FileIO] | list[FileIO] = None):
        """
        Args:
            path: storage path (folder)
            io: interface for reading/writing from storage, see DynamicIO
        """
        self.path = Path(path)
        self.io = DynamicFileIO(io)
        self._supported_exts = tuple(self.io.io_dict.keys())
        self._ext = self._supported_exts[0] if len(self._supported_exts) == 1 else ''

    @classmethod
    def PickleStore(cls, path: Path | str):
        return cls(path=path, io=PickleIO())

    @classmethod
    def PandasCsvStore(cls, path: Path | str):
        return cls(path=path, io=PandasCsvIO())

    def file(self, key):
        return self.path / f'{key}{self._ext}'

    def keys(self):
        try:
            filenames = os.listdir(str(self.path))
        except FileNotFoundError:
            filenames = []
        trim = -len(self._ext) if self._ext else None
        return (fn[:trim] for fn in filenames if fn.endswith(self._supported_exts))

    def touch(self):
        self.path.mkdir(exist_ok=True, parents=True)
        self.path.touch()

    def rename(self, key, newkey):
        self.file(key).rename(self.file(newkey))

    def __getitem__(self, key):
        return self.io.read(self.file(key))

    def __setitem__(self, key, obj):
        self.touch()
        self.io.write(self.file(key), obj)

    def __delitem__(self, key):
        os.remove(str(self.file(key)))

    def __len__(self):
        return len(list(self.keys()))

    def __iter__(self):
        return self.keys()
