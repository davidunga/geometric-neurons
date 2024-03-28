from pathlib import Path
import pandas as pd
import pickle


class FileIO:
    """ Unified interface for file i/o """

    default_ext: str

    @staticmethod
    def write(file: Path, obj):
        raise NotImplementedError

    @staticmethod
    def read(file: Path):
        raise NotImplementedError


class PickleIO(FileIO):
    default_ext = 'pkl'
    @staticmethod
    def write(file: Path, obj):
        with file.open('wb') as f:
            pickle.dump(obj, f)
    @staticmethod
    def read(file: Path):
        with file.open('rb') as f:
            ret = pickle.load(f)
        return ret


class TextIO(FileIO):
    default_ext = 'txt'
    @staticmethod
    def write(file: Path, obj):
        with file.open('w') as f:
            f.write(obj)
    @staticmethod
    def read(file: Path):
        with file.open('r') as f:
            ret = f.read()
        return ret


class PandasCsvIO(FileIO):
    default_ext = 'csv'
    @staticmethod
    def write(file: Path, obj: pd.DataFrame):
        obj.to_csv(file)
    @staticmethod
    def read(file: Path) -> pd.DataFrame:
        return pd.read_csv(file)


class PandasExcelIO(FileIO):
    default_ext = 'xlsx'
    @staticmethod
    def write(file: Path, obj: pd.DataFrame):
        obj.to_excel()
    @staticmethod
    def read(file: Path) -> pd.DataFrame:
        return pd.read_excel(file)


class DynamicFileIO(FileIO):
    """
    Read/write based on file extension
    Examples:
        DynamicIO({'pkl': PickleIO(), 'csv': PandasCsvIO()})
        DynamicIO([PickleIO(), PandasCsvIO()])  # use default extensions for lookup
        DynamicIO(PickleIO()) # similar to PickleIO()
        DynamicIO()  # use a default set of FileIOs
    """

    def __init__(self, io: FileIO | list[FileIO] | dict[str, FileIO] = None):
        if io is None:
            io = [PickleIO(), PandasCsvIO(), PandasExcelIO(), TextIO()]
        elif isinstance(io, FileIO):
            io = [io]
        if isinstance(io, list):
            io = {io_.default_ext: io_ for io_ in io}
        self.io_dict = {(ext if ext.startswith('.') else f'.{ext}'): file_io for ext, file_io in io.items()}

    def get_io(self, file: Path | str) -> FileIO:
        return self.io_dict[Path(file).suffix]

    def write(self, file: Path, obj):
        self.get_io(file).write(file, obj)

    def read(self, file: Path):
        return self.get_io(file).read(file)

