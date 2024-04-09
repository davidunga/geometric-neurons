import io
import os
from pathlib import Path
import pandas as pd
import pickle

"""

Augment pickle with a key-based file storage.

# usual pickle dump/load:
dump(obj, file)
load(file)

# key-based:
dump(obj, file, key1)
dump(obj, file, key2)
load(file, key1)
load(file, [key1, key2])  # returns dict {key1: obj1, key2: obj2}

# key '.' is reserved for referring to the main pickle, i.e.,:
dump(obj, file, '.') <-> dump(obj, file)
load(file, '.') <-> load(file, obj)

# load main pickle and key2:
load(file, ['.', key2])

# load all:
load(file, '*')

"""

__all__ = ['load', 'dump', 'delete', 'rename', 'get_keys', 'get_files', 'get_file', 'get_store_dir']


def load(pkl: str | io.BufferedReader | Path, keys: str | list[str] = '?'):
    """
        load(file) or load(file, '?') - if pickle is a regular pickle -- load it as usual.
            if pickle has a store -- load all keys and return as dict.
        load(file, key) - load specific key
        load(file, [key1, key2, ..]) - load keys, return as dict
        load(file, '*') - load all keys, return as dict
        load(file, '.') - load pickle as usual
    """

    pkl_buffer = None
    if isinstance(pkl, Path):
        pkl = str(pkl)
    elif isinstance(pkl, io.BufferedReader):
        # pkl is an open file object
        pkl_buffer = pkl
        pkl = pkl_buffer.name

    assert isinstance(pkl, str)

    all_keys = get_keys(pkl)
    if keys == '*':
        keys = all_keys
    elif keys == '?':
        keys = '.' if all_keys == ['.'] else all_keys

    bad_keys = set([keys] if isinstance(keys, str) else keys).difference(all_keys)
    if bad_keys:
        all_keys_str = ", ".join(all_keys) if all_keys else "<no keys>"
        raise KeyError("Attempted to load: " + ", ".join(list(bad_keys)) + " But found only: " + all_keys_str)

    def _get(key: str):
        if pkl_buffer is not None and key == '.':
            # (file is already open)
            return pickle.load(pkl_buffer)
        try:
            #from common.utils.robust_pickle import RobustUnpickler2
            #import pickle as pickle2
            #pickle2.Unpickler = RobustUnpickler2
            with open(get_file(pkl, key), 'rb') as f:
                return pickle.load(f)
        except:
            print("Error in " + get_file(pkl, key))
            raise

    if isinstance(keys, str):
        ret = _get(keys)
    else:
        ret = {key: _get(key) for key in keys}

    return ret


def dump(obj, pkl: str | io.BufferedWriter | Path, key: str = '.'):
    """
    dump(obj, file)         # dump to pickle as usual
    dump(obj, file, key)    # store under key
    """
    assert is_valid_key(key)

    if isinstance(pkl, Path):
        pkl = str(pkl)
    elif isinstance(pkl, io.BufferedWriter):
        # pkl is an open file object
        if key == '.':
            pickle.dump(obj, pkl)
            return

        else:
            pkl = pkl.name

    assert isinstance(pkl, str)

    file = get_file(pkl, key)
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(get_file(pkl, key), 'wb') as f:
        pickle.dump(obj, f)

    Path(pkl).touch()


def store(objs, pkl: str | io.BufferedWriter, to: str = 'auto'):
    """
    to: determines where to store objs. either 'keys'/'main'/'auto'
        'keys' -- objs must be a dict with valid keys, i.e.: {key1: obj1, key2: obj2, .. }
            dumps each object under its corresponding key, same as: dump(obj1, pkl, key1); dump(obj2, pkl, key2); ..
        'main' -- dumps to main pickle, same as: dump(objs, pkl, '.')
        'auto' -- sets to='keys' if objs is a valid dict, otherwise 'main'.
    """

    assert to in ('main', 'keys', 'auto')

    is_storable_dict = isinstance(objs, dict) and all(isinstance(key, str) for key in objs)

    if to == 'auto':
        to = 'keys' if is_storable_dict else 'main'

    if to == 'main':
        dump(objs, pkl, '.')
    elif to == 'keys':
        if not is_storable_dict:
            raise ValueError("Objects are not in a valid {key1: obj1, ..} format.")
        for key, obj in objs.items():
            dump(obj, pkl, key)
    else:
        ValueError("Unknown destination")





def delete(pkl: str, key: str):
    """ delete key """
    os.remove(get_file(pkl, key))
    if not os.listdir(get_store_dir(pkl)):
        os.rmdir(get_store_dir(pkl))


def rename(pkl: str, key: str, newkey: str):
    """ rename key """
    assert is_valid_key(newkey)
    os.rename(get_file(pkl, key), get_file(pkl, newkey))


def rename_file(src: str, dst: str):
    os.rename(src, dst)
    os.rename(get_store_dir(src), get_store_dir(dst))


def get_keys(pkl: str) -> list[str]:
    """ get all keys """
    keys = ['.'] if os.path.isfile(pkl) and os.path.getsize(pkl) else []
    if os.path.isdir(get_store_dir(pkl)):
        keys += [key for key in os.listdir(get_store_dir(pkl)) if is_valid_key(key)]
    return keys


def get_files(pkl: str) -> dict[str, str]:
    """ get all files """
    return {key: get_file(pkl, key) for key in get_keys(pkl)}


def get_file(pkl: str, key: str) -> str:
    """ file associated with key """
    return pkl if key == '.' else os.path.join(get_store_dir(pkl), key)


def get_store_dir(pkl: str) -> str:
    return f'{pkl}.store'


def load_dataframe(pkl, keys, axis) -> pd.DataFrame:
    return pd.concat(load(pkl, keys).values(), axis=axis)


def is_valid_key(key: str) -> bool:
    if key == '.':
        return True
    else:
        return not (key.startswith('.') or key in ('*', '?'))


def _run_example():
    import tempfile

    pkl = os.path.join(tempfile.mktemp(), 'myPickle.pkl')
    os.makedirs(os.path.dirname(pkl))

    def _print_files_and_keys():
        root = os.path.dirname(pkl)
        print("-> Files:", {k: fn.replace(root, '') for k, fn in get_files(pkl).items()})

    obj1 = "I am object 1"
    obj2 = ["I", "am", "object", "2"]
    obj3 = {'text': "I am object 3"}

    print("Storing obj1 in main pickle. obj1=", obj1)
    dump(obj1, pkl)
    _print_files_and_keys()

    print("Storing obj2 under key2. obj2=", obj2)
    dump(obj2, pkl, 'key2')
    _print_files_and_keys()

    print("Storing obj3 under key3. obj3=", obj3)
    dump(obj3, pkl, 'key3')
    _print_files_and_keys()

    print("Load from main pickle:", load(pkl))
    print("Load key2:", load(pkl, 'key2'))
    print("Load key2 and key3:", load(pkl, ['key2', 'key3']))
    print("Load key2 and main:", load(pkl, ['key2', '.']))
    print("Load all:", load(pkl, '*'))

    print("Renaming key3 to newKey3")
    rename(pkl, 'key3', 'newKey3')
    _print_files_and_keys()

    print("Deleting all")
    for key in get_keys(pkl):
        delete(pkl, key)
    _print_files_and_keys()


if __name__ == "__main__":
    _run_example()
