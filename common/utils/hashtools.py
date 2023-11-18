import hashlib
import json


def calc_hash(obj, fmt: str = 'hex', algo: str = 'sha1', type_agnostic: bool = True):
    """
    Args:
        obj: object to hash
        fmt: digest format: 'hex'/'int'/'raw'
        algo: hashing algorithm, e.g., 'sha1'/'sha256'/'md5',..
        type_agnostic: if True and obj has __dict__, then: calc_hash(obj) == calc_hash(obj.__dict__)
    """

    if not isinstance(obj, bytes):
        type_str = "" if type_agnostic else str(type(obj))
        if hasattr(obj, '__dict__'):
            obj = obj.__dict__
        obj = type_str + json.dumps(obj)
        obj = obj.encode('utf-8')

    hash_fcn = getattr(hashlib, algo)
    raw_hash = hash_fcn(obj)

    if fmt == 'hex':
        return raw_hash.hexdigest()
    elif fmt == 'int':
        return int.from_bytes(raw_hash.digest(), 'big')
    elif fmt == 'raw':
        return raw_hash
    else:
        raise ValueError("Unknown hash kind")
