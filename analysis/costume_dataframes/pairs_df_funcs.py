import pandas as pd
from scipy import sparse
from common.utils import symmetric_pairs


def validate(df: pd.DataFrame):
    assert 'seg1' in df.columns
    assert 'seg2' in df.columns
    assert 'isSame' in df.columns
    assert 'dist' in df.columns
    assert 'num_segments' in df.attrs
    assert 'num_pairs' in df.attrs


def to_sparse_matrix(pairs_df: pd.DataFrame, value_col: str, map_zero_value=None, dtype=None) -> sparse.csr_array:
    validate(pairs_df)
    values = pairs_df[value_col].to_numpy(dtype=dtype)
    if map_zero_value:
        values[values == 0] = map_zero_value
    n = pairs_df.attrs['num_segments']
    mtx = symmetric_pairs.to_sparse_matrix(values, n, pairs_df[['seg1', 'seg2']].to_numpy())
    return mtx
