import pandas as pd
from scipy import sparse


def validate_pairs_dataframe(df: pd.DataFrame):
    assert 'seg1' in df.columns
    assert 'seg2' in df.columns
    assert 'isSame' in df.columns
    assert 'dist' in df.columns
    assert 'num_segments' in df.attrs
    assert 'num_pairs' in df.attrs


def pairs_to_matrix(pairs_df: pd.DataFrame, value_col: str, map_zero_value=None, dtype=None) -> sparse.csr_array:
    validate_pairs_dataframe(pairs_df)
    values = pairs_df[value_col].to_numpy(dtype=dtype)
    if map_zero_value:
        values[values == 0] = map_zero_value
    n = pairs_df.attrs['num_segments']
    i, j = pairs_df[['seg1', 'seg2']].to_numpy().T
    mtx = sparse.coo_array((values, (i, j)), (n, n)).tocsr()
    return mtx
