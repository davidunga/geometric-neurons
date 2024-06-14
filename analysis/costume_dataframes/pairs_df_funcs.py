import pandas as pd
from scipy import sparse
from common.utils import symmetric_pairs
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import chisquare
from common.utils import strtools


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


def uniform_sample_pairs(pairs_df: pd.DataFrame, max_n_pairs: int):

    pairs = pairs_df[['seg1', 'seg2']].to_numpy().astype(int)
    is_same = pairs_df['isSame'].to_numpy().astype(int)

    rng = np.random.default_rng(1)
    target_n_pairs = min(len(pairs), max_n_pairs)
    target_n_same = int(is_same.mean() * target_n_pairs)
    target_n_notSame = target_n_pairs - target_n_same

    segment_counts = Counter(pairs.flatten())
    n_segments = len(segment_counts)
    pair_counts = np.fromiter((min(segment_counts[i], segment_counts[j]) for i, j in pairs), float)

    target_count = {0: target_n_notSame / n_segments + 1, 1: target_n_same / n_segments + 1}
    counts = {0: {i: 0 for i in segment_counts}, 1: {i: 0 for i in segment_counts}}

    is_chosen = np.zeros(len(pairs), bool)
    for ix in np.argsort(pair_counts):
        a, b = pairs[ix]
        label = is_same[ix]
        if max(counts[label][a], counts[label][b]) < target_count[label]:
            counts[label][a] += 1
            counts[label][b] += 1
            is_chosen[ix] = True

    n_same = is_same[is_chosen].sum()
    n_notSame = is_chosen.sum() - n_same

    cand_ixs = np.nonzero(~is_chosen & is_same)[0]
    is_chosen[rng.permutation(cand_ixs)[:target_n_same - n_same]] = True

    cand_ixs = np.nonzero(~is_chosen & ~is_same)[0]
    is_chosen[rng.permutation(cand_ixs)[:target_n_notSame - n_notSame]] = True

    ixs = np.nonzero(is_chosen)[0]
    return ixs


def report_segment_uniformity(pairs_df: pd.DataFrame):
    observed_freq = np.array(list(Counter(pairs_df[['seg1', 'seg2']].to_numpy().flatten()).values()), float)
    uniform_freq = np.sum(observed_freq) / len(observed_freq)
    chi_stat, p_value = chisquare(f_obs=observed_freq, f_exp=np.ones_like(observed_freq) * uniform_freq)
    mape = np.abs((observed_freq - uniform_freq) / uniform_freq).mean()
    print("Segment uniformity: ", end="")
    print(f" Chi2 stat={chi_stat:2.3f}, p-value={p_value:2.5f}")
    print(f" MAPE relative to uniform={mape:2.3f}")


def report_sameness_part(pairs_df: pd.DataFrame, raise_unbalanced: bool = False):
    tol = .01
    n_same = pairs_df['isSame'].sum()
    print(strtools.parts(Same=pairs_df['isSame'].sum(), NotSame=len(pairs_df) - n_same))
    if raise_unbalanced:
        p_same = n_same / len(pairs_df)
        assert abs(.5 - p_same) < tol, p_same

