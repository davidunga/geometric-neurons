from common.utils import dlutils
from common.utils.randtool import Rnd
import numpy as np

EMBED_TYPES = ('NO', 'YES', 'RAND')


def prep_embeddings(model, vecs, shuff: bool = False, seed: int = 1, embed_types: list = None) -> dict[str, np.ndarray]:

    if embed_types is None:
        embed_types = EMBED_TYPES

    if shuff:
        vecs = Rnd(seed=seed).shuffle(vecs)

    ret = {}
    for embed in embed_types:
        assert embed in EMBED_TYPES
        if embed == 'RAND':
            ret[embed] = dlutils.safe_predict(dlutils.randomize_weights(model, seed=seed), vecs.copy())
        else:
            ret[embed] = dlutils.safe_predict(model, vecs.copy()) if embed == 'YES' else vecs.copy()
    return ret
