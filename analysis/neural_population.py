import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from common.utils import sigproc
from data_manager import DataMgr
from common.utils.typings import *
from common.utils import stats
from common.utils import dlutils
from enum import Enum
from common.metric_learning import embedding_eval
from common.utils import strtools
import paths
from sklearn.decomposition import PCA
from cv_results_mgr import group_models_by_config, get_chosen_pop_specs, get_chosen_model_per_monkey


class NEURAL_POP(Enum):
    MAJORITY = 'MAJORITY'
    MINORITY = 'MINORITY'
    MIDMAJ = 'MIDMAJ'
    FULL = 'FULL'

    def is_(self, other) -> bool:
        return (self == other or
                (NEURAL_POP.FULL in (self, other)) or
                (self == NEURAL_POP.MIDMAJ and other == NEURAL_POP.MAJORITY))

    def __lt__(self, other):
        if not isinstance(other, NEURAL_POP):
            return NotImplemented
        order = {NEURAL_POP.MINORITY: 0, NEURAL_POP.MAJORITY: 1, NEURAL_POP.MIDMAJ: 2, NEURAL_POP.FULL: 3}
        return order[self] < order[other]

    def __str__(self):
        return self.value


def get_importance_file(model_file) -> Path:
    return paths.PROCESSED_DIR / (Path(model_file).stem + '.IMPORTANCE.json')


def get_neural_population(model_file, importance_method: str, importance_power: float, split_method: str):
    items = json.load(get_importance_file(model_file).open('r'))
    neurons = [item['neuron'] for item in items['importances']]
    importance = np.array([item[importance_method] for item in items['importances']]) ** importance_power
    population = split_populations_by_importance(importance, method=split_method)
    data = pd.DataFrame({'neuron': neurons, 'importance': importance, 'population': population})
    return data, items['input_neuron_names']


def split_populations_by_importance(importances: np.ndarray[float], method: str) -> list[NEURAL_POP]:
    iqr_thresh = 1.

    if method == 'otsu':
        inliers_mask = importances <= sigproc.otsu_threshold(importances)
    else:
        thresh = iqr_thresh if method == 'iqr' else None
        inliers_mask = stats.Inliers(method, thresh=thresh, side='r').is_inlier(importances)

    # from common.utils import plotting
    # plotting.plot(importances, 'o', hue=inliers_mask, hue_labels={True: 'IN', False: 'OUT'})
    # plt.show()
    #
    if np.all(inliers_mask):
        inliers_mask[np.argmax(importances)] = False
    # elif (~inliers_mask).sum() < n_min:
    #     inliers_mask[np.argsort(importances)[-n_min:]] = False

    n_mid = min(inliers_mask.sum(), (~inliers_mask).sum())
    inlier_ixs = np.nonzero(inliers_mask)[0]
    med = np.median(importances[inlier_ixs])
    mid_ixs = inlier_ixs[np.argsort(np.abs(importances[inlier_ixs] - med))[:n_mid]]

    labels = [NEURAL_POP.MAJORITY if is_inlier else NEURAL_POP.MINORITY
              for is_inlier in inliers_mask]
    for i in mid_ixs:
        labels[i] = NEURAL_POP.MIDMAJ

    return labels


class NeuralPopulation:

    def __init__(self, data: pd.DataFrame, input_neuron_names: Sequence[str], spec: dict):
        self.data = data.copy()
        self.input_neuron_names = np.array(input_neuron_names)
        self._population = dict(zip(self.data['neuron'], self.data['population']))
        self._importance = dict(zip(self.data['neuron'], self.data['importance']))
        if spec['importance_power'] == int(spec['importance_power']):
            spec['importance_power'] = int(spec['importance_power'])
        self._spec = spec

    @classmethod
    def from_model(cls, model_file, spec: str | dict = 'chosen'):
        if isinstance(spec, str):
            if spec == 'chosen':
                spec = get_chosen_pop_specs()
            else:
                spec = spec.split('.')
                spec = {'importance_method': spec[0], 'split_method': spec[1], 'importance_power': float(spec[2])}
        data, input_neuron_names = get_neural_population(model_file, **spec)
        return cls(data=data, input_neuron_names=input_neuron_names, spec=spec)

    def __str__(self):
        s = strtools.parts(MINORITY=len(self.neurons(NEURAL_POP.MINORITY)),
                           MAJORITY=len(self.neurons(NEURAL_POP.MAJORITY)))
        return f"NeuralPop {s}"

    @property
    def spec_str(self) -> str:
        return '{importance_method}.{split_method}.{importance_power}'.format(**self.spec_dict)

    @property
    def spec_dict(self) -> dict:
        return self._spec

    @property
    def population(self) -> dict[str, NEURAL_POP]:
        return self._population

    @property
    def importance(self) -> dict[str, float]:
        return self._importance

    def dispname(self, neuron: str) -> str:
        sfx = '*' if self.population[neuron] == NEURAL_POP.MINORITY else '-'
        return f"{self.normalized_importance[neuron]:2.3f}{sfx}"

    def neurons(self, pop: NEURAL_POP = NEURAL_POP.FULL) -> list[str]:
        """ get neurons by population """
        return [neuron for neuron, neuron_pop in self.population.items() if neuron_pop.is_(pop)]

    def inputs_mask(self, pop: list[str] | NEURAL_POP = NEURAL_POP.FULL) -> np.ndarray[bool]:
        if isinstance(pop, NEURAL_POP):
            pop = self.neurons(pop)
        assert isinstance(pop, list)
        mask = np.fromiter((neuron in pop for neuron in self.input_neuron_names), bool)
        assert np.any(mask)
        return mask

    @property
    def normalized_importance(self) -> dict[str, float]:
        max_weight = self.data['importance'].to_numpy().max()
        return {neuron: weight / max_weight for neuron, weight in self.importance.items()}

    def draw(self, neurons_to_highlight: list[str] = None, show_names: bool = True, show_legend: bool = False):
        xlm_margin = 2
        ylm_margin_factor = .1

        if neurons_to_highlight is None:
            neurons_to_highlight = []

        palette = {NEURAL_POP.MAJORITY: 'black', NEURAL_POP.MINORITY: 'red', NEURAL_POP.MIDMAJ: 'blue'}
        data = self.data.copy()
        data['population'] = [NEURAL_POP.MINORITY if pop == NEURAL_POP.MINORITY else NEURAL_POP.MAJORITY
                              for pop in data['population']]

        w = data['importance'].to_numpy()
        ylm_margin = (w.max() - w.min()) * ylm_margin_factor

        jp = sns.jointplot(data=data, x=self.data.index, y='importance', hue='population', kind='scatter',
                           palette=palette, alpha=.75, marginal_kws={'bw_adjust': .99})
        jp.ax_marg_x.remove()
        jp.ax_joint.set_xlim([-xlm_margin, len(self.neurons()) + xlm_margin - 1])
        jp.ax_joint.set_ylim([w.min() - ylm_margin, w.max() + ylm_margin])
        if show_names:
            for r in self.data.loc[:].itertuples():
                if r.population == NEURAL_POP.MINORITY or r.neuron in neurons_to_highlight:
                    plt.text(r.Index, r.importance, r.neuron, color=palette[r.population])

        if neurons_to_highlight:
            data_ = self.data.loc[self.data['neuron'].isin(neurons_to_highlight)]
            jp.ax_joint.plot(data_.index, data_['importance'], 'w+')

        if not show_legend:
            jp.ax_joint.get_legend().remove()

        #plt.title("Contribution of Neurons to Embedded Representation\n" + str(model_file) + "\n" + str(self))
        #plt.title("Contribution of Neurons to Embedded Representation\n" + str(model_file) + "\n" + str(self))
        plt.xlabel("Neuron Index")

    def filter_inputs(self, vecs: np.ndarray, include: NEURAL_POP = None, exclude: NEURAL_POP = None):
        if include is not None:
            assert exclude is None
            nullify_mask = ~self.inputs_mask(include)
        else:
            nullify_mask = self.inputs_mask(exclude)
        vecs = vecs.copy()
        vecs[:, nullify_mask] = .0
        return vecs


def calc_and_save_neural_importances(model_files: list):
    # -----
    max_n_pairs = 100_000
    pca_var_thresh = .95
    pca_n_samples = 10_000
    # -----

    models, cfgs = group_models_by_config(model_files)
    total_models_count = sum(len(v) for v in models.values())

    def _calc_auc_for_embedding(x_, pairs_) -> float:
        embedded_dists2 = -embedding_eval.pairs_dists2(x_, pairs=pairs_)
        return roc_auc_score(y_true=is_same, y_score=embedded_dists2)

    count = 0
    for cfg_id in cfgs:

        cfg = cfgs[cfg_id]
        data_mgr = DataMgr(cfg.data)
        pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)
        is_same = pairs_df['isSame'].to_numpy(dtype=int)
        pairs = pairs_df[['seg1', 'seg2']].to_numpy()
        input_vecs, inputs_meta = data_mgr.get_inputs()
        input_neuron_names = np.array(inputs_meta['input_neuron_names'])
        neuron_names = sorted(set(input_neuron_names))

        for model_file, model in models[cfg_id]:
            count += 1
            print(f"{count}/{total_models_count} {model_file}")

            x0 = dlutils.safe_predict(model, input_vecs)
            E = list(model.parameters())[0].cpu().detach().numpy().copy()
            U, S, Vt = np.linalg.svd(E, full_matrices=False)

            svd_square_loadings = np.sum(np.square(Vt.T) * np.square(S), axis=1)
            full_auc = _calc_auc_for_embedding(x0, pairs)

            pca_results = {'pcaR1': {}, 'pcaR2': {}, 'pcaD1': {}, 'pcaD2': {}}
            for pc_kind in ['D', 'R']:
                xx = x0 if pc_kind == 'D' else np.random.default_rng(1).standard_normal(size=(pca_n_samples, E.shape[0]))
                pca = PCA().fit(np.dot(xx, E))
                n_pcs = np.argmax(np.cumsum(pca.explained_variance_ratio_) > pca_var_thresh) + 1
                for neuron in neuron_names:
                    v = np.zeros((1, pca.n_features_in_), float)
                    v[:, input_neuron_names == neuron] = 1.
                    v = pca.transform(v)[:, :n_pcs] ** 2
                    pca_results[f'pca{pc_kind}1'][neuron] = np.sqrt(v).mean()
                    pca_results[f'pca{pc_kind}2'][neuron] = v.mean()

            importance_items = []
            for neuron in neuron_names:
                print("Neuron", neuron)
                neuron_mask = input_neuron_names == neuron
                vecs = input_vecs.copy()
                vecs[:, neuron_mask] = .0
                x = dlutils.safe_predict(model, vecs)
                importance_items.append({
                    'neuron': neuron,
                    'auc': (full_auc - _calc_auc_for_embedding(x, pairs)) / full_auc,
                    'dist1': np.sqrt(np.sum((x - x0) ** 2, axis=1)).mean(),
                    'dist2': np.sum((x - x0) ** 2, axis=1).mean(),
                    'pcaD1': pca_results['pcaD1'][neuron],
                    'pcaD2': pca_results['pcaD2'][neuron],
                    'pcaR1': pca_results['pcaR1'][neuron],
                    'pcaR2': pca_results['pcaR2'][neuron],
                    'svd1_max': np.max(svd_square_loadings[neuron_mask] ** .5),
                    'svd1_avg': np.mean(svd_square_loadings[neuron_mask] ** .5),
                    'svd2_max': np.max(svd_square_loadings[neuron_mask]),
                    'svd2_avg': np.mean(svd_square_loadings[neuron_mask])
                })

            with get_importance_file(model_file).open('w') as f:
                json.dump({'input_neuron_names': input_neuron_names.tolist(), 'importances': importance_items}, f)


if __name__ == "__main__":
    for monkey, model_file in get_chosen_model_per_monkey().items():
        neural_pop = NeuralPopulation.from_model(model_file)
        neural_pop.draw(show_names=False, show_legend=False)
        plt.title("Contribution of Neurons to Embedded Representation - " + monkey)
    plt.show()
