import json
from common.utils import dictools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from common.utils import sigproc
from data_manager import DataMgr
from common.utils.typings import *
import cv_results_mgr
from common.utils import stats
from common.utils import dlutils
from scipy.spatial.distance import squareform, pdist
from enum import Enum
from common.utils.randtool import Rnd
from common.metric_learning import embedding_eval
from common.utils import strtools
import paths


def get_weights_file(model_file) -> Path:
    return paths.PROCESSED_DIR / (Path(model_file).stem + '.WEIGHTS.json')


class NEURAL_POP(Enum):
    MAJORITY = 'MAJORITY'
    MINORITY = 'MINORITY'
    FULL = 'FULL'

    def is_(self, other) -> bool:
        return self == other or NEURAL_POP.FULL in (self, other)

    def __lt__(self, other):
        if not isinstance(other, NEURAL_POP):
            return NotImplemented
        order = {NEURAL_POP.MINORITY: 0, NEURAL_POP.MAJORITY: 1, NEURAL_POP.FULL: 2}
        return order[self] < order[other]


class NeuralPopulation:

    def __init__(self, data: pd.DataFrame, input_neuron_names: Sequence[str]):
        self.data = data.copy()
        self.input_neuron_names = np.array(input_neuron_names)
        self._population = dict(zip(self.data['neuron'], self.data['population']))
        self._weight = dict(zip(self.data['neuron'], self.data['weight']))

    @classmethod
    def from_model(cls, model_file, weight_by: str = None):
        if weight_by is None: weight_by = 'svd_avg'
        cls_ = get_neural_population(model_file, weight_by=weight_by)
        return cls(data=cls_.data, input_neuron_names=cls_.input_neuron_names)

    def __str__(self):
        s = strtools.parts(MINORITY=len(self.neurons(NEURAL_POP.MINORITY)),
                           MAJORITY=len(self.neurons(NEURAL_POP.MAJORITY)))
        return f"NeuralPop {s}"

    @property
    def population(self) -> dict[str, NEURAL_POP]:
        return self._population

    @property
    def weight(self) -> dict[str, float]:
        return self._weight

    def dispname(self, neuron: str) -> str:
        return f"{self.normalized_weight[neuron]:2.3f}{'*' if self.population[neuron] == NEURAL_POP.MINORITY else '-'}"

    def neurons(self, pop: NEURAL_POP = NEURAL_POP.FULL, n: int = None, ranks: str = 'u') -> list[str]:
        """
        get neurons by population
        Args:
            pop: population name
            n: max number of neurons to get, None = all
            ranks: one of {'t', 'b', 'm', 'u'}. if n is specified, return the n [t]op / [b]ottom / [m]id
                ranked neurons, or [u]niformly sampled ranks.
        Returns:
            list of neuron names
        """
        neurons = [neuron for neuron, neuron_pop in self.population.items() if neuron_pop.is_(pop)]
        if n is None or len(neurons) <= n:
            return neurons
        si = np.argsort([self.weight[neuron] for neuron in neurons])
        match ranks:
            case 't':
                si = si[-n:]
            case 'b':
                si = si[:n]
            case 'm':
                ifm = (len(si) - n) // 2
                si = si[ifm: ifm + n]
            case 'u':
                si = si[np.linspace(0, len(si) - 1, n).round().astype(int)]
            case _:
                raise ValueError("Unknown rank type")
        return [neurons[i] for i in si]

    def inputs_mask(self, pop: list[str] | NEURAL_POP = NEURAL_POP.FULL) -> np.ndarray[int]:
        if isinstance(pop, NEURAL_POP):
            pop = self.neurons(pop)
        assert isinstance(pop, list)
        mask = np.fromiter((neuron in pop for neuron in self.input_neuron_names), bool)
        assert np.any(mask)
        return mask

    @property
    def normalized_weight(self) -> dict[str, float]:
        max_weight = self.data['weight'].to_numpy().max()
        return {neuron: weight / max_weight for neuron, weight in self.weight.items()}

    def draw(self, neurons_to_highlight: list[str] = None):
        xlm_margin = 2

        if neurons_to_highlight is None:
            neurons_to_highlight = []

        palette = {NEURAL_POP.MAJORITY: 'black', NEURAL_POP.MINORITY: 'red'}
        jp = sns.jointplot(data=self.data, x=self.data.index, y='weight', hue='population', kind='scatter',
                           palette=palette, alpha=.75)
        jp.ax_marg_x.remove()
        jp.ax_joint.set_xlim([-xlm_margin, len(self.neurons()) + xlm_margin - 1])
        for r in self.data.loc[:].itertuples():
            if r.population == NEURAL_POP.MINORITY or r.neuron in neurons_to_highlight:
                plt.text(r.Index, r.weight, r.neuron, color=palette[r.population])

        if neurons_to_highlight:
            data_ = self.data.loc[self.data['neuron'].isin(neurons_to_highlight)]
            jp.ax_joint.plot(data_.index, data_['weight'], 'w+')

        plt.title("Contribution of Neurons to Embedded Representation")
        plt.xlabel("Neuron Index")


def calc_and_save_neural_weights(model_file, seed: int = 1):
    max_n_pairs = 100_000

    print("Computing neural weights for " + str(model_file))

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    data_mgr = DataMgr(cfg.data)

    pairs_df = data_mgr.load_pairing(n_pairs=max_n_pairs)
    is_same = pairs_df['isSame'].to_numpy(dtype=int)

    input_vecs, inputs_meta = data_mgr.get_inputs()
    x0 = dlutils.safe_predict(model, input_vecs)

    input_neuron_names = np.array(inputs_meta['input_neuron_names'])
    neuron_names = sorted(set(input_neuron_names))

    E = list(model.parameters())[0].cpu().detach().numpy().copy()
    U, S, Vt = np.linalg.svd(E, full_matrices=False)
    svd_square_loadings = np.sum(np.square(Vt.T) * np.square(S), axis=1)

    def _calc_auc_for_embedding(x_) -> float:
        embedded_dists = -embedding_eval.pairs_dists(x_, pairs=pairs_df[['seg1', 'seg2']].to_numpy())
        return roc_auc_score(y_true=is_same, y_score=embedded_dists)

    full_auc = _calc_auc_for_embedding(x0)

    weight_items = []
    for neuron in neuron_names:
        print("Neuron", neuron)
        neuron_mask = input_neuron_names == neuron
        vecs = input_vecs.copy()
        vecs[:, neuron_mask] = .0
        x = dlutils.safe_predict(model, vecs)
        weight_items.append({
            'neuron': neuron,
            'auc': (full_auc - _calc_auc_for_embedding(x)) / full_auc,
            'dist': np.sum((x - x0) ** 2, axis=1).mean(),
            'svd_max1': np.max(svd_square_loadings[neuron_mask] ** .5),
            'svd_avg1': np.mean(svd_square_loadings[neuron_mask] ** .5),
            'svd_max': np.max(svd_square_loadings[neuron_mask]),
            'svd_avg': np.mean(svd_square_loadings[neuron_mask])
        })

    items = dictools.to_json({'input_neuron_names': input_neuron_names, 'weights': weight_items})

    weights_file = get_weights_file(model_file)
    json.dump(items, weights_file.open('w'))
    print("Saved to " + str(weights_file))


def get_neural_population(model_file, weight_by: str) -> NeuralPopulation:
    items = json.load(get_weights_file(model_file).open('r'))
    input_neuron_names = np.array(items['input_neuron_names'])
    data = pd.DataFrame(items['weights']).rename({weight_by: 'weight'}, axis=1).loc[:, ['neuron', 'weight']]
    data['population'] = NEURAL_POP.FULL
    neural_pop = NeuralPopulation(data=data, input_neuron_names=input_neuron_names)
    neural_pop = cluster_populations_by_weights(neural_pop)
    return neural_pop


def cluster_populations_by_weights(neural_pop: NeuralPopulation, method: str = 'otsu',
                                   add_neighbors: bool = False) -> NeuralPopulation:

    neuron_weight = neural_pop.data['weight'].to_numpy(dtype=float) ** 2

    if method == 'otsu':
        outliers_mask = neuron_weight > sigproc.otsu_threshold(neuron_weight)
    else:
        outliers_mask = stats.Inliers(method).is_outlier(neuron_weight)

    # assert neuron_weight[outliers_mask].min() > neuron_weight[~outliers_mask].max(), \
    #     "Minority population score is not strictly greater than majority's"

    if add_neighbors:
        dists = squareform(pdist(neuron_weight.reshape(-1, 1), 'seuclidean'))
        dists[np.arange(len(dists)), np.arange(len(dists))] = np.inf
        nearest = np.argmin(dists, axis=1)
        outliers_mask[outliers_mask[nearest]] = True

    data = neural_pop.data.copy()
    data.loc[~outliers_mask, 'population'] = NEURAL_POP.MAJORITY
    data.loc[outliers_mask, 'population'] = NEURAL_POP.MINORITY
    #data.sort_values(by='weight', ascending=False, inplace=True, ignore_index=True)
    ret = NeuralPopulation(data=data, input_neuron_names=neural_pop.input_neuron_names)
    return ret


if __name__ == "__main__":
    for monkey, file in cv_results_mgr.get_chosen_model_per_monkey().items():
        weight_bys = ['svd_avg', 'svd_max', 'auc', 'svd_avg1', 'svd_max1']
        #calc_and_save_neural_weights(file)
        for by in weight_bys:
            pop = get_neural_population(file, weight_by=by)
            pop.draw()
            plt.title(monkey + " " + by + "\n" + str(pop))
        #plt.title("Contribution of Neurons to Embedded Representation\n" + monkey)
    plt.show()
