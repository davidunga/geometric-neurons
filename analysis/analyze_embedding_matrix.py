import numpy as np
from common.utils import dlutils
from analysis import cv_results_mgr
import matplotlib.pyplot as plt
from common.utils import plotting
from analysis.data_manager import DataMgr
import torch
from common.metric_learning.embedding_models import LinearEmbedder


def find_elbow_point(squared_loadings):
    """
    Estimate the threshold for retaining components based on a Scree Plot-like approach,
    identifying the 'elbow' in the array of squared loadings without plotting it.

    Args:
    squared_loadings (np.array): Array of squared loadings from PCA.

    Returns:
    int: Estimated number of components to retain based on the elbow criterion.
    """
    # Ensure squared_loadings is a numpy array
    squared_loadings = np.array(squared_loadings)

    # Calculate the angles between each point
    angles = np.array([0] * len(squared_loadings))  # Initialize angles array
    for i in range(1, len(squared_loadings) - 1):
        prev_point = np.array([i - 1, squared_loadings[i - 1]])
        curr_point = np.array([i, squared_loadings[i]])
        next_point = np.array([i + 1, squared_loadings[i + 1]])

        # Form vectors
        v1 = curr_point - prev_point
        v2 = next_point - curr_point

        # Calculate angle using dot product and norms
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles[i] = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip cos_theta to avoid numerical issues

    # Find the index of the maximum angle which corresponds to the elbow
    elbow_index = np.argmax(angles)

    return elbow_index


def concentrate_matrix_variance(w, b=None):
    """ Transform matrix to concentrate variance, maintain original shape """
    has_bias = b is not None
    if has_bias:
        w = np.c_[w, b]
    eigvals, eigvecs = np.linalg.eigh(np.cov(w, rowvar=False))
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    w = np.dot(w, eigvecs)
    if has_bias:
        w, b = w[:, :-1], w[:, -1]
        return w, b
    else:
        return


def set_weight_and_bias(model: LinearEmbedder, w, b):
    linear_layer = model.embedder[1]
    linear_layer.weight = torch.nn.Parameter(torch.tensor(w, dtype=torch.float32))
    linear_layer.bias = torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))


def concentrate_embedder_model(model: LinearEmbedder):
    E, b = dlutils.extract_weight_and_bias_from_linear_model(model)
    Eb = np.c_[E, b]
    E = concentrate_matrix_variance(E)
    E, b = Eb[:, :-1], Eb[:, -1]
    set_weight_and_bias(model, E, b)
    return model


def find_elbow_point2(a, smooth_r=1) -> int:
    a = np.asarray(a)
    if smooth_r > 0:
        size = 1 + 2 * smooth_r
        a = np.convolve(np.pad(a, (smooth_r, smooth_r), mode='edge'), np.ones(size), 'valid') / size
    drv = np.diff(a, n=2)
    elbow_ix = np.argmin(drv) + 1
    return elbow_ix


def calc_squared_loadings(m) -> np.ndarray[float]:
    U, S, Vt = np.linalg.svd(m, full_matrices=False)
    return np.sum(np.square(Vt.T) * np.square(S), axis=1)


def draw_explained_variance(model_file):

    model, cfg = cv_results_mgr.get_model_and_config(model_file)

    _, inputs_meta = DataMgr(cfg.data).get_inputs()
    input_neuron_names = np.array(inputs_meta['input_neuron_names'])

    E, _ = dlutils.extract_weight_and_bias_from_linear_model(model)
    s = calc_squared_loadings(E)
    Ec = concentrate_variance(E)
    sc = calc_squared_loadings(Ec)

    plt.plot(np.cumsum(np.sort(s)[::-1]), 'r.-')
    plt.plot(np.cumsum(np.sort(sc)[::-1]), 'c.-')
    plt.show()

    assert s.shape == input_neuron_names.shape

    #s = np.array([s[input_neuron_names == neuron] for neuron in set(input_neuron_names)])
    ss = np.sort(s)[::-1]
    plt.plot(ss, '.-')
    i = find_elbow_point(ss)
    plt.plot(i, ss[i], 'r*')
    plt.show()

    #plt.plot(np.cumsum(np.sort(s)))
    plt.title(cfg.data.trials.name)


if __name__ == "__main__":
    axs = plotting.subplots(ncols=2)
    for i, (monkey, file) in enumerate(cv_results_mgr.get_chosen_model_per_monkey().items()):
        plt.sca(axs[i])
        draw_explained_variance(file)
    plt.show()
