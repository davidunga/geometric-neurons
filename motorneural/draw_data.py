from motorneural.data import DataSlice, Segment, Trial, KinData, NeuralData
import numpy as np
import matplotlib.pyplot as plt
from common.utils.typings import *


def draw_neural_raster(s: DataSlice, trace: bool = True):

    rng = np.random.default_rng(1)
    plt.subplots(figsize=(12, 8))

    num_neurons = s.neural.num_neurons
    raster = s.neural[:].T  # rows = neuron, cols = time bin
    from common.utils.sigproc import normalize
    #raster = normalize(raster, axis=1, kind='max')
    n = 6  # plot subdivision

    # Raster plot
    ax1 = plt.subplot2grid((n, n), (0, 0), colspan=n - 1, rowspan=n - 1)
    ax1.imshow(raster, aspect='auto', origin='lower', cmap='gray')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neuron')
    plt.title(str(s))

    # event lines
    text_ymin = 5
    text_ymax = num_neurons - 5
    #s.kin.events = s.kin.events
    for i, (event_name, event_ix) in enumerate(s.events.items()):
        p = ax1.plot([event_ix, event_ix], [0, num_neurons - 1])
        x = max(1, min(event_ix, .98 * raster.shape[1]))
        y = text_ymin + i * (text_ymax - text_ymin) / len(s.events)
        plt.text(x, y, event_name, color=p[0].get_color())

    # Average firing rate
    ax2 = plt.subplot2grid((n, n), (0, n - 1), rowspan=n - 1, sharey=ax1)
    ax2.plot([0, 0], [0, num_neurons - 1], color='gray')
    ax2.plot(np.mean(raster, axis=1), range(num_neurons), 'b-')
    ax2.set_title('Avg Firing Rate')
    ax2.invert_yaxis()  # Invert y-axis to align with the left plot

    # Population trace
    ax3 = plt.subplot2grid((n, n), (n - 1, 0), colspan=n - 1, sharex=ax1)
    ax3.plot(np.mean(raster, axis=0), 'r-')
    ax3.set_title('Population Trace')
    ax3.set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.92)





def draw_trajectory(xy: NpPoints):
    ax = plt.gca()
    plt.plot(xy.T)
    ax.set_aspect('equal', adjustable='box')
