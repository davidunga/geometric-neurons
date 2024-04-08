from motorneural.data import DataSlice, Segment, Trial, KinData, NeuralData
import numpy as np
import matplotlib.pyplot as plt
from common.utils.typings import *


def draw_neural_rates(s: DataSlice, trace: bool = True):

    rng = np.random.default_rng(1)
    plt.subplots(figsize=(12, 8))

    num_neurons = s.neural.num_neurons
    rates = s.neural[:].T  # rows = neuron, cols = time bin
    nrow, ncol = 6, 5
    h, w = nrow - 2, ncol - 1

    # Raster plot
    ax1 = plt.subplot2grid((nrow, ncol), (0, 0), colspan=w, rowspan=h)
    ax1.imshow(rates, aspect='auto', origin='lower', cmap='gray')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neuron')
    plt.title(str(s))

    # event lines
    for i, (event_name, event_ix) in enumerate(s.events.items()):
        p = ax1.plot([event_ix, event_ix], [0, num_neurons - 1])
        x = max(1, min(event_ix, .98 * rates.shape[1]))
        y = 5 + i * num_neurons / len(s.events)
        plt.text(x, y, event_name, color=p[0].get_color())

    # Average per neuron
    ax2 = plt.subplot2grid((nrow, ncol), (0, w), rowspan=h, sharey=ax1)
    ax2.plot([0, 0], [0, num_neurons - 1], color='gray')
    ax2.plot(np.mean(rates, axis=1), range(num_neurons), 'b-')
    ax2.set_title('Avg over time')
    ax2.invert_yaxis()  # Invert y-axis to align with the left plot

    # Population average vs time
    ax3 = plt.subplot2grid((nrow, ncol), (h, 0), colspan=h, sharex=ax1)
    ax3.plot(np.mean(rates, axis=0), 'b-')
    ax3.set_title('Avg over population')
    ax3.set_xlabel('Time')

    # Speed profile
    ax4 = plt.subplot2grid((nrow, ncol), (nrow - 1, 0), colspan=ncol - 1, sharex=ax1)
    ax4.plot(s.kin['EuSpd'], 'r-')
    ax4.set_title('Speed profile')
    ax4.set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.92)





def draw_trajectory(xy: NpPoints):
    ax = plt.gca()
    plt.plot(xy.T)
    ax.set_aspect('equal', adjustable='box')
