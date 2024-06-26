import matplotlib.pyplot as plt
import numpy as np

from config import Config
from data_manager import DataMgr
from motorneural import draw_data


def adjust_legend(ax=None, orient: str = 'auto', loc: str = 'best'):

    if ax is None:
        ax = plt.gca()

    handles, labels = ax.get_legend_handles_labels()
    kws = {'fancybox': True, 'shadow': False, 'loc': loc, 'borderaxespad': 0, 'framealpha': .5}
    if orient is not None:
        if orient == 'auto':
            orient = 'h' if ('lower' in loc or 'upper' in loc) else 'v'
        kws['ncol'] = len(labels) if orient == 'h' else 1

    ax.legend(**kws)


if __name__ == "__main__":

    cfg = Config.from_default()
    cfg.data.trials.name = 'TP_RS'
    data_mgr = DataMgr(cfg.data)
    trials, meta = data_mgr.load_trials()
    print(meta)
    for trial_ix in np.round(np.linspace(0, len(trials) - 1, 10)).astype(int):
        draw_data.draw_kin_events(trials[trial_ix], traj=True)
        plt.show()
