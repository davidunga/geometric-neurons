import tbparse
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import product
import numpy as np
import pandas as pd
from common.utils.typings import *
from common.utils import plotting
from torch.utils.tensorboard import SummaryWriter


def load_tensorboard_as_df(tbdir: PathLike, smooth_sigma: float = 2) -> pd.DataFrame:

    reader = tbparse.SummaryReader(str(tbdir), extra_columns={'dir_name'})
    df = reader.scalars
    df['split'] = [dn.split('_')[-1] for dn in df['dir_name']]
    smoothed = df['value'].to_numpy().copy()
    for dir_name in df['dir_name'].unique():
        ii = np.nonzero(df['dir_name'] == dir_name)[0]
        smoothed[ii] = gaussian_filter1d(smoothed[ii], sigma=smooth_sigma, mode='nearest')
    df['smoothed_value'] = smoothed

    return df


def add_counts_as_histogram(tb: SummaryWriter, counts: Sequence[int], tag: str, global_step: int = None):
    fake_samples = []
    for k, count in enumerate(counts):
        fake_samples += [k] * count
    if not fake_samples:
        return
    fake_samples = np.array(fake_samples)
    bin_edges = np.arange(len(counts))
    bin_edges[0] -= .1
    bin_edges[-1] += .1
    tb.add_histogram(tag, fake_samples, global_step, bins=bin_edges)


def plot_tensorboard(tbdir: PathLike, stat_win_size: int = 10, title_suffix: str = '', txt: str = ''):

    plotting.MplParams.set('small')

    def _text_wrap(txt: str | list, max_line_len: int = 50):
        link_chars = (':', '=')
        split_chars = (' ', ',')

        if isinstance(txt, str):
            txt = txt.split('\n')
        assert isinstance(txt, list)

        wraped_lines = []
        for line in txt:

            splittable_ixs = [0]
            sizes = [0]
            for i in range(1, len(line) - 1):
                if line[i] in split_chars and line[i - 1] not in link_chars and line[i + 1] not in split_chars:
                    sizes.append(i - splittable_ixs[-1])
                    splittable_ixs.append(i)

            split_ixs = [0]
            line_size = 0
            for sz, ix in zip(sizes, splittable_ixs):
                if line_size + sz > max_line_len:
                    split_ixs.append(ix)
                    line_size = 0
                else:
                    line_size += sz
            split_ixs.append(len(line))

            for i in range(len(split_ixs) - 1):
                wraped_lines.append(line[split_ixs[i]: split_ixs[i + 1]])

        return '\n'.join(wraped_lines)

    reader = tbparse.SummaryReader(str(tbdir), extra_columns={'dir_name'})
    df = reader.scalars
    df['split'] = [dn.split('_')[-1] for dn in df['dir_name']]
    smoothed = df['value'].to_numpy().copy()
    for dir_name in df['dir_name'].unique():
        ii = np.nonzero(df['dir_name'] == dir_name)[0]
        smoothed[ii] = gaussian_filter1d(smoothed[ii], sigma=2, mode='nearest')
    df['smoothed_value'] = smoothed

    tags = df['tag'].unique()
    fig, axs = plt.subplots(nrows=1, ncols=len(tags))
    fig.set_size_inches(15, 5, forward=True)

    for i, tag in enumerate(tags):
        tag_data = df[df['tag'] == tag]
        title_txt = '\n' + tag
        g = sns.lineplot(data=tag_data, x='step', y='value', hue='split',
                         ax=axs[i], palette=['c', 'm'], alpha=.5)
        g = sns.lineplot(data=tag_data, x='step', y='smoothed_value',
                         hue='split', ax=axs[i], palette=['c', 'm'], alpha=1, legend=False)
        g.set(title=title_txt)
        g.get_legend().set_title(None)

    plotting.remove_inner_labels(axs)
    if txt:
        txt = _text_wrap(txt, 100)
        axs[0].text(0.01, -.1, txt, transform=axs[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=0.2))

    plt.subplots_adjust(top=0.85, bottom=0.15)

    name = '/'.join(Path(reader.log_path).parts[-2:]) + ' ' + title_suffix
    fig.canvas.manager.set_window_title(name)


if __name__ == "__main__":
    from common.utils import ostools
    from common.utils.dlutils import checkpoint
    from paths import MODELS_DIR
    tbdirs = ostools.ls("/Users/davidu/tensorboard/geometric-neurons/TP_RS*")
    for tbdir in tbdirs[::-1]:
        plot_tensorboard(tbdir)
        #model_file = MODELS_DIR / (Path(tbdir).name + ".pth.checkpt")
        #meta = checkpoint.get_meta(model_file)
        plt.show()