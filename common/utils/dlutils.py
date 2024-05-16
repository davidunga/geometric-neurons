""" Deep-learning utilities """
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from common.metric_learning import embedding_models
from common.utils import ostools
from common.utils.typings import *
from common.utils.robust_pickle import get_robust_pickle_module

pickle = get_robust_pickle_module([embedding_models])


def safe_predict(model: torch.nn.Module, x: NDArray | torch.Tensor) -> NDArray:
    is_training = model.training
    model.train(False)
    x = model(torch.as_tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    model.train(is_training)
    return x


def get_optimizer(model_params, kind: str = 'Adam', **optim_kws):
    return getattr(torch.optim, kind)(model_params, **optim_kws)


def get_torch_device(device: str = 'auto') -> str:
    """
    Get the PyTorch device.
    Args:
        device: 'cpu' / 'gpu' / 'auto'
            'gpu' = 'cuda' or 'mps'. raises error if no gpu is available.
            'auto' = 'cuda' or 'mps' or 'cpu', in that order.
    Returns:
        str: The string identifier of the device: ('cuda', 'mps', or 'cpu')
    Raises:
        RuntimeError: If 'gpu' is requested but neither CUDA nor MPS is available.
    """

    assert device in ('cpu', 'gpu', 'auto')
    if device == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if device == 'gpu':
        raise RuntimeError("No GPU support (CUDA or MPS) available.")
    return 'cpu'


def device_count(model):
    return torch.cuda.device_count() if next(model.parameters()).is_cuda else 1


class SnapshotMgr:

    """
    dump & load tagged snapshots of (model, optimizer, meta)
    """

    def __init__(self, file: PathLike, base_meta: dict = None):
        self.file = Path(file)
        self.base_meta = base_meta if base_meta else {}

    def load(self, tag: str) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
        return self.multi_load(tags=[tag])[tag]

    def multi_load(self, tags: list[str] = None) -> dict[str, tuple[torch.nn.Module, torch.optim.Optimizer, dict]]:
        snapshots = self._fetch_all_snapshots()
        tags = tags if tags else snapshots.keys()
        result = {}
        for tag in tags:
            items = snapshots[tag]
            model = items['model']
            model.load_state_dict(items['state_dict'])
            optimizer = items['optimizer']
            if optimizer:
                optimizer.load_state_dict(items['optimizer_state'])
            meta = items['meta']
            result[tag] = (model, optimizer, meta)
        return result

    def get_metas(self) -> dict:
        snapshots = self._fetch_all_snapshots()
        return {tag: snapshots[tag]['meta'] for tag, items in snapshots.items()}

    def set_metas(self, metas: dict):
        snapshots = self._fetch_all_snapshots()
        for tag, meta in metas.items():
            snapshots[tag]['meta'] = meta
        self._safe_dump_snapshots(snapshots, allow_new_tag=False)

    def dump(self,
             tag: str,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer = None,
             meta: dict = None,
             allow_new_tag: bool = True):

        items = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'meta': deepcopy(self.base_meta)
        }

        if meta:
            items['meta'].update(meta)

        self._safe_dump_snapshots({tag: items}, allow_new_tag)

    def _fetch_all_snapshots(self) -> dict:
        if self.file.is_file() and self.file.stat().st_size:
            return torch.load(str(self.file), pickle_module=pickle)
        else:
            return {}

    def wipe(self, tags: list[str] = None):
        snapshots = self._fetch_all_snapshots()
        tags = tags if tags else list(snapshots.keys())
        for tag in tags:
            del snapshots[tag]
        if not snapshots:
            # if no snapshots, clear file completely
            os.remove(str(self.file))
            self.file.touch()
        else:
            torch.save(snapshots, str(self.file))

    def _safe_dump_snapshots(self, snapshots: dict, allow_new_tag: bool):
        base_snapshots = self._fetch_all_snapshots()
        for tag, items in snapshots.items():
            assert allow_new_tag or (tag in base_snapshots)
            SnapshotMgr.validate_snapshot_tag_and_items(tag=tag, items=items)
        base_snapshots.update(snapshots)
        torch.save(base_snapshots, str(self.file))

    @staticmethod
    def validate_snapshot_tag_and_items(tag: str, items: dict):

        # validate tag:
        assert isinstance(tag, str)
        assert "*" not in tag

        # validate items
        assert set(items.keys()) == {'model', 'state_dict', 'optimizer', 'optimizer_state', 'meta'}
        assert isinstance(items['model'], torch.nn.Module)
        assert isinstance(items['state_dict'], dict)
        assert (items['optimizer'] is None) == (items['optimizer_state'] is None)
        assert items['optimizer'] is None or isinstance(items['optimizer'], torch.optim.Optimizer)
        assert items['optimizer_state'] is None or isinstance(items['optimizer_state'], dict)
        assert items['meta'] is None or isinstance(items['meta'], dict)


class checkpoint:

    @staticmethod
    def load(model_file: PathLike):
        items = checkpoint._safe_load_items(model_file)
        model = items['model']
        model.load_state_dict(items['state_dict'])
        optimizer = items['optimizer']
        if optimizer:
            optimizer.load_state_dict(items['optimizer_state'])
        meta = items['meta']
        return model, optimizer, meta

    @staticmethod
    def get_meta(model_file: PathLike):
        return checkpoint._safe_load_items(model_file)['meta']

    @staticmethod
    def set_meta(model_file: PathLike, meta):
        items = checkpoint._safe_load_items(model_file)
        items['meta'] = meta
        checkpoint._safe_save_items(model_file, items)

    @staticmethod
    def update_meta(model_file: PathLike, **kwargs) -> dict:
        items = checkpoint._safe_load_items(model_file)
        items['meta'].update(kwargs)
        checkpoint._safe_save_items(model_file, items)
        return items

    @staticmethod
    def dump(model_file: PathLike, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer = None, meta: dict = None):
        items = {'model': model,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer,
                 'optimizer_state': optimizer.state_dict() if optimizer else None,
                 'meta': meta}
        checkpoint._safe_save_items(model_file, items)

    @staticmethod
    def _validate_items(items: dict):
        assert set(items.keys()) == {'model', 'state_dict', 'optimizer', 'optimizer_state', 'meta'}
        assert isinstance(items['model'], torch.nn.Module)
        assert isinstance(items['state_dict'], dict)
        assert (items['optimizer'] is None) == (items['optimizer_state'] is None)
        assert items['optimizer'] is None or isinstance(items['optimizer'], torch.optim.Optimizer)
        assert items['optimizer_state'] is None or isinstance(items['optimizer_state'], dict)
        assert items['meta'] is None or isinstance(items['meta'], dict)

    @staticmethod
    def _safe_load_items(model_file: PathLike) -> dict:
        items = torch.load(str(model_file))
        checkpoint._validate_items(items)
        return items

    @staticmethod
    def _safe_save_items(model_file: PathLike, items: dict):
        checkpoint._validate_items(items)
        torch.save(items, str(model_file))


class ProgressManager:

    class StopCriterion:
        stop_if: Literal['low', 'high']

        def __init__(self, thresh: float):
            assert self.stop_if in ('low', 'high')
            self.thresh = thresh
            self.value: float = .0
            self.count: int = 0
            self._last_val_loss: float = None
            self._last_train_loss: float = None

        @staticmethod
        def calc(val_loss: float, train_loss: float, last_val_loss: float, last_train_loss: float) -> float:
            raise NotImplementedError()

        def update(self, val_loss: float, train_loss: float, enable_count: bool) -> None:
            self.value = self.calc(val_loss, train_loss, self._last_val_loss, self._last_train_loss)
            self._last_val_loss = val_loss
            self._last_train_loss = train_loss
            if enable_count and ((self.value > self.thresh) == (self.stop_if == 'high')):
                self.count += 1
            else:
                self.count = 0

    class StopCriterionConverge(StopCriterion):
        stop_if = 'low'

        @staticmethod
        def calc(val_loss: float, train_loss: float, last_val_loss: float, last_train_loss: float) -> float:
            return 1. if last_val_loss is None else (last_val_loss - val_loss) / last_val_loss

    class StopCriterionOverfit(StopCriterion):
        stop_if = 'high'

        @staticmethod
        def calc(val_loss: float, train_loss: float, last_val_loss: float, last_train_loss: float) -> float:
            return val_loss / max(train_loss, 1e-9) - 1

    def __init__(self, patience=5, converge: float = .001, overfit: float = .2, epochs: int = None, grace_period=0):
        """
        Args:
            patience: number of steps (epochs) to wait before stopping, once a threshold values is met. None = never.
                if patience value is between 0 & 1, patience = round(value * epochs)
            converge: convergence threshold
                convergence = (last_val_loss - val_loss) / last_val_loss
            overfit: overfit threshold
                overfit = (val_loss - train_loss) / train_loss
            epochs: total number of training epochs
            grace_period: steps to wait before starting. None = same as patience.
        """

        if patience and 0 < patience < 1:
            assert epochs is not None, "Patience given as fraction of epochs, but epochs was not specified."
            patience = int(round(patience * epochs))

        if grace_period is None:
            grace_period = patience if patience is not None else 0
        elif 0 < grace_period < 1:
            assert epochs is not None, "Grace period given as fraction of epochs, but epochs was not specified."
            grace_period = int(round(grace_period * epochs))

        self.patience = patience
        self.epochs = epochs
        self.grace_period = grace_period
        self._best_val_score = None
        self._epoch = 0

        self._stop_reason = ''
        self._stop_epoch = None
        self._is_new_nest = False

        self.criteria: dict[str, ProgressManager.StopCriterion] = {
            'converge': ProgressManager.StopCriterionConverge(thresh=converge),
            'overfit': ProgressManager.StopCriterionOverfit(thresh=overfit)
        }

    @property
    def should_stop(self) -> bool:
        return len(self._stop_reason) > 0

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def stop_epoch(self) -> int:
        return self._stop_epoch

    @property
    def is_new_best(self) -> bool:
        return self._is_new_nest

    def report(self) -> str:
        s = ""
        for criterion in self.criteria:
            s += "{:s}:{:+2.3f} ".format(criterion, self.criteria[criterion].value)
        s = "[" + s[:-1] + "]"
        if self.is_new_best:
            s += " -- New best"
        return s

    @property
    def status_dict(self) -> dict:
        return {'stopped': self.should_stop,
                'state': 'ongoing' if not self.should_stop else f'stopped:{self.stop_reason}',
                'epoch': self._epoch}

    def process(self, val_loss: float, train_loss: float, val_score: float, epoch: int = None) -> None:

        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1
        epoch = self._epoch

        self._is_new_nest = False
        self._stop_reason = ''

        for criterion in self.criteria:
            self.criteria[criterion].update(val_loss, train_loss, enable_count=epoch >= self.grace_period)
            if self.patience is not None and self.criteria[criterion].count > self.patience:
                self._stop_reason = criterion
                self._stop_epoch = epoch

        if self._best_val_score is None or val_score > self._best_val_score:
            self._is_new_nest = True
            self._stop_reason = ''
            self._stop_epoch = None
            self._best_val_score = val_score

        if self.epochs is not None and (epoch + 1) >= self.epochs:
            self._stop_reason = 'last_epoch'
            self._stop_epoch = epoch


class BatchManager:

    def __init__(self, items: int | Sequence,
                 batch_size: int = 64,
                 batches_in_epoch: int = None,
                 shuffle: bool = True):

        self._items = np.arange(items) if isinstance(items, int) else items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch_items = self._items
        self.batches_in_epoch = len(self._items) // self.batch_size if batches_in_epoch is None else batches_in_epoch

    def init_epoch(self, epoch: int | None = None) -> None:
        if self.shuffle:
            assert epoch is not None
            self.epoch_items = np.random.default_rng(epoch).permutation(self._items)
        else:
            assert epoch is None

    def get_items(self, batch: int) -> Sequence:
        assert 0 <= batch < self.batches_in_epoch, "Batch out of range"
        start = batch * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.epoch_items):
            start = 0
            stop = self.batch_size
        return self.epoch_items[start: stop]



from datetime import timedelta, datetime


class TrainingLifeSign:
    def __init__(self, log_dir: PathLike, training_id: str, tolerance: (float, timedelta) = 10.):
        self.tolerance = timedelta(seconds=tolerance) if isinstance(tolerance, float) else tolerance
        self.log_file = Path(log_dir) / (training_id + ".lifeSign")
        self.last_refresh = datetime(year=1970, month=1, day=1)
        if self.log_file.is_file():
            self.last_refresh = datetime.fromtimestamp(ostools.stats(self.log_file)['modify'])

    def time_since_last_seen(self) -> timedelta:
        return datetime.now() - self.last_refresh

    def is_alive(self) -> bool:
        return self.time_since_last_seen() <= self.tolerance

    def refresh(self):
        self.log_file.touch()


def load_tensorboard_as_df(tbdir: PathLike, smooth_sigma: float = 2) -> pd.DataFrame:
    import tbparse
    from scipy.ndimage import gaussian_filter1d

    reader = tbparse.SummaryReader(str(tbdir), extra_columns={'dir_name'})
    df = reader.scalars
    df['split'] = [dn.split('_')[-1] for dn in df['dir_name']]
    smoothed = df['value'].to_numpy().copy()
    for dir_name in df['dir_name'].unique():
        ii = np.nonzero(df['dir_name'] == dir_name)[0]
        smoothed[ii] = gaussian_filter1d(smoothed[ii], sigma=smooth_sigma, mode='nearest')
    df['smoothed_value'] = smoothed

    return df





def plot_tensorboard(tbdir: PathLike, stat_win_size: int = 10, title_suffix: str = '', txt: str = ''):
    import tbparse
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    from itertools import product

    params = {'legend.fontsize': 'x-small',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'x-small',
              'ytick.labelsize': 'x-small'}
    pylab.rcParams.update(params)

    def remove_inner_labels(axs, keep_legend: tuple | str = (0, 0)):
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)
        if not isinstance(keep_legend, str):
            keep_legend = tuple([k if k >= 0 else axs.shape[i] + k for i, k in enumerate(keep_legend)])
        for i, j in product(range(axs.shape[0]), range(axs.shape[1])):
            if i != axs.shape[0] - 1:
                axs[i, j].set_xlabel(None)
            if j != 0:
                axs[i, j].set_ylabel(None)
            if keep_legend != 'all' and keep_legend != (i, j):
                axs[i, j].get_legend().remove()

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


    def set_out_labels(axs, x: str | list = None, y: str | list = None):
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)
        for ax in axs:
            ax.set_xlabel(None)
            ax.set_ylabel(None)

        def _set_labels(xory: str, axs_, labels_):
            assert xory in ('x', 'y')
            if labels_ is None:
                return
            if isinstance(labels_, str):
                labels_ = [labels_] * len(axs_)
            assert len(axs_) == len(labels_)
            for ax_, label_ in zip(axs_, labels_):
                if xory == 'x':
                    ax_.set_xlabel(label_)
                else:
                    ax_.set_ylabel(label_)

        _set_labels('x', axs[-1, :], x)
        _set_labels('y', axs[:, 0], y)


    from scipy.ndimage import gaussian_filter1d

    reader = tbparse.SummaryReader(str(tbdir), extra_columns={'dir_name'})
    df = reader.scalars
    df['split'] = [dn.split('_')[-1] for dn in df['dir_name']]
    smoothed = df['value'].to_numpy().copy()
    for dir_name in df['dir_name'].unique():
        ii = np.nonzero(df['dir_name'] == dir_name)[0]
        smoothed[ii] = gaussian_filter1d(smoothed[ii], sigma=2, mode='nearest')
    df['smoothed_value'] = smoothed

    tags = df['tag'].unique()
    splits = df['split'].unique()
    #
    #
    #
    # def _smooth(v, r: int):
    #     u = np.zeros(len(v), float)
    #     for i in range(len(v)):
    #         i_start = max(i - r, 0)
    #         i_end = min(i + r + 1, len(v))
    #         u[i] = sum(v[i_start: i_end]) / (i_end - i_start + 1)
    #     return u
    # df['smooth_value'] = _smooth(df['value'].to_numpy(), 1) # gaussian_filter1d(df['value'].to_numpy(), sigma=.5,
    # # mode='nearest', truncate=4)

    fig, axs = plt.subplots(nrows=1, ncols=len(tags))
    fig.set_size_inches(15, 6, forward=True)

    for i, tag in enumerate(tags):
        tag_data = df[df['tag'] == tag]
        title_txt = '\n' + tag
        for split in splits:
            v = tag_data[tag_data['split'] == split]['smoothed_value'].to_numpy()[-10:]
            #gaussian_filter1d(v, sigma=1., mode='nearest')
            imprv = np.mean(np.diff(v)/v[:-1]) * 100
            title_txt += f'\n{split:5s} '
            title_txt += f'imprv={imprv:2.3f}'
            #title_txt += f'(min,avg,max) = ({np.min(v):2.2f},{np.mean(v[-stat_win_size:]):2.2f},{np.max(v):2.2f})'

        g = sns.lineplot(data=tag_data, x='step', y='value', hue='split',
                         ax=axs[i], palette=['c', 'm'], alpha=.5)
        g = sns.lineplot(data=tag_data, x='step', y='smoothed_value',
                         hue='split', ax=axs[i], palette=['c', 'm'], alpha=1)
        g.set(title=title_txt)
        g.get_legend().set_title(None)

    remove_inner_labels(axs)
    if txt:
        txt = _text_wrap(txt, 100)
        axs[0].text(0.01, -.1, txt, transform=axs[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=0.2))

    plt.subplots_adjust(top=0.85, bottom=0.15)

    name = '/'.join(Path(reader.log_path).parts[-2:]) + ' ' + title_suffix
    fig.canvas.manager.set_window_title(name)
