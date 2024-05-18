import shutil
from time import time
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from common.metric_learning.embedding_eval import EmbeddingEvalResult, EmbeddingEvaluator
from common.metric_learning.triplet_sampler import TripletSampler
from common.metric_learning.triplet_loss import TripletLossWithIntermediates
from common.utils import dlutils
from common.utils import tbtools
from common.utils.devtools import progbar
from common.utils.typings import *
from itertools import product
import wandb
from wandb_tools import sync_wandb_run
import auth


def triplet_train(
        train_sampler: TripletSampler,
        val_sampler: TripletSampler,
        inputs: torch.Tensor | NDArray,
        model: torch.nn.Module = None,
        batch_size: int = 64,
        epochs: int = 100,
        batches_in_epoch: int = None,
        n_eval_triplets: int = 2000,
        noise_sigma: float = 0,
        optim_params: dict | str = None,
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_file: Path = None,
        progress_mgr_params: dict = None,
        checkpoint_every: int = 5,
        exists_handling: Literal["warm_start", "overwrite", "skip", "error"] = "error",
        wandb_run: wandb.sdk.wandb_run.Run = None,
        base_meta: dict = None,
        dbg_run: bool = False
):

    #dbg_run = True
    assert exists_handling in ("warm_start", "overwrite", "skip", "error")

    device = dlutils.get_torch_device(device)

    # ------

    MAX_INIT_AUC = .6
    DEFAULT_OPTIMIZER_KIND = 'Adam'

    # ------

    def _add_to_wandb(train_eval: EmbeddingEvalResult, val_eval: EmbeddingEvalResult,
                      epoch: int, hists: dict = None):

        if wandb_run is None:
            return

        _metrics_to_log = ('loss', 'auc', 'tscore')

        items = {}

        if train_eval is not None:
            items.update({f'train_{metric}': getattr(train_eval, metric) for metric in _metrics_to_log})

        if val_eval is not None:
            items.update({f'val_{metric}': getattr(val_eval, metric) for metric in _metrics_to_log})

        if hists is not None:
            for hist_name, hist in hists.items():
                if not isinstance(hist, tuple):
                    bin_edges = np.arange(len(hist) + 1)
                    hist = (hist, bin_edges)
                items[hist_name] = wandb.Histogram(np_histogram=hist)

        wandb_run.log(data=items, step=epoch)
        sync_wandb_run(wandb_run.path)

    def _save_snapshot(kind: str, epoch_: int):
        """
            kind:   'init' = save under 'init', 'best', & 'checkpoint'
                    'best' = save under 'best' & 'checkpoint'
                    'checkpoint' = save only under 'checkpoint'
        """
        meta = {'epoch': epoch_,
                'train_status': progress_mgr.status_dict,
                'val': val_eval.metrics_dict,
                'train': train_eval.metrics_dict}
        tag_hierarchy = ['init', 'best', 'checkpoint']
        hierarchy_level = tag_hierarchy.index(kind)
        tags = tag_hierarchy[hierarchy_level:]
        for tag in tags:
            snapshot_mgr.dump(tag, model, optimizer, meta)

    def _init_evaluator(sampler: TripletSampler) -> tuple[EmbeddingEvaluator, EmbeddingEvalResult]:
        evaluator = EmbeddingEvaluator.from_triplets(
            vecs=None,
            triplets=sampler.sample_uniform(n=n_eval_triplets, rand_state=1),
            loss_margin=loss_margin,
            n=sampler.n_total_items)
        print("   ", evaluator.summary_string())
        null_result = evaluator.evaluate(inputs=inputs)
        print("   ", "Results without embedding:       ", str(null_result))
        init_result = evaluator.evaluate(embedder=model, inputs=inputs)
        print("   ", "Results with pre-train embedding:", str(init_result))
        return evaluator, init_result

    # ------

    start_epoch = 0
    warm_start = False
    model_file_exists = model_file.stat().st_size > 0
    if model_file_exists:
        match exists_handling:
            case "error":
                raise FileExistsError("Model file already exists")
            case "skip":
                print("Model file exists. Skipping.")
                return
            case "overwrite":
                pass
            case "warm_start":
                warm_start = True
            case _:
                raise ValueError("Unknown exists handling mode")

    snapshot_mgr = dlutils.SnapshotMgr(model_file, base_meta=base_meta)

    if not warm_start:
        assert model is not None, "Either provide model, or allow warm start from file"
        if optim_params is None:
            optim_params = DEFAULT_OPTIMIZER_KIND
        if isinstance(optim_params, str):
            optim_params = {'kind': optim_params}
        optimizer = dlutils.get_optimizer(model.parameters(), **optim_params)
    else:
        model, optimizer, meta = snapshot_mgr.load('checkpoint')
        start_epoch = meta['epoch'] + 1
        del meta
        warm_start = True
        print("Warm start from", str(snapshot_mgr.file))

    if not model_file.parent.is_dir():
        model_file.parent.mkdir(parents=True)

    inputs = torch.as_tensor(inputs, device=device, dtype=torch.float32)
    model.to(device=device, dtype=torch.float32)
    device_count = dlutils.device_count(model)
    if device_count > 1:
        model = torch.nn.DataParallel(model)
    model.train()

    torch_rng = torch.Generator(device=device)
    noise_scale = noise_sigma * torch.std(inputs, axis=0)

    assert set(train_sampler.included_items).isdisjoint(val_sampler.included_items)
    print("Confirmed Zero train/validation overlap")

    print(f"Device: {device} ({device_count})")
    print("Train data:")
    print("   ", train_sampler.summary_string())
    print("Train Eval:")
    train_evaluator, train_eval = _init_evaluator(train_sampler)
    print("Val Eval:")
    val_evaluator, val_eval = _init_evaluator(val_sampler)

    if dbg_run:
        print(" !!! DEBUG RUN !!! ")
        epochs = start_epoch + 10
        batches_in_epoch = 4
    if batches_in_epoch is None:
        batches_in_epoch = train_sampler.n_anchors // batch_size

    if progress_mgr_params:
        progress_mgr = dlutils.ProgressManager(**progress_mgr_params, epochs=epochs)
    else:
        progress_mgr = dlutils.ProgressManager(patience=None, epochs=epochs)

    items_for_hist = train_sampler.included_items
    if wandb is not None and len(items_for_hist) > wandb.Histogram.MAX_LENGTH:
        items_for_hist = [items_for_hist[int(round(i))]
                          for i in np.linspace(0, len(items_for_hist) - 1, wandb.Histogram.MAX_LENGTH)]

    if not warm_start:
        _add_to_wandb(train_eval=train_eval, val_eval=val_eval, epoch=-1,
                      hists={'sample_counts': np.zeros(len(items_for_hist), int)})
        snapshot_mgr.wipe()
        _save_snapshot('init', epoch_=-1)

    triplet_loss = TripletLossWithIntermediates(margin=loss_margin)

    epoch_history = pd.DataFrame(
        data=np.zeros((batches_in_epoch * batch_size, 3), float),
        columns=['losses', 'p_dists', 'n_dists']
    )
    epoch_history['is_hard'] = 0
    dists = train_sampler.get_dist_matrix()

    for epoch in range(start_epoch, epochs):
        sample_counts = np.zeros(train_sampler.n_total_items, int)
        epoch_start_t = time()
        epoch_history.loc[:, :] = 0
        train_sampler.update_dist_mtx(dists)
        for batch in progbar(batches_in_epoch, span=20, prefix=f'[{epoch:3d}]', leave='prefix'):
            torch_rng.manual_seed(batch)

            if noise_sigma:
                white_noise = torch.randn(inputs.shape, generator=torch_rng, device=device, dtype=inputs.dtype)
                noisy_inputs = inputs + noise_scale * white_noise
            else:
                noisy_inputs = inputs

            optimizer.zero_grad()
            A, P, N, is_hard = train_sampler.sample(n=batch_size, rand_state=batch).T

            embedded_A = model(noisy_inputs[A])
            embedded_P = model(noisy_inputs[P])
            embedded_N = model(noisy_inputs[N])

            loss, losses, p_dists, n_dists = triplet_loss(anchor=embedded_A, positive=embedded_P, negative=embedded_N)
            loss.backward()
            optimizer.step()

            model_file.touch()

            dists[A, P] = p_dists
            dists[A, N] = n_dists

            indexes = range(batch * batch_size, (batch + 1) * batch_size)
            epoch_history.loc[indexes, :] = np.c_[losses, p_dists, n_dists, is_hard]
            sample_counts[A] += 1
            sample_counts[P] += 1
            sample_counts[N] += 1

        assert sample_counts[train_sampler.included_items].sum() == sample_counts.sum()

        train_eval = train_evaluator.evaluate(embedder=model, inputs=inputs)
        val_eval = val_evaluator.evaluate(embedder=model, inputs=inputs)

        print(f' ({time() - epoch_start_t:2.1f}s) Train: {train_eval} Val: {val_eval}', end='')
        _add_to_wandb(train_eval=train_eval, val_eval=val_eval, epoch=epoch,
                      hists={'sample_counts': sample_counts[items_for_hist]})

        progress_mgr.process(val_eval.loss, train_eval.loss, val_eval.auc, epoch=epoch)
        print(' ' + progress_mgr.report(), end='')

        if progress_mgr.is_new_best:
            _save_snapshot('best', epoch_=epoch)
            print(' [Saved]', end='')
        elif epoch % checkpoint_every == 0 or progress_mgr.should_stop:
            _save_snapshot('checkpoint', epoch_=epoch)
            print(' [Chckpt]', end='')

        indexes = epoch_history['is_hard'] == 0
        avg_diff0 = epoch_history.loc[indexes, 'n_dists'].mean() - epoch_history.loc[indexes, 'p_dists'].mean()
        avg_loss0 = epoch_history.loc[indexes, 'losses'].mean()
        indexes = epoch_history['is_hard'] == 1
        avg_diff1 = epoch_history.loc[indexes, 'n_dists'].mean() - epoch_history.loc[indexes, 'p_dists'].mean()
        avg_loss1 = epoch_history.loc[indexes, 'losses'].mean()

        print(f' Hard/Not: AvgDiff={avg_diff0:2.2f}/{avg_diff1:2.2f}, AvgLoss={avg_loss0:2.2f}/{avg_loss1:2.2f}', end='')

        print('')

        if train_eval.auc > MAX_INIT_AUC:
            progress_mgr.set_stop('high_init_auc')

        if progress_mgr.should_stop:
            print("Stopping due to " + progress_mgr.stop_reason)
            break

    wandb_run.finish()




