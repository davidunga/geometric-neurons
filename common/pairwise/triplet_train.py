import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path
import shutil
from common.utils import dlutils
from common.pairwise.sameness import SamenessData
from common.pairwise.embedding_eval import EmbeddingEvalResult, EmbeddingEvaluator
from common.utils.devtools import progbar
from common.utils.typings import *
from common.utils.devtools import verbolize
from common.utils import tbtools


def triplet_train(
        train_sameness: SamenessData,
        val_sameness: SamenessData,
        model: torch.nn.Module = None,
        batch_size: int = 64,
        epochs: int = 100,
        batches_in_epoch: int = None,
        optim_params: dict | str = None,
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_file: Path = None,
        tensorboard_dir: Path = None,
        progress_mgr_params: dict = None,
        checkpoint_every: int = 5,
        exists_handling: Literal["warm_start", "overwrite", "skip", "error"] = "error",
        base_meta: dict = None,
        dbg_run: bool = False
):

    #dbg_run = True
    assert exists_handling in ("warm_start", "overwrite", "skip", "error")

    device = dlutils.get_torch_device(device)

    # ------

    DEFAULT_OPTIMIZER_KIND = 'Adam'

    # ------

    def _add_to_tensborboard(eval_results: dict[str, EmbeddingEvalResult], epoch: int):

        def _add_triplet_sampled_hist(name, sameness: SamenessData):
            for col in sameness.triplet_sampled_counts.columns:
                tag = f'{name} sampling counts - {col}'
                counts_ = sameness.triplet_sampled_counts.loc[sameness.triplet_samplable_items, col].to_numpy()
                tbtools.add_counts_as_histogram(tb, counts_, tag, epoch)

        if tb is None:
            return

        for metric in ('loss', 'auc', 'tscore'):
            tb.add_scalars(metric.capitalize(), {name: getattr(eval_res, metric)
                                                 for name, eval_res in eval_results.items()}, epoch)

    def _save_snapshot(kind: str, epoch_: int):
        """
            kind:   'init' = save under 'init', 'best', & 'checkpoint'
                    'best' = save under 'best' & 'checkpoint'
                    'checkpoint' = save only under 'checkpoint'
        """
        meta = {'epoch': epoch_, 'train_status': progress_mgr.status_dict, 'val': val_eval.metrics_dict}
        tag_hierarchy = ['init', 'best', 'checkpoint']
        hierarchy_level = tag_hierarchy.index(kind)
        tags = tag_hierarchy[hierarchy_level:]
        for tag in tags:
            snapshot_mgr.dump(tag, model, optimizer, meta)

    def _init_sameness_and_evaluator(sameness: SamenessData) -> [EmbeddingEvaluator, EmbeddingEvalResult]:
        sameness.to(device=device, dtype=torch.float32)
        sameness.init_triplet_sampling()
        print("   ", sameness.triplet_summary_string())
        evaluator = sameness.make_evaluator(loss_margin=loss_margin)
        null_result = evaluator.evaluate()
        print("   ", "Results without embedding:       ", str(null_result))
        init_result = evaluator.evaluate(embedder=model)
        print("   ", "Results with pre-train embedding:", str(init_result))
        sameness.reset_triplet_sampled_counts()
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

    if tensorboard_dir is not None:
        if tensorboard_dir.is_dir() and not warm_start:
            shutil.rmtree(tensorboard_dir.as_posix())
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    model.to(device=device, dtype=torch.float32)
    model.train()
    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)

    print("Device:", device)
    print("Train triplets:")
    train_evaluator, train_eval = _init_sameness_and_evaluator(train_sameness)

    print("Val triplets:")
    val_evaluator, val_eval = _init_sameness_and_evaluator(val_sameness)

    if dbg_run:
        print(" !!! DEBUG RUN !!! ")
        epochs = start_epoch + 10
        batches_in_epoch = 4

    if progress_mgr_params:
        progress_mgr = dlutils.ProgressManager(**progress_mgr_params, epochs=epochs)
    else:
        progress_mgr = dlutils.ProgressManager(patience=None, epochs=epochs)

    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=str(tensorboard_dir))

    batcher = dlutils.BatchManager(batch_size=batch_size, items=list(train_sameness.triplet_anchors),
                                   batches_in_epoch=batches_in_epoch)

    if not warm_start:
        _add_to_tensborboard(dict(train=train_eval, val=val_eval), epoch=-1)
        snapshot_mgr.wipe()
        _save_snapshot('init', epoch_=-1)

    for epoch in range(start_epoch, epochs):
        epoch_start_t = time()
        batcher.init_epoch(epoch)
        for batch in progbar(batcher.batches_in_epoch, span=20, prefix=f'[{epoch:3d}]', leave='prefix'):
            optimizer.zero_grad()
            anchors = batcher.get_items(batch)
            A, P, N = train_sameness.sample_triplets(anchors, rand_seed=batch)
            loss = triplet_loss(anchor=model(A), positive=model(P), negative=model(N))
            loss.backward()
            optimizer.step()
            model_file.touch()

        train_eval = train_evaluator.evaluate(embedder=model)
        val_eval = val_evaluator.evaluate(embedder=model)

        print(f' ({time() - epoch_start_t:2.1f}s) Train: {train_eval} Val: {val_eval}', end='')
        _add_to_tensborboard(dict(train=train_eval, val=val_eval), epoch=epoch)

        progress_mgr.process(val_eval.loss, train_eval.loss, val_eval.auc, epoch=epoch)
        print(' ' + progress_mgr.report(), end='')

        if progress_mgr.is_new_best:
            _save_snapshot('best', epoch_=epoch)
            print(' [Saved]', end='')
        elif epoch % checkpoint_every == 0 or progress_mgr.should_stop:
            _save_snapshot('checkpoint', epoch_=epoch)
            print(' [Chckpt]', end='')

        print('')

        if progress_mgr.should_stop:
            print("Stopping due to " + progress_mgr.stop_reason)
            break




