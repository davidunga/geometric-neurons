import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path
import shutil
from common.utils import dlutils
from common.pairwise.sameness import SamenessData, SamenessEval
from common.utils.devtools import progbar
from common.utils.typings import *


def triplet_train(
        train_sameness: SamenessData,
        val_sameness: SamenessData,
        model: torch.nn.Module = None,
        batch_size: int = 64,
        epochs: int = 100,
        epoch_size_factor: float = 1.,
        optim_params: dict | str = None,
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_file: Path = None,
        tensorboard_dir: Path = None,
        progress_mgr_params: dict = None,
        checkpoint_every: int = 5,
        warm_start_mode: Literal["never", "allow", "always"] = "always"
):

    _DBG_RUN = False

    assert warm_start_mode in ("never", "allow", "always")
    # allow = warm start if checkpoint exists and model is not provided
    # always = warm start if checkpoint exists, ignore provided model if provided
    # never = no warm start, model must be provided

    # ------

    DEFAULT_OPTIMIZER_KIND = 'Adam'
    TEMP_FILE_EXT = ".training"
    CHECKPOINT_FILE_EXT = ".checkpt"

    # ------

    def _add_to_tensborboard(eval_results: dict[str, SamenessEval]):
        if tb is None: return
        for metric in ('loss', 'auc', 'tscore'):
            tb.add_scalars(metric.capitalize(), {name: getattr(eval_res, metric)
                                                 for name, eval_res in eval_results.items()}, epoch)

    def _save_model(model_file: Path):
        dlutils.checkpoint.dump(model_file, model, optimizer, meta={
            'epoch': epoch, 'train_status': progress_mgr.status_dict, 'val': val_eval.results_dict()})

    # ------

    model_file = Path(model_file)
    model_temp_file = Path(model_file.as_posix() + TEMP_FILE_EXT)
    model_ckeckpt_file = Path(model_file.as_posix() + CHECKPOINT_FILE_EXT)

    start_epoch = 0
    warm_start = False
    if (model is None or warm_start_mode == "always") and (model_ckeckpt_file.is_file() or model_temp_file.is_file()):
        assert warm_start_mode != "never", "Either provide model, or allow warm start from file"
        ckeckpt_epoch = bestmodel_epoch = 0
        if model_ckeckpt_file.is_file():
            ckeckpt_epoch = dlutils.checkpoint.get_meta(model_ckeckpt_file)['epoch']
        if model_temp_file.is_file():
            bestmodel_epoch = dlutils.checkpoint.get_meta(model_temp_file)['epoch']
        if ckeckpt_epoch >= bestmodel_epoch:
            model, optimizer, meta = dlutils.checkpoint.load(model_ckeckpt_file)
        else:
            model, optimizer, meta = dlutils.checkpoint.load(model_temp_file)
        start_epoch = meta['epoch'] + 1
        del meta
        warm_start = True
        print("Warm start from", model_file.as_posix())
    else:
        assert model is not None, "Either provide model, or allow warm start from file"
        if optim_params is None:
            optim_params = DEFAULT_OPTIMIZER_KIND
        if isinstance(optim_params, str):
            optim_params = {'kind': optim_params}
        optimizer = dlutils.get_optimizer(model.parameters(), **optim_params)

    if not model_file.parent.is_dir():
        model_file.parent.mkdir(parents=True)

    if tensorboard_dir is not None:
        if tensorboard_dir.is_dir() and not warm_start:
            shutil.rmtree(tensorboard_dir.as_posix())
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    model.to(device=device, dtype=torch.float32)
    model.train()

    train_sameness.to(device=device, dtype=torch.float32)
    train_sameness.init_triplet_sampling()
    train_eval = SamenessEval(sameness=train_sameness)

    val_sameness.to(device=device, dtype=torch.float32)
    val_sameness.init_triplet_sampling()
    val_eval = SamenessEval(sameness=val_sameness)

    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)

    if progress_mgr_params:
        progress_mgr = dlutils.ProgressManager(**progress_mgr_params, epochs=epochs)
    else:
        progress_mgr = dlutils.ProgressManager(patience=None, epochs=epochs)

    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=str(tensorboard_dir))

    if _DBG_RUN:
        print(" !!! DEBUG RUN !!! ")
        epochs = start_epoch + 3
        batches_in_epoch = 5
    else:
        n_pairs_ballpark = train_sameness.triplet_participating_n * (train_sameness.triplet_participating_n - 1) // 2
        batches_in_epoch = int(epoch_size_factor * n_pairs_ballpark / batch_size)

    batcher = dlutils.BatchManager(batch_size=batch_size, items=list(train_sameness.triplet_participating_items),
                                   batches_in_epoch=batches_in_epoch)

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

        train_eval.evaluate(embedder=model)
        val_eval.evaluate(embedder=model)

        print(f' ({time() - epoch_start_t:2.1f}s) Train: {train_eval} Val: {val_eval}', end='')
        _add_to_tensborboard(dict(train=train_eval, val=val_eval))

        progress_mgr.process(val_eval.loss, train_eval.loss, val_eval.auc, epoch=epoch)
        print(' ' + progress_mgr.report(), end='')

        if epoch % checkpoint_every == 0 or progress_mgr.should_stop:
            _save_model(model_ckeckpt_file)
            print(' [Chckpt]', end='')

        if progress_mgr.is_new_best:
            _save_model(model_temp_file)
            print(' [Saved]', end='')

        print('')

        if progress_mgr.should_stop:
            print("Stopping due to " + progress_mgr.stop_reason)
            break

    model_temp_file.rename(model_file)
    dlutils.checkpoint.update_meta(model_file, train_status=progress_mgr.status_dict)



