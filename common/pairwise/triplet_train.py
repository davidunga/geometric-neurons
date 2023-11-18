import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path
import shutil
from common.utils import dlutils
from common.pairwise.sameness import SamenessData, SamenessEval
from common.utils.devtools import progbar


def triplet_train(
        train_sameness: SamenessData,
        val_sameness: SamenessData,
        model: torch.nn.Module,
        batch_size: int = 64,
        epochs: int = 100,
        epoch_size_factor: float = 1.,
        optim_params: dict | str = 'Adam',
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_dump_file: Path = None,
        tensorboard_dir: Path = None,
        progress_mgr_params: dict = None,
    ):

    _DBG_RUN = False

    if isinstance(optim_params, str):
        optim_params = {'kind': optim_params}

    if model_dump_file:
        model_dump_file.parent.mkdir(parents=True, exist_ok=True)
        model_temp_file = Path(model_dump_file.as_posix() + ".training")

    if tensorboard_dir is not None:
        if tensorboard_dir.is_dir():
            shutil.rmtree(tensorboard_dir.as_posix())
        tensorboard_dir.mkdir(parents=True, exist_ok=False)

    model.to(device=device, dtype=torch.float32)
    train_sameness.to(device=device, dtype=torch.float32)
    val_sameness.to(device=device, dtype=torch.float32)

    model.train()

    train_sameness.init_triplet_sampling()
    train_eval = SamenessEval(sameness=train_sameness)

    val_sameness.init_triplet_sampling()
    val_eval = SamenessEval(sameness=val_sameness)

    optimizer = dlutils.get_optimizer(model.parameters(), **optim_params)
    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)

    if progress_mgr_params:
        progress_mgr = dlutils.ProgressManager(**progress_mgr_params, epochs=epochs)
    else:
        progress_mgr = dlutils.ProgressManager(patience=None, epochs=epochs)

    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=str(tensorboard_dir))

    n_pairs_ballpark = train_sameness.triplet_participating_n * (train_sameness.triplet_participating_n - 1) // 2
    batches_in_epoch = int(epoch_size_factor * n_pairs_ballpark / batch_size)
    if _DBG_RUN:
        print(" !!! DEBUG RUN !!! ")
        epochs = 3
        batches_in_epoch = 5
    batcher = dlutils.BatchManager(batch_size=batch_size, items=list(train_sameness.triplet_participating_items),
                                   batches_in_epoch=batches_in_epoch)

    def _add_to_tensborboard(eval_results: dict[str, SamenessEval]):
        if tb is None:
            return
        for metric in ('loss', 'auc', 'tscore'):
            tb.add_scalars(metric.capitalize(), {name: getattr(eval_res, metric)
                                                 for name, eval_res in eval_results.items()}, epoch)

    for epoch in range(epochs):
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

        if progress_mgr.is_new_best and model_dump_file:
            dlutils.checkpoint.dump(model_temp_file, model, optimizer, meta={
                'epoch': epoch, 'train_status': progress_mgr.status_dict, 'val': val_eval.results_dict()})

            print(' [Saved]', end='')

        print('')

        if progress_mgr.should_stop:
            print("Stopping due to " + progress_mgr.stop_reason)
            break

    if model_dump_file:
        model_temp_file.rename(model_dump_file)
        dlutils.checkpoint.update_meta(model_dump_file, train_status=progress_mgr.status_dict)


