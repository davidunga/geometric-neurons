import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from pathlib import Path
from common.utils import dlutils
from tqdm import tqdm
from common.pairwise.sameness import SamenessData, SamenessEval
from common.utils.devtools import printer


def triplet_train(
        train_sameness: SamenessData,
        val_sameness: SamenessData,
        model: torch.nn.Module,
        batch_size: int = 64,
        epochs: int = 100,
        optim_params: str | dict = 'Adam',
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_dump_file: Path = None,
        tensorboard_dir: Path = None,
        progress_mgr_params: dict = None,
    ):

    if model_dump_file is not None:
        model_dump_file.parent.mkdir(parents=True, exist_ok=True)
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=False)

    model.to(device=device, dtype=torch.float32)
    train_sameness.to(device=device, dtype=torch.float32)
    val_sameness.to(device=device, dtype=torch.float32)

    model.train()

    printer.dbg("initializing triplet sampling: train")
    train_sameness.init_triplet_sampling()
    printer.dbg("initializing train eval:")
    train_eval = SamenessEval(sameness=train_sameness)

    printer.dbg("initializing triplet sampling: validation")
    val_sameness.init_triplet_sampling()
    printer.dbg("initializing val eval:")
    val_eval = SamenessEval(sameness=val_sameness)

    printer.dbg("starting training")

    optimizer = dlutils.get_optimizer(model.parameters(), optim_params)
    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)

    if progress_mgr_params is None:
        progress_mgr = dlutils.ProgressManager(patience=None)
    else:
        if progress_mgr_params['patience'] < 1:
            progress_mgr_params['patience'] = int(.5 + progress_mgr_params['patience'] * epochs)
        progress_mgr = dlutils.ProgressManager(**progress_mgr_params)

    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=str(tensorboard_dir))
    batcher = dlutils.BatchManager(batch_size=batch_size, items=list(train_sameness.triplet_participating_items))

    def _add_to_tensborboard(eval_results: dict[str, SamenessEval]):
        if tb is None:
            return
        for metric in ('loss', 'auc', 'tscore'):
            tb.add_scalars(metric.capitalize(), {name: getattr(eval_res, metric)
                                                 for name, eval_res in eval_results.items()}, epoch)

    for epoch in range(epochs):
        epoch_start_t = time()
        batcher.init_epoch(epoch)
        for batch in tqdm(range(batcher.batches_in_epoch), desc=f'[{epoch}]', leave=False):
            optimizer.zero_grad()
            anchors = batcher.get_items(batch)
            A, P, N = train_sameness.sample_triplets(anchors, rand_seed=batch)
            loss = triplet_loss(anchor=model(A), positive=model(P), negative=model(N))
            loss.backward()
            optimizer.step()

        train_eval.evaluate(embedder=model)
        val_eval.evaluate(embedder=model)

        print('[{:3d}] ({:2.1f}s) '.format(epoch, time() - epoch_start_t), end='')
        print(f'Train: {train_eval} Val: {val_eval}', end='')

        _add_to_tensborboard(dict(train=train_eval, val=val_eval))

        progress_mgr.process(val_eval.loss, train_eval.loss, val_eval.auc)
        print(' ' + progress_mgr.report(), end='')

        if progress_mgr.is_new_best and model_dump_file is not None:
            dlutils.checkpoint.dump(model_dump_file, model, optimizer,
                                    meta={'epoch': epoch, 'val': val_eval.results_dict()})

            print(' [Saved]', end='')

        print('')

        if progress_mgr.should_stop:
            print("Early stopping due to " + progress_mgr.stop_reason)
            break

    return model


