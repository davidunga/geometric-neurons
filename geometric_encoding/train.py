import torch
import numpy as np
from common.type_utils import *
from torch.utils.tensorboard import SummaryWriter
from time import time
from common import dlutils
from tqdm import tqdm
from triplet import SamenessData, SamenessEval
from embedding import Embedder


def triplet_train(
        train_sameness: SamenessData,
        val_sameness: SamenessData,
        model: Embedder,
        batch_size: int = 64,
        epochs: int = 100,
        opt_params: str | dict = 'Adam',
        loss_margin: float = 1.,
        device: str = 'cpu',
        model_out_file: str = None,
        tensorboard_dir: str = None
    ):

    train_sameness.to(device=device, dtype=torch.float32)
    train_sameness.init_triplet_sampling()
    train_eval = SamenessEval(sameness=train_sameness)

    val_sameness.to(device=device, dtype=torch.float32)
    val_sameness.init_triplet_sampling()
    val_eval = SamenessEval(sameness=val_sameness)

    model.to(device, dtype=torch.float32)
    model.train()

    optimizer = dlutils.get_optimizer(model.parameters(), opt_params)
    early_stopping = dlutils.EarlyStopping(patience=None, overfit=0.2, converge=0.001)
    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)
    best_val_score = 0
    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=tensorboard_dir)
    batcher = dlutils.BatchManager(batch_size=batch_size, items=list(train_sameness.triplet_participating_items))

    for epoch in range(epochs):
        epoch_start_t = time()
        batcher.init_epoch(epoch)
        for batch in tqdm(range(batcher.batches_in_epoch), desc=f'[{epoch}]', leave=False):
            optimizer.zero_grad()
            anchors = batcher.get_items(batch)
            A, P, N = sameness_data.sample_triplets(anchors, rand_seed=batch)
            a = model(A)
            p = model(P)
            n = model(N)
            loss = triplet_loss(anchor=a, positive=p, negative=n)
            loss.backward()
            optimizer.step()

        # ----
        # Post-epoch eval + report

        train_eval.evaluate(embedder=model)
        val_eval.evaluate(embedder=model)

        print('[{:3d}] ({:2.1f}s) '.format(epoch, time() - epoch_start_t), end='')
        print(f'Train: {train_eval} Val: {val_eval}', end='')

        if tb is not None:
            timestamp = int(time())
            for eval_result in (train_eval, val_eval):
                tb.add_scalars('Train/Loss', {'loss': eval_result.loss}, timestamp)
                tb.add_scalars('Train/TScore', {'tscore': eval_result.tscore}, timestamp)
                tb.add_scalars('Train/AUC', {'auc': eval_result.auc}, timestamp)

        early_stopping.process(val_eval.loss, train_eval.loss)
        print(f" [{early_stopping.report()}]", end='')

        if model_out_file is not None and best_val_score < val_eval.auc:
            dlutils.checkpoint.dump(model_out_file, model, optimizer, meta={'epoch': epoch, **val_eval.results_dict()})
            print(' -- Saved best', end='')
            best_val_score = val_eval.auc

        print('')

        if early_stopping.should_stop():
            print("Early stopping due to " + early_stopping.stop_reason())
            break

    return model


from analysis.data_manager import DataMgr
from embedding import LinearEmbedder

data_mgr = DataMgr.from_default_config()
sameness_data, pairs, segmets = data_mgr.load_sameness()
model = LinearEmbedder(input_size=sameness_data.X.shape[1], output_size=2, dropout=.5)
triplet_train(train_sameness=sameness_data, val_sameness=sameness_data, model=model)

