import torch
import numpy as np
from common.type_utils import *
from torch.utils.tensorboard import SummaryWriter
from time import time
from common import dlutils
from tqdm import tqdm
from triplet import TripletBatcher, SamenessData, SamenessEval
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

    train_sameness.to(device)
    val_sameness.to(device)
    model.to(device)
    model.train()

    train_batcher = TripletBatcher(sameness_data=train_sameness, batch_size=batch_size)
    optimizer = dlutils.get_optimizer(model.parameters(), opt_params)
    early_stopping = dlutils.EarlyStopping(patience=None, max_overfit=0.2, min_improve=0.001)
    triplet_loss = torch.nn.TripletMarginLoss(margin=loss_margin)
    best_val_score = 0
    tb = None if tensorboard_dir is None else SummaryWriter(log_dir=tensorboard_dir)

    for epoch in range(epochs):

        train_batcher.init_epoch(epoch)
        epoch_start_t = time()

        for batch in tqdm(range(train_batcher.batches_in_epoch), desc=f'[{epoch}]', leave=False):
            optimizer.zero_grad()
            A, P, N = train_batcher.next_batch()
            a = model(A)
            p = model(P)
            n = model(N)
            loss = triplet_loss(anchor=a, positive=p, negative=n)
            loss.backward()
            optimizer.step()

        # ----
        # Post-epoch eval + report

        train_eval = SamenessEval(embedder=model, sameness=train_sameness, loss_margin=loss_margin)
        val_eval = SamenessEval(embedder=model, sameness=val_sameness, loss_margin=loss_margin)

        print('[{:2d}] ({:2.1f}s) '.format(epoch, time() - epoch_start_t), end='')
        print(str(train_eval) + ' ' + str(val_eval), end='')

        if tb is not None:
            timestamp = int(time())
            for eval_result in (train_eval, val_eval):
                tb.add_scalars('Train/Loss', {'loss': eval_result.loss}, timestamp)
                tb.add_scalars('Train/TScore', {'tscore': eval_result.tscore}, timestamp)
                tb.add_scalars('Train/AUC', {'auc': eval_result.auc}, timestamp)

        early_stopping.process(val_eval.loss, train_eval.loss)
        print(f" [{early_stopping.report_scores()}]", end='')

        if model_out_file is not None and best_val_score < val_eval.auc:
            dlutils.save_checkpoint(model_out_file, model, optimizer, meta={'epoch': epoch, **val_eval.results_dict()})
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
model = LinearEmbedder(input_size=sameness_data.X.shape[1], output_size=2)
triplet_train(train_sameness=sameness_data, val_sameness=sameness_data, model=model)

