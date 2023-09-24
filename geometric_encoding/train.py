import torch
import numpy as np
from config import get_default_config, get_device, make_analysis_name, output_dir
from model import LinearEmbedder, embdded_dist_fnc
from torch.utils.tensorboard import SummaryWriter
from time import time
from utils.dlutils import save_checkpoint, EarlyStopping
from utils import dictools, utils
from tqdm import tqdm
from triplet import make_triplet_batches, TripletLoss
from evaluation import eval_model_triplet
from data import DATASPLIT

device = get_device()
models_dir = output_dir + '/models'
tb_dir = output_dir + '/tb'


def adjust_threshold(model, cfg, train_batches=None):
    """ Compute and update model's decision threshold """

    if train_batches is None:
        _, [train_batches] = make_triplet_batches(cfg, device, (DATASPLIT.TRAIN,), ctrls=False)
    assert train_batches.split == DATASPLIT.TRAIN
    train_batches.make_epoch()
    loss_fnc = TripletLoss(margin=cfg['train']['loss_margin'])
    _, optimal_thresh = eval_model_triplet(model, train_batches, loss_fnc, batches=10, use_optimal_thresh=True)
    model.decision_thresh = optimal_thresh
    return model, train_batches


def train(cfg=None, tbwrt=False, **kwargs):
    """
    train a model using parameters specified in configurations
    :param cfg: configurations json, if None - using default
    :param tbwrt: flag- write to tensorboard?
    :param kwargs: optional arguments, to modify the configurations by
    :return:
        model_file - path to model file
    """

    if cfg is None:
        cfg = get_default_config()
    cfg = cfg.copy()

    for key in kwargs:
        paths_to_key = dictools.find_key_in_dict(cfg, key)
        assert len(paths_to_key) == 1
        dictools.update_nested_dict_(cfg, paths_to_key[0], kwargs[key])

    torch.manual_seed(cfg['train']['rand_seed'])
    np.random.seed(cfg['train']['rand_seed'])

    # ---------------------------------------
    # get batches:

    samples_meta, [train_batches, val_batches] = make_triplet_batches(
        cfg, device, (DATASPLIT.TRAIN, DATASPLIT.VAL), ctrls=False)
    assert train_batches.split == DATASPLIT.TRAIN
    assert val_batches.split == DATASPLIT.VAL
    train_batches.make_epoch()
    val_batches.make_epoch()
    num_eval_batches = len(val_batches)

    print(train_batches)
    print(val_batches)

    # ---------------------------------------
    # make model:

    model = LinearEmbedder(np.prod(train_batches.X.size()[1:]), **cfg['model'])
    model.to(device)
    model.train()

    cfg_name = make_analysis_name(cfg)
    model_file = models_dir + f'/model_{cfg_name}.pth'

    # ---------------------------------------

    early_stopping = EarlyStopping(patience=None, max_overfit=0.2, min_improve=0.001)
    loss_fnc = TripletLoss(margin=cfg['train']['loss_margin'])
    optimizer = torch.optim.Adam(model.parameters(), **cfg['train']['optimizer_params'])
    best_val_score = 0
    tb = SummaryWriter(log_dir=f'{tb_dir}/{cfg_name}_{utils.time_str()}') if tbwrt else None

    for epoch in range(cfg['train']['epochs']):

        train_batches.make_epoch()
        epoch_start_t = time()

        for batch in tqdm(range(len(train_batches)), desc=f'[{epoch}]', leave=False):
            optimizer.zero_grad()
            A, P, N, _ = train_batches[batch]
            a = model(A)
            p = model(P)
            n = model(N)
            p_dists = embdded_dist_fnc(a, p)
            n_dists = embdded_dist_fnc(a, n)
            loss = loss_fnc(p_dists, n_dists)
            loss.backward()
            optimizer.step()

        # ----
        # Post-epoch eval + report

        train_res, optimal_thresh = eval_model_triplet(model, train_batches, loss_fnc, num_eval_batches,
                                                       use_optimal_thresh=True)
        model.decision_thresh = optimal_thresh
        val_res, _ = eval_model_triplet(model, val_batches, loss_fnc, num_eval_batches)

        print('[{:2d}] ({:2.1f}s) '.format(epoch, time() - epoch_start_t), end='')
        print(str(train_res) + ' ' + str(val_res), end='')
        print(' Thresh={:2.2f}'.format(model.decision_thresh), end='')

        timestamp = int(time())
        if tbwrt:
            tb.add_scalars('Train/Loss', {'loss': train_res.loss}, timestamp)
            tb.add_scalars('Train/Acc', {'acc': train_res.acc}, timestamp)
            tb.add_scalars('Train/AUC', {'auc': train_res.auc}, timestamp)
            tb.add_scalars('Val/Loss', {'loss': val_res.loss}, timestamp)
            tb.add_scalars('Val/Acc', {'acc': val_res.acc}, timestamp)
            tb.add_scalars('Val/AUC', {'auc': val_res.auc}, timestamp)

        early_stopping.process(val_res.loss, train_res.loss)
        print(f" [{early_stopping.report_scores()}]", end='')

        if best_val_score < val_res.acc:
            save_checkpoint(model_file, model, optimizer, meta={
                'cfg': cfg, 'val_acc': val_res.acc, 'samples_meta': samples_meta, 'epoch': epoch})
            print(' -- Saved best', end='')
            best_val_score = val_res.acc

        print('')

        if early_stopping.should_stop():
            print("Early stopping due to " + early_stopping.stop_reason())
            break

    return model_file


if __name__ == "__main__":
    train(random_neural=False, epochs=50)

