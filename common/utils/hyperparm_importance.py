import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from common.utils import plotting


def compute_effects(df: pd.DataFrame, score: Sequence):
    cols = list(df.columns)
    s = {}
    for c in cols:
        s[c] = len(set(df.loc[:, c].values))

    best_score = np.max(score)
    best_hyperparams = df.iloc[np.argmax(score)]
    effects = {col: [] for col in cols}
    for i in range(len(df)):
        isdiff = best_hyperparams != df.iloc[i]
        if np.sum(isdiff) == 1:
            col = cols[int(np.nonzero(isdiff)[0])]
            effects[col].append(score[i] - best_score)

    result = {
        'max': {k: np.max(v) if len(v) else None for k, v in effects.items()},
        'mean': {k: np.mean(v) if len(v) else None for k, v in effects.items()},
        'median': {k: np.median(v) if len(v) else None for k, v in effects.items()},
    }
    return result


def plot_effects(effects):
    effects = {k: v for k, v in effects.items() if v is not None}
    plt.bar(effects.keys(), effects.values())
    plt.xlabel('Hyperparameter')
    plt.ylabel('Effect (Gradient)')
    plt.xticks(rotation=45, ha="right")
    plt.title('Hyperparameter Effects')


if __name__ == "__main__":
    full_df = pd.read_csv("/Users/davidu/Downloads/wandb_export_2024-05-20T08_20_53.386+03_00.csv")
    hyperparam_cols = [col for col in full_df.columns if col.startswith('cfg.')]
    score_cols = [col for col in full_df.columns if col.startswith(('val_', 'train_'))]

    hyperparam_df = full_df.groupby('Name')[hyperparam_cols].head(1)
    score_df = full_df.groupby('Name')[score_cols].mean()

    varcols = [c for c in hyperparam_df.columns
               if len(set(hyperparam_df[c].values)) > 1 and c != 'cfg.data.trials.name']

    score = score_df['val_auc'].to_numpy()
    datasets_names = sorted(list(set(hyperparam_df['cfg.data.trials.name'].values)))
    axs = None
    for name in datasets_names:
        print("======")
        print(name)
        ii = np.nonzero(hyperparam_df['cfg.data.trials.name'] == name)[0]
        df = hyperparam_df.iloc[ii]
        sc = score[ii]

        print("best=", df.iloc[np.argmax(sc)][varcols])
        print(np.sum(full_df['Name'] == full_df.iloc[ii].iloc[np.argmax(sc)]['Name']))
        continue
        for c in varcols:
            vs = df.loc[:, c].to_numpy()
            print(c, ":")
            for v in set(vs):
                print(f" {v} -> {sc[vs == v].max():2.4f}")
        #
        # effects = compute_effects(hyperparam_df.iloc[ii], score[ii])
        # effects = effects['max']
        # effects = {k.split('.')[-1]: v for k, v in effects.items()}
        # if axs is None:
        #     axs = plotting.named_subplots(cols=datasets_names)
        # plt.sca(axs[name])
        # plot_effects(effects)
        # plt.title(f'{name} - effects')
    plt.show()

