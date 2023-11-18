import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.utils.typings import *


def get_best_model(result_csv: PathLike, by: str = 'mean.auc'):
    agg, metric = by.split('.')
    agg_fcn = getattr('np', agg)
    results_df = pd.read_csv(str(result_csv))
    names = results_df['name'].unique().tolist()
    scores = []
    for name in names:
        scores.append(agg_fcn(results_df[results_df['name'] == name][metric].to_numpy()))
    best_name = names[int(np.argmax(scores))]




if __name__ == "__main__":
    get_best_model('/Users/davidu/geometric-neurons/outputs/cv/results 10-13--14-58-31.csv')


