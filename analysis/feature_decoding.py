import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn
from motorneural.data import DataSlice
from data_manager import DataMgr
from common.utils.typings import *
import cv_results_mgr
from common.utils.inlier_detection import InlierDetector
from analysis.config import DataConfig, Config
from common.utils import dlutils
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from common.utils.scoring import BootstrapEvaluator


def calc_kinematic_features(data_slices: list[DataSlice], reduce: str = 'median'):
    var_names = ['AfSpd', 'AfCrv', 'EuSpd', 'EuAcc', 'SaSpd']
    reduce_func = getattr(np, reduce)
    ret = {var_name: np.fromiter((reduce_func(s.kin[var_name]) for s in data_slices), float)
           for var_name in var_names}
    return ret


def decode_features_from_embedded(model_file):

    evaluator = BootstrapEvaluator(n_shuffs=1000, seed=1)

    model, cfg = cv_results_mgr.get_model_and_config(model_file)
    cfg = cfg.get_as_eval_config()
    data_mgr = DataMgr(cfg.data)
    segments = data_mgr.load_segments()
    features = calc_kinematic_features(segments)
    vecs, _ = data_mgr.get_inputs()
    embedded_vecs = dlutils.safe_predict(model, vecs)

    metrics = {}
    for feature_name, ytrue in features.items():
        print(feature_name, "...")
        regressor = LinearRegression()
        regressor.fit(embedded_vecs, ytrue)
        yhat = regressor.predict(embedded_vecs)
        metrics[feature_name] = evaluator.evaluate(ytrue, yhat)
        print("  ", {k: metrics[feature_name][k]['zscore'] for k in metrics[feature_name]})



if __name__ == "__main__":
    file = "/Users/davidu/geometric-neurons/outputs/models/TP_RS bin10 lag100 dur200 affine-kinX-nmahal f70c5c.Fold0.pth"
    #draw_trajectories_grouped_by_embedded_dist(file)
    decode_features_from_embedded(file)
