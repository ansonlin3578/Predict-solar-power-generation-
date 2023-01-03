import numpy as np
import os

import xgboost as xgb
import lightgbm as lgb
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

def rmsle_cv(X, y, model, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

class Model:
    def __init__(self) -> None:
        self.models = []

    # TODO Add build model function and corrsponding name EX: model_type == 'lite_GBM'
    # TODO --> notice model need to append in self.models
    def build(self, model_type):
        if model_type == "XGB":
            model = xgb.XGBRegressor(
                colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,
                min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
                subsample=0.5213, random_state =7, nthread = -1
            )
            self.models.append(model)

        elif model_type == "LGBM":
            model = lgb.LGBMRegressor(
                objective='regression', num_leaves=5, learning_rate=0.05,
                n_estimators=720, max_bin=55, feature_fraction_seed=9, bagging_seed=9
            )
            self.models.append(model)

        elif model_type == "histGBM":
            model = HistGradientBoostingRegressor(
               l2_regularization=1.32e-10, early_stopping=True, learning_rate=0.05,
               max_iter=10000, max_depth=3, max_bins=55, min_samples_leaf=20, max_leaf_nodes=68
            )
            self.models.append(model)

        return self

    def Kfold(self, X, y, n_folds):
        model_list = {}
        for model in self.models:
            model_list[model.__str__().split('\n')[0].split('(')[0]] = model
        for n,model in model_list.items():
            score = rmsle_cv(X, y, model, n_folds)
            print('\n{} score: {:.4f} ({:.4f})\n'.format(n, score.mean(), score.std()))
        return model_list

    def save_model_weights(self, model, savedir, savename):
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(savedir, f"{savename}.json")
        if savename[0] != 'X':
            joblib.dump(model, savepath)
        else:
            model.save_model(savepath)
        print(f"Model weights save in {savepath}")

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
