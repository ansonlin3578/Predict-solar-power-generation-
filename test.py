import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from dataset.data_utils import load_data
from configs import DataConfigs, TestConfigs
from preprocessing import fill_null
from Analysis.Exploratory_Data_Analysis import Corr, module_performance_analysis
from models import Model

import csv
import os

dataset = DataConfigs.dataset
load_data.load_test(r"./dataset/test.csv", dataset)

df_original = dataset["test"]["data"]
df_with_module = df_original[["Generation", "Module", "Irradiance", "Capacity", "Irradiance_m", "Temp"]]
##calculate the correlation matrix and filts the columns which have value lower than 0.1 with "Generation"
df_wo_module = df_original.drop(columns=["Module"])

corr_matrix = df_wo_module.corr()
to_filter = corr_matrix["Generation"]
for index in to_filter.index:
    if abs(to_filter[index]) < 0.1:
        df_wo_module = df_wo_module.drop(columns=[index])
dataset["test"]["data"] = df_wo_module
df_with_module = fill_null(df_original=df_original, df_with_module=df_with_module)

################################# Preprocessing ####################################
only_module = df_with_module[["Module"]]
only_module = pd.concat((only_module, pd.get_dummies(only_module.Module)), 1)
only_module = only_module.drop(["Module"], axis=1)
only_module["MM60-6RT-300"] = 1.5 * only_module["MM60-6RT-300"] #onehot encodeing後，將module("MM60-6RT-300")的scale * 1.5
onehot_df = pd.concat(objs=(df_with_module, only_module), axis=1)
test_data = onehot_df.drop(columns=["Generation", "Module"])

######################################## Create model  #################################################
model_builder = Model()
for model in TestConfigs.model_type:
    model_builder.build(model)
model_list = model_builder.models

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, random_state=7, nthread=-1)



for (model_art, model_weight) in zip(model_builder.models, TestConfigs.model_weights):
    model_art.load_model(model_weight)
    model_xgb.load_model(model_weight)
    print(f"Load {TestConfigs.model_weights} weights !")
    xgb_pred = model_art.predict(test_data)
    
    os.makedirs(TestConfigs.prediction_dir, exist_ok=True)
    with open(os.path.join(TestConfigs.prediction_dir, 'submission.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(["ID", "Generation"])
        for i, g in enumerate(xgb_pred):
            w.writerow([i + 1, int(g)])
    print(f"Save prediction result in {os.path.join(TestConfigs.prediction_dir, 'submission.csv')}")
