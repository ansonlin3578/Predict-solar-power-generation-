import numpy as np
import pandas as pd
import os

import pdb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib

from dataset.data_utils import load_data
from configs import DataConfigs, TrainConfigs
from Analysis.Exploratory_Data_Analysis import Corr, module_performance_analysis
from preprocessing import fill_null
from models import Model, rmsle

############################# Read data #################################
dataset = DataConfigs.dataset
load_data.load_train(r"./dataset/train.csv", dataset)
load_data.load_test(r"./dataset/test.csv", dataset)

# Reformat
df_original = dataset["train"]["data"]
df_with_module = df_original[["Generation", "Module", "Irradiance", "Capacity", "Irradiance_m", "Temp"]]
df_wo_module = df_original.drop(columns=["Module"])
##########################################################################



################################# EDA ####################################
corr_matrix = Corr(df_wo_module)
to_filter = corr_matrix["Generation"]
for index in to_filter.index:
    if abs(to_filter[index]) < 0.1:
        df_wo_module = df_wo_module.drop(columns=[index])
dataset["train"]["data"] = df_wo_module
# fill non-value
df_with_module = fill_null(df_original=df_original, df_with_module=df_with_module)
df_with_module = module_performance_analysis(df_with_module, './Analysis/train_data')
################################################################################

############################### Preprocessing ##################################

# Module encoding
only_module = df_with_module[["Module"]]
only_module = pd.concat((only_module, pd.get_dummies(only_module.Module)), 1)
only_module = only_module.drop(["Module"], axis=1)
only_module["MM60-6RT-300"] = 1.5 * only_module["MM60-6RT-300"] #onehot encodeing後，將module("MM60-6RT-300")的scale * 1.5

onehot_df = pd.concat(objs=(df_with_module, only_module), axis=1)
# for column in onehot_df:
#     print("nan value in {} : ".format(column) , onehot_df[column].isna().sum())
################################################################################

################################ Create model  #################################
train_label = onehot_df.Generation.values   #整理成最後要丟進去model的資料型態
train_data = onehot_df.drop(columns=["Generation", "Module", "grade"])
n_folds = 5
model_builder = Model()
for model in TrainConfigs.model_type:
    model_builder.build(model)
model_list = model_builder.Kfold(train_data, train_label ,n_folds)
################################################################################

############################### Training PipeLine ##############################
for model_name in model_list:
    print(f'{model_name} training start... ')
    model = model_list[model_name]
    model.fit(train_data, train_label)
    # save in JSON format
    model_builder.save_model_weights(model, './checkpoints', model_name)
    xgb_train_pred = model.predict(train_data)
    print(f'Training Loss: {rmsle(train_label, xgb_train_pred)}')
################################################################################

