# -*- coding:utf-8 -*-
# basic
from datetime import datetime, timedelta
import time
import re
import sys
import pickle
import os

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import netCDF4 as nc
from netCDF4 import Dataset

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

data_path_1 = "..\\data\\"
data_path_2 = "..\\data\\tmp\\"
output_path = "..\\output\\"
model_path = "..\\model\\"

from test_models import prd_time
from data_utils import *

file_names = ['ai_challenger_wf2018_trainingset_20150301-20180531.nc','ai_challenger_wf2018_validation_20180601-20180828_20180905.nc',
    'ai_challenger_wf2018_testa1_20180829-20180924.nc','ai_challenger_weather_testingsetB_20180829-20181015.nc', 'ai_challenger_wf2018_testb5_20180829-20181101.nc']

#file_names = ['ai_challenger_wf2018_trainingset_20150301-20180531.nc','ai_challenger_wf2018_validation_20180601-20180828_20180905.nc',
#    'ai_challenger_wf2018_testa1_20180829-20180924.nc','ai_challenger_weather_testingsetB_20180829-20181015.nc']

model_t2m_file = model_path+"t2m.old_days_model"
model_rh2m_file = model_path+"rh2m.old_days_model"
model_w10m_file = model_path+"w10m.old_days_model"


def get_train_val(tr_X, tr_y):
    val_num = int(len(tr_X) / 10)
    return tr_X.iloc[:-val_num].values, tr_y[:-val_num].values, tr_X.iloc[-val_num:].values, tr_y.iloc[-val_num:].values

def train_bst(X_tr, y_tr, X_val, y_val, feature_names='auto'):
    params = {
        'num_leaves': 31,
        'objective': 'regression',
        'min_data_in_leaf': 300,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'metric': 'l1',
        'num_threads': 8, 
        'min_data': 1, 
        'min_data_in_bin': 1, 
        #'device': 'gpu', 
    }
    
    MAX_ROUNDS = 2000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
        feature_name=list(feature_names),
        #categorical_feature = [0,1]
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    return bst

def get_fea(df, postfix):
    df_fea = pd.merge(
        df, 
        df.drop('stations', axis=1).groupby(['dates', 'foretimes'], as_index=False).mean(), 
        left_on=['dates', 'foretimes'], 
        right_on=['dates', 'foretimes']
    )
    return df_fea.rename(columns=dict(zip(df_fea.columns[3:], [f'{col}_{postfix}' for col in df_fea.columns[3:]])))    

def exract_feature(df, test_flag=False):
    key_list = ['stations', 'dates']
    obs_list = ['psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs',
        'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
    M_list = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M',
        'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M',
        'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', 'wspd925_M',
        'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
        'Q700_M', 'Q500_M']
    tar_list = ['t2m_obs', 'rh2m_obs', 'w10m_obs']

    #for column in df.columns[4:]: # 将超出范围值设置为缺失值
    #    df[THRESHOLD[column][0]>df[column]]=np.nan
    #    df[THRESHOLD[column][1]<df[column]]=np.nan

    df['t2m_obj'] = df.t2m_obs - df.t2m_M
    df['rh2m_obj'] = df.rh2m_obs - df.rh2m_M
    df['w10m_obj'] = df.w10m_obs - df.w10m_M

    # feature 2 days ago
    df_all = df.copy()
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('2 days')
    df_all = pd.merge(
        df_all, 
        get_fea(df_fea, '2d'), 
        left_on=['stations', 'dates', 'foretimes'], 
        right_on=['stations', 'dates', 'foretimes']
    )
    
    # feature- 7 days ago
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('7 days')
    df_all = pd.merge(
        df_all, 
        get_fea(df_fea, '7d'), 
        left_on=['stations', 'dates', 'foretimes'], 
        right_on=['stations', 'dates', 'foretimes']
    )
    
    # feature - 1 year ago
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('365 days')
    df_processed = pd.merge(
        df_all, 
        get_fea(df_fea, '1y'), 
        left_on=['stations', 'dates', 'foretimes'], 
        right_on=['stations', 'dates', 'foretimes']
    )

    return df_processed  
    
if __name__ == "__main__":
    # 转换数据格式
    if(not os.path.exists("..\\data\\all_data.csv")):
        combine_data(file_names)

    df = load_data("..\\data\\all_data.csv")
    
    # train model
    df_processed = exract_feature(df)
    
    test_idx = df_processed[lambda x: x.station_date_time.str.split('_').apply(lambda x: x[1]) == prd_time.replace('-','').replace(' ','')].index
    # df_test = df_processed.loc[test_idx]
    df_train = df_processed.loc[df_processed.index.difference(test_idx)]
    df_train.dropna(subset=['t2m_obj', 'rh2m_obj', 'w10m_obj'], inplace=True)
    df_train_y = df_train[['t2m_obj', 'rh2m_obj', 'w10m_obj']]
    df_train.drop(df_train.columns[33:45], axis=1, inplace=True)
    #df_test.drop(df_train.columns[33:45], axis=1, inplace=True)

    # df_train_X = pd.concat([df_train[df_train.columns[1]], df_train[df_train.columns[3:]]], axis=1)
    df_train_X = df_train[df_train.columns[3:]]
    # df_test_X = pd.concat([df_test[df_test.columns[1]], df_test[df_test.columns[3:]]], axis=1)

    # t2m model 
    target_col = 't2m_obj'
    X_tr, y_tr, X_val, y_val = get_train_val(df_train_X, df_train_y)
    print("Training {} model...".format(target_col))
    bst_t2m = train_bst(X_tr, y_tr[:, 0], X_val, y_val[:, 0], feature_names=df_train_X.columns)
    
    # rh2m model
    target_col = 'rh2m_obj'
    print("Training {} model...".format(target_col))
    bst_rh2m = train_bst(X_tr, y_tr[:, 1], X_val, y_val[:, 1], feature_names=df_train_X.columns)
       
    # rh2m model
    target_col = 'w10m_obj'
    print("Training {} model...".format(target_col))
    bst_w10m = train_bst(X_tr, y_tr[:, 2], X_val, y_val[:, 2], feature_names=df_train_X.columns)
    
    pickle.dump(bst_t2m, open(model_t2m_file, 'wb'))
    pickle.dump(bst_rh2m, open(model_rh2m_file, 'wb'))
    pickle.dump(bst_w10m, open(model_w10m_file, 'wb'))