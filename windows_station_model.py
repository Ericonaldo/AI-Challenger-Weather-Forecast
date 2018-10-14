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

from data_utils import *

train_file = "..\\data\\ai_challenger_wf2018_trainingset_20150301-20180531.nc"
valid_file = "..\\data\\ai_challenger_wf2018_validation_20180601-20180828_20180905.nc"
test_file = "..\\data\\ai_challenger_wf2018_testa1_20180829-20180924.nc"

model_t2m_file = model_path+"t2m.windows_station_model"
model_rh2m_file = model_path+"rh2m.windows_station_model"
model_w10m_file = model_path+"w10m.windows_station_model"

def get_x_y(df, fea_cols, target_col):
    val_num = int(len(df) / 10)
    df = df.dropna(subset=[target_col])
    X = df[fea_cols].values
    y = df[target_col].values
    return X, y

def train_bst(X_tr, y_tr, X_val, y_val, init_model=None):
    params = {
        'num_leaves': 31,
        'objective': 'regression',
        'min_data_in_leaf': 300,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'metric': 'rmse',
        'num_threads': 8, 
        'min_data': 1, 
        'min_data_in_bin': 1, 
        #'device': 'gpu', 
    }
    
    MAX_ROUNDS = 3000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
    )

    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        init_model=init_model, valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    return bst

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
    
    df['dates'] = pd.to_datetime(df.dates, format='%Y%m%d%H') + df.foretimes.apply(lambda x: pd.Timedelta(x, unit='h')) 
    
    df_obs = df[key_list+obs_list].drop_duplicates().sort_values(key_list)

    list_df_fea = []
    # 滑动窗平均
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[tar_list].shift(1).rolling(tw).mean())
        df_obs_tw.columns = [col+'_mean_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗最大值    
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[tar_list].shift(1).rolling(tw).max())
        df_obs_tw.columns = [col+'_max_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗最小值    
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[obs_list].shift(1).rolling(tw).min())
        df_obs_tw.columns = [col+'_min_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    df_fea = pd.concat(list_df_fea, axis=1).dropna() # 拼接并舍弃nan
    df_fea = pd.concat([df_obs[['stations', 'dates']], df_fea], axis=1).dropna()
    
    if test_flag:
        df_processed = pd.merge(df[key_list+M_list+ tar_list], 
            df_fea, left_on=['stations', 'dates'], right_on=['stations', 'dates']).dropna().drop_duplicates(subset=['stations', 'dates'], keep='last')
    else:
        df_processed = pd.merge(df[key_list+M_list+ tar_list], 
            df_fea, left_on=['stations', 'dates'], right_on=['stations', 'dates']).dropna()
    
    return df_processed    
    
if __name__ == "__main__":
    # 转换数据格式
    if(not os.path.exists("..\\data\\train.csv")):
        transfer_data_to_csv(train_file, "..\\data\\train.csv")
    if(not os.path.exists("..\\data\\valid.csv")):
        transfer_data_to_csv(valid_file, "..\\data\\valid.csv")
    
    train_df = load_data("..\\data\\train.csv")
    valid_df = load_data("..\\data\\valid.csv")

    station_id=[90001,90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009, 90010]
    train_station_df = {}
    valid_station_df = {}
    
    # 填充缺失值
    train_df = fill_missing_data(train_df)
    valid_df = fill_missing_data(valid_df)    
    
    # train model
    train_df_processed = exract_feature(train_df)
    valid_df_processed = exract_feature(valid_df)
    
    for id in station_id:
        train_station_df[str(id)] = train_df_processed[train_df_processed["stations"]==id]
        valid_station_df[str(id)] = valid_df_processed[valid_df_processed["stations"]==id]

    # features
    fea_cols = pd.Index(train_df_processed.columns[2:31].tolist() + train_df_processed.columns[34:].tolist())
    
    # t2m_obs
    target_col = 't2m_obs'
    bst_t2m = {}
    from windows_model2 import model_t2m_file as init_model_t2m_file
    t2m_model = pickle.load(open(init_model_t2m_file, 'rb'))
    init_model = t2m_model
    for id in station_id:
        X_tr, y_tr = get_x_y(train_station_df[str(id)], fea_cols, target_col)  
        X_val, y_val = get_x_y(valid_station_df[str(id)], fea_cols, target_col)
        print("Training {} model for station {}...".format(target_col, id))
        bst_t2m[str(id)] = train_bst(X_tr, y_tr, X_val, y_val, init_model)
    
    # rh2m_obs
    target_col = 'rh2m_obs'
    bst_rh2m = {}
    from windows_model2 import model_rh2m_file as init_model_rh2m_file
    rh2m_model = pickle.load(open(init_model_rh2m_file, 'rb'))
    init_model = rh2m_model
    for id in station_id:
        X_tr, y_tr = get_x_y(train_station_df[str(id)], fea_cols, target_col)  
        X_val, y_val = get_x_y(valid_station_df[str(id)], fea_cols, target_col)
        print("Training {} model for station {}...".format(target_col, id))
        bst_rh2m[str(id)] = train_bst(X_tr, y_tr, X_val, y_val, init_model)
       
    # rh2m_obs
    target_col = 'w10m_obs'
    bst_w10m = {}
    from windows_model2 import model_w10m_file as init_model_w10m_file
    w10m_model = pickle.load(open(init_model_w10m_file, 'rb'))
    init_model = w10m_model
    for id in station_id:
        X_tr, y_tr = get_x_y(train_station_df[str(id)], fea_cols, target_col)  
        X_val, y_val = get_x_y(valid_station_df[str(id)], fea_cols, target_col)
        print("Training {} model for station {}...".format(target_col, id))
        bst_w10m[str(id)] = train_bst(X_tr, y_tr, X_val, y_val, init_model)
    
    pickle.dump(bst_t2m, open(model_t2m_file, 'wb'))
    pickle.dump(bst_rh2m, open(model_rh2m_file, 'wb'))
    pickle.dump(bst_w10m, open(model_w10m_file, 'wb'))