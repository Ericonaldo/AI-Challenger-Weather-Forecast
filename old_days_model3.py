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

prd_time = '2018-10-15 03'
from data_utils import *

file_names = ['ai_challenger_wf2018_trainingset_20150301-20180531.nc','ai_challenger_wf2018_validation_20180601-20180828_20180905.nc',
    'ai_challenger_wf2018_testa1_20180829-20180924.nc','ai_challenger_weather_testingsetB_20180829-20181015.nc', 'ai_challenger_wf2018_testb5_20180829-20181101.nc']

#file_names = ['ai_challenger_wf2018_trainingset_20150301-20180531.nc','ai_challenger_wf2018_validation_20180601-20180828_20180905.nc',
#    'ai_challenger_wf2018_testa1_20180829-20180924.nc','ai_challenger_weather_testingsetB_20180829-20181015.nc']

model_t2m_file = model_path+"t2m.old_days_model_catlog2"
model_rh2m_file = model_path+"rh2m.old_days_model_catlog2"
model_w10m_file = model_path+"w10m.old_days_model_catlog2"


def get_train_val(tr_X, tr_y):
    val_num = int(len(tr_X) / 10)
    return tr_X.iloc[:-val_num].values, tr_y[:-val_num].values, tr_X.iloc[-val_num:].values, tr_y.iloc[-val_num:].values

def train_bst(X_tr, y_tr, X_val, y_val, feature_names='auto'):
    params = {
        'boosting': 'gbdt',
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
        # 'device': 'gpu', 
    }
    
    MAX_ROUNDS = 2000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
        feature_name=list(feature_names),
        categorical_feature = ['stations','foretimes','week','month','quarter', 'd10m_M', 'RAIN_M', 'is_today'] # 加上后两个会变少，有点奇怪
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
    key_list = ['station_date_time','stations', 'dates','foretimes']
    obs_list = ['psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs',
        'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
    M_list = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M',
        'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M',
        'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', 'wspd925_M',
        'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
        'Q700_M', 'Q500_M']
    tar_list = ['t2m_obs', 'rh2m_obs', 'w10m_obs']

    '''
    # 滑动窗特征
    list_df_fea = []
    # 滑动窗平均
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[tar_list].shift(1).rolling(tw, min_periods=int(tw/2)).mean())
        df_obs_tw.columns = [col+'_mean_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗最大值    
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[tar_list].shift(1).rolling(tw, min_periods=int(tw/2)).max())
        df_obs_tw.columns = [col+'_max_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗最小值    
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[obs_list].shift(1).rolling(tw, min_periods=int(tw/2)).min())
        df_obs_tw.columns = [col+'_min_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗中位数   
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[obs_list].shift(1).rolling(tw, min_periods=int(tw/2)).median())
        df_obs_tw.columns = [col+'_min_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗方差
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[obs_list].shift(1).rolling(tw, min_periods=int(tw/2)).var())
        df_obs_tw.columns = [col+'_min_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列
    # 滑动窗标准差
    for tw in [3, 6, 12, 24]: #滑动窗
        df_obs_tw = df_obs.groupby('stations').apply(lambda g: g[obs_list].shift(1).rolling(tw, min_periods=int(tw/2)).std())
        df_obs_tw.columns = [col+'_min_{}'.format(tw) for col in df_obs_tw.columns]
        list_df_fea.append(df_obs_tw) #添加该列

    df_all = pd.concat(list_df_fea, axis=1) # 拼接
    df_all = pd.concat([df_obs[['stations', 'dates']], df_all], axis=1)
    df_all = pd.merge(df[key_list+M_list+ obs_list], 
        df_all, left_on=['stations', 'dates'], right_on=['stations', 'dates']).drop_duplicates(subset=['stations', 'dates'], keep='last')
    '''

    df['t2m_obj'] = df.t2m_obs - df.t2m_M
    df['rh2m_obj'] = df.rh2m_obs - df.rh2m_M
    df['w10m_obj'] = df.w10m_obs - df.w10m_M  
    

    # 和之前相同时间段的均值特征
    # feature 1 hour ago
    df_all = df.copy()
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('1 hour')
    df_all = pd.merge(
        df_all, 
        get_fea(df_fea, '1h'), 
        left_on=['stations', 'dates', 'foretimes'], 
        right_on=['stations', 'dates', 'foretimes']
    )

    # feature 12 hour ago
    df_all = df.copy()
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('12 hours')
    df_all = pd.merge(
        df_all, 
        get_fea(df_fea, '12h'), 
        left_on=['stations', 'dates', 'foretimes'], 
        right_on=['stations', 'dates', 'foretimes']
    )

    # feature 1 days ago
    df_all = df.copy()
    df_fea = df.copy().drop('station_date_time', axis=1)
    df_fea['dates'] = df_fea['dates'] + pd.Timedelta('1 day')
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

    # 傅里叶变换特征
    fft_df = df_processed.copy()[key_list+M_list]
    fft_df[M_list] = np.fft.fft(fft_df[M_list])
    fft_df[M_list] = fft_df[M_list].apply(lambda x:x.real)
    fft_df.rename(columns=dict(zip(fft_df.columns[4:], [f'{col}_fft' for col in fft_df.columns[4:]])), inplace=True)
    df_processed = pd.merge(
        df_processed, 
        fft_df, 
        left_on=key_list, 
        right_on=key_list
    )
    
    for i in range(90001,90011):
        df_processed.replace(i, i-90001)
    df_processed['month'] = df['dates'].apply(lambda x:x.month-1) # 月份信息
    df_processed['week'] = df['dates'].apply(lambda x:x.weekday()) # 周几信息
    df_processed['quarter']  = df['dates'].apply(lambda x:x.quarter-1) # 季度信息
    df_processed['d10m_obs']  = (df['d10m_obs']/45).fillna(-1).astype(int)
    df_processed['d10m_M']  = (df['d10m_M']/45).fillna(-1).astype(int) # 风向改为8个类别
    df_processed.RAIN_M = (df_processed.RAIN_M>0).astype(int)
    df_processed.RAIN_obs = (df_processed.RAIN_obs>0).astype(int)# 下雨改为01
    df_processed['is_today'] = df_processed.dates.apply(lambda x:str(x.day)) == df_processed.station_date_time.str.split('_').apply(lambda x: x[1][6:8]) # 是否是今天标志位
    # df_processed['is_today'] = df_processed['is_today'].astype(int) # 加了效果就变差0.1左右
    return df_processed  
    
if __name__ == "__main__":
    # 转换数据格式
    if(not os.path.exists("..\\data\\all_data.csv")):
        combine_data(file_names)

    df = load_data("..\\data\\all_data.csv")
    df = check(df)

    # train model
    df_processed = exract_feature(df)
    
    test_idx = df_processed[lambda x: x.dates >= prd_time].index
    # df_test = df_processed.loc[test_idx]
    df_train = df_processed.loc[df_processed.index.difference(test_idx)]
    df_train.dropna(subset=['t2m_obj', 'rh2m_obj', 'w10m_obj'], inplace=True)
    df_train_y = df_train[['t2m_obj', 'rh2m_obj', 'w10m_obj']]
    
    df_train.drop(pd.Index(['psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs',
        'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs', 't2m_obj', 'rh2m_obj', 'w10m_obj']), axis=1, inplace=True)
    df_train_X = pd.concat([df_train[df_train.columns[1]], df_train[df_train.columns[3:]]], axis=1)
    # df_test_X = pd.concat([df_test[df_test.columns[1:]], df_test[df_train.columns[3:]]], axis=1)

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