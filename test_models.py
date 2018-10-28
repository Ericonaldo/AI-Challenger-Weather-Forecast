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
from sklearn.externals import joblib

# utils
from data_utils import *
from weather_forecasting2018_eval import *

data_path_1 = "..\\data\\"
data_path_2 = "..\\data\\tmp\\"
output_path = "..\\output\\"
model_path = "..\\model\\"

# test_file = "..\\data\\ai_challenger_weather_testingsetB_20180829-20181015.nc"
# test_file = "..\\data\\ai_challenger_wf2018_testb1_20180829-20181028.nc"

# 测试分数
obs_file = "../data/aic_wf2018_testa2_obs_2018101503.csv" # 观测结果
fore_file = "../data/aic_wf2018_testa2_fore_2018101503.csv" # 超算结果

SUPER_START = 4
OBS_START = 4+29

out_time = prd_time = '2018-10-28 03'
ans_name = 'forecast-' + "".join(re.split('-| ', out_time))+".csv"

def print_score(fore_file, obs_file, anen_file):
    result = eval_result(fore_file, obs_file, anen_file)
    print(result)

def predict(df_processed, model, prd_time):
    '''
    df_processed: 处理特征列后的数据
    model: 三个模型
    prd_time: 要预测的起始时间
    '''
    t2m_model, rh2m_model, w10m_model = model
    fea_cols = pd.Index(df_processed.columns[2:31].tolist() + df_processed.columns[34:].tolist())
    FORE_data = []
    stat_l , stat_r = df_processed.stations.drop_duplicates().min(), df_processed.stations.drop_duplicates().max()

    for i in range(stat_l, stat_r+1): # 所有站点
        for j in range(37):
            FORE_data.append('{}_{:02d}'.format(i, j))
    pred_t2m = t2m_model.predict(df_processed.loc[lambda x: x.dates >= prd_time, fea_cols])
    pred_rh2m = rh2m_model.predict(df_processed.loc[lambda x: x.dates >= prd_time, fea_cols])
    pred_w10m = w10m_model.predict(df_processed.loc[lambda x: x.dates >= prd_time, fea_cols])
    
    df_submit = pd.DataFrame([
        np.array(FORE_data), 
        pred_t2m, 
        pred_rh2m, 
        pred_w10m, 
    ]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
      
    return df_submit

# test windows model
def test_windows_model():
    from windows_model import model_t2m_file,model_rh2m_file,model_w10m_file,exract_feature
    
    if(not os.path.exists(data_path_1 + "test.csv")):
        transfer_data_to_csv(test_file, data_path_1 + "test.csv")
    test_df = load_data(data_path_1 + "test.csv")
    # test_df = fill_missing_data(test_df)
    for col in ['psur_obs', 't2m_obs', 'q2m_obs',
       'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']:
        col_filled = col.split('_')[0] + '_M' if 'psur' not in col else 'psfc_M'
        test_df[col].fillna(test_df[col_filled], inplace=True) #用超算值填充        
    test_df_processed = exract_feature(test_df,True)
    
    # 加载模型   
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    df_submit = predict(test_df_processed,[t2m_model, rh2m_model, w10m_model], prd_time) # 预测并打印输出
    df_submit.to_csv(output_path+ans_name, index=False)

    # 计算分数
    anen_file = output_path+ans_name
    print_score(fore_file, obs_file, anen_file)

def test_windows_model2():
    from windows_model2 import model_t2m_file,model_rh2m_file,model_w10m_file,exract_feature
    
    if(not os.path.exists(data_path_1 + "test.csv")):
        transfer_data_to_csv(test_file, data_path_1 + "test.csv")   
    test_df = load_data(data_path_1 + "test.csv")
    # test_df = fill_missing_data(test_df)
    for col in ['psur_obs', 't2m_obs', 'q2m_obs',
       'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']:
        col_filled = col.split('_')[0] + '_M' if 'psur' not in col else 'psfc_M'
        test_df[col].fillna(test_df[col_filled], inplace=True) #用超算值填充        
    test_df_processed = exract_feature(test_df,True)
    
    # 加载模型   
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    df_submit = predict(test_df_processed,[t2m_model, rh2m_model, w10m_model], prd_time) # 预测并打印输出
    df_submit.to_csv(output_path+ans_name, index=False)

    # 计算分数
    anen_file = output_path+ans_name
    print_score(fore_file, obs_file, anen_file)

def test_windows_station_model():
    from windows_station_model import model_t2m_file,model_rh2m_file,model_w10m_file,exract_feature
    
    if(not os.path.exists(data_path_1 + "test.csv")):
        transfer_data_to_csv(test_file, data_path_1 + "test.csv")   
    test_df = load_data(data_path_1 + "test.csv")
    # test_df = fill_missing_data(test_df)
    for col in ['psur_obs', 't2m_obs', 'q2m_obs',
       'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']:
        col_filled = col.split('_')[0] + '_M' if 'psur' not in col else 'psfc_M'
        test_df[col].fillna(test_df[col_filled], inplace=True) #用超算值填充  
    
    station_id=[90001,90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009, 90010]
    test_station_df = {} 
    test_df_processed = exract_feature(test_df,True)
    for id in station_id:
        test_station_df[str(id)] = test_df_processed[test_df_processed["stations"]==id]
    
    # 加载模型   
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    df_submit = predict(test_station_df[str(station_id[0])],[t2m_model[str(station_id[0])], rh2m_model[str(station_id[0])], w10m_model[str(station_id[0])]], prd_time)
    for id in station_id[1:]:
        df_submit = df_submit.append(predict(test_station_df[str(id)],[t2m_model[str(id)], rh2m_model[str(id)], w10m_model[str(id)]], prd_time))
    
    df_submit.to_csv(output_path+ans_name, index=False) # 预测并打印输出
    
    # 计算分数
    anen_file = output_path+ans_name
    print_score(fore_file, obs_file, anen_file)


# test model2    
def test_old_days_model():
    from old_days_model import model_t2m_file,model_rh2m_file,model_w10m_file,exract_feature
    
    if(not os.path.exists(data_path_1 + "test.csv")):
        transfer_data_to_csv(test_file, data_path_1 + "test.csv")

    train_df = load_data("..\\data\\train.csv")
    valid_df = load_data("..\\data\\valid.csv")
    test_df = load_data("..\\data\\test.csv")

    # 填充缺失值
    # train_df = fill_missing_data(train_df)
    # valid_df = fill_missing_data(valid_df)
    df = pd.concat([
    train_df, 
    valid_df, 
    test_df, 
    ]).reset_index(drop=True)

    df_processed = exract_feature(df)
    test_idx = df_processed[lambda x: x.station_date_time.str.split('_').apply(lambda x: x[1]) == prd_time.replace('-','').replace(' ','')].index
    df_test = df_processed.loc[test_idx]
    df_test.drop(df_test.columns[33:45], axis=1, inplace=True)
    df_test_X = df_test[df_test.columns[3:]]
    
    # 加载模型
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    df_submit = pd.DataFrame([
        [f'{item[0]}_{int(item[2]):02d}' for item in df_test.station_date_time.str.split('_')], 
        (df_test.t2m_M + t2m_model.predict(df_test_X)).tolist(), 
        (df_test.rh2m_M + rh2m_model.predict(df_test_X)).tolist(), 
        (df_test.w10m_M + w10m_model.predict(df_test_X)).tolist(), 
    ]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
    
    df_submit.to_csv(output_path+ans_name, index=False) # 预测并打印输出

    # 计算分数
    anen_file = output_path+ans_name
    # print_score(fore_file, obs_file, anen_file)
    