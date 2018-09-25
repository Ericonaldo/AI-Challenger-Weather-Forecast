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
from data_utils import *

data_path_1 = "..\\data\\"
data_path_2 = "..\\data\\tmp\\"
output_path = "..\\output\\"
model_path = "..\\model\\"

SUPER_START = 4
OBS_START = 4+29

def predict(df_processed, model, prd_time):
    '''
    df_processed: 处理特征列后的数据
    model: 三个模型
    prd_time: 要预测的起始时间
    '''
    t2m_model, rh2m_model, w10m_model = model
    fea_cols = pd.Index(df_processed.columns[2:31].tolist() + df_processed.columns[34:].tolist())
    FORE_data = []
    for i in range(90001, 90011):
        for j in range(37):
            FORE_data.append('{}_{:02d}'.format(i, j))
    pred_t2m = bst_t2m.predict(df_processed.loc[lambda x: x.date_value >= prd_time, fea_cols])
    pred_rh2m = bst_rh2m.predict(df_processed.loc[lambda x: x.date_value >= prd_time, fea_cols])
    pred_w10m = bst_w10m.predict(df_processed.loc[lambda x: x.date_value >= prd_time, fea_cols])
    
    df_submit = pd.DataFrame([
        np.array(FORE_data), 
        pred_t2m, 
        pred_rh2m, 
        pred_w10m, 
    ]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
  
    ans_name = 'forecast-' + "".join(re.split('-| ', prd_time))+".csv"
    df_submit.to_csv(output_path+ans_name)

# test windows model
def test_windows_model():
    from windows_model import model_t2m_file,model_rh2m_file,model_w10m_file,exract_feature
    
    if(not os.path.exists("..\\data\\test.csv")):
        transfer_data_to_csv(test_file, "..\\data\\test.csv")   
    test_df = load_data(".\\data\\test.csv")
    test_df = fill_missing_data(test_df)
    for col in ['psur_obs', 't2m_obs', 'q2m_obs',
       'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']:
        col_filled = col.split('_')[0] + '_M' if 'psur' not in col else 'psfc_M'
        test_df[col].fillna(test_df[col_filled], inplace=True) #用超算值填充        
    test_df_processed = exract_feature(test_df)
    
    # 加载模型   
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    predict(test_df_processed,[t2m_model, rh2m_model, w10m_model], PRD_TIME) # 预测并打印输出

# test model2    
def test_model2():
   from model2 import model_t2m_file,model_rh2m_file,model_w10m_file
   
    if(not os.path.exists("..\\data\\test.csv")):
        transfer_data_to_csv(test_file, "..\\data\\test.csv")   
    test_df = load_data(".\\data\\test.csv")
    test_df = fill_missing_data(test_df)
    
    #......
    
    # 加载模型
    t2m_model = pickle.load(open(model_t2m_file, 'rb'))
    rh2m_model = pickle.load(open(model_rh2m_file, 'rb'))
    w10m_model = pickle.load(open(model_w10m_file, 'rb'))
    
    predict(test_df_processed,[t2m_model, rh2m_model, w10m_model], PRD_TIME) # 预测并打印输出
    