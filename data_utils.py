# -*- coding:utf-8 -*-
# basic
from datetime import datetime, timedelta
import time
import re
import sys

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

base_path_1 = "../data/"
base_path_2 = "../data/tmp/"
base_path_3 = "../output/"


SUPER_START = 4
OBS_START = 4+29

THRESHOLD = {"psur_obs":[850,1100],
    "t2m_obs":[-40,55],
    "q2m_obs":[0,30],
    "rh2m_obs":[0,100],
    "w10m_obs":[0,30],
    "d10m_obs":[0,360],
    "u10m_obs":[-30,30],
    "v10m_obs":[-30,30],
    "RAIN_obs":[0,400],
    "psfc_M":[850,1100],
    "t2m_M":[-40,55],
    "q2m_M":[0,30],
    "rh2m_M":[0,100],
    "w10m_M":[0,30],
    "d10m_M":[0,360],
    "u10m_M":[-30,30],
    "v10m_M":[-30,30],
    "SWD_M":[0,1500],
    "GLW_M":[0,800],
    "HFX_M":[-400,1000],
    "LH_M":[-100,1000],
    "RAIN_M":[0,400],
    "PBLH_M":[0,6000],
    "TC975_M":[-50,45],
    "TC925_M":[-50,45],
    "TC850_M":[-55,40],
    "TC700_M":[-60,35],
    "TC500_M":[-70,30],
    "wspd975_M":[0,60],
    "wspd925_M":[0,70],
    "wspd850_M":[0,80],
    "wspd700_M":[0,90],
    "wspd500_M":[0,100],
    "Q975_M":[0,30],
    "Q925_M":[0,30],
    "Q850_M":[0,30],
    "Q700_M":[0,25],
    "Q500_M":[0,25]
}

# 将数据转换为csv格式
def transfer_data_to_csv(file_name, output):
    data=Dataset(file_name)
    keys = list(data.variables.keys())
    tmp = []
    for i in data.variables.values():
        tmp.append(i[:].filled())
    tmp = tmp[3:]
    
    value = []
    for i in tmp:
        i = i.transpose(2,0,1)
        i = i.reshape(i.shape[0]*i.shape[1] * i.shape[2])
        value.append(i)
    value = np.array(value)   
    
    columns = keys[3:]
    stations=[]
    dates=[]
    times=[]
    station_date_time = []
    for station in data.variables['station'][:]:
        for date in data.variables['date'][:]:
            for time in data.variables['foretimes'][:]:
                stations.append(str(station))
                dates.append(str(int(date)))
                times.append(str(time))
                station_date_time.append(str(station)+"_"+str(int(date))+"_"+str(time))
    df = pd.DataFrame(value.transpose(),columns=columns)
    df.insert(0,"foretimes",times)
    df.insert(0,"dates",dates)
    df.insert(0,"stations",stations)
    df.insert(0,"station_date_time",station_date_time)    
    
    df = df.replace(-9999,np.nan) # 将填充值设置为缺失值
    df.to_csv(output, index=False)

# 合并数据
def combine_data(file_names):
    for i,file_name in enumerate(file_names):
        transfer_data_to_csv(base_path_1+file_name, base_path_2+str(i)+'.csv')
        tmp_df = load_data(base_path_2+str(i)+'.csv')
        if i==0:
            df = tmp_df
        else:
            df = pd.concat([
            df, 
            tmp_df, 
            ]).reset_index(drop=True)
    df.to_csv("..\\data\\all_data.csv", index=False)

# 加载数据
def load_data(file_name):
    df = pd.read_csv(file_name,sep=',')
    df['dates'] = pd.to_datetime(df.dates, format='%Y%m%d%H') + df.foretimes.apply(lambda x: pd.Timedelta(x, unit='h')) 
    return df

# 将数据分站点存储
def divide_data_by_ports():
    # TODO
    pass

def check(df):
    """
    检查一个样本的特征是否合法
    :return: True or False
    """
    for column in df.columns: # 将超出范围值设置为缺失值
        if column in THRESHOLD.keys():
            df[THRESHOLD[column][0]>df[column]]=np.nan
            df[THRESHOLD[column][1]<df[column]]=np.nan   
    return df 

def output_mean(file1, file2, columns, output):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1[columns] = df1[columns] / 2 + df2[columns] / 2
    df1.to_csv(output, index=False)

def transfer_valid_to_test_format():
    """
    将valid转换为官方测评脚本需要的格式，方便测试分数。
    """
    prd_time = "2018-08-28 03"
    output_path = "..\\data\\"
    # 转换数据格式
    if(not os.path.exists("..\\data\\valid.csv")):
        transfer_data_to_csv(valid_file, "..\\data\\valid.csv")
    valid_df = load_data("..\\data\\valid.csv")
    # valid_df = fill_missing_data(valid_df)
    valid_df['dates'] = pd.to_datetime(valid_df.dates, format='%Y%m%d%H') + valid_df.foretimes.apply(lambda x: pd.Timedelta(x, unit='h'))
    valid_df = valid_df.drop_duplicates(subset=['stations', 'dates'], keep='last').fillna(-9999)

    # 转换OBS
    obs_name = "obs.csv"
    tar_cols = pd.Index(["t2m_obs", "rh2m_obs", "w10m_obs"])
    obs_dates = []
    stat_l , stat_r = valid_df.stations.drop_duplicates().min(), valid_df.stations.drop_duplicates().max()
    for i in range(stat_l, stat_r+1): # 所有站点
        for j in range(37):
            obs_dates.append('{}_{:02d}'.format(i, j))
    df_obs = valid_df.loc[lambda x: x.dates >= prd_time, tar_cols].rename(columns={"t2m_obs": 't2m', "rh2m_obs": 'rh2m', "w10m_obs": 'w10m'})
    df_obs.insert(0, 'FORE_data', np.array(obs_dates))
    df_obs.to_csv(output_path+obs_name)
    
    # 转换ANEN
    ane_name = "anen.csv"
    tar_cols = pd.Index(["t2m_M", "rh2m_M", "w10m_M"])
    df_ane = valid_df.loc[lambda x: x.dates >= prd_time, tar_cols].rename(columns={"t2m_M": 't2m', "rh2m_M": 'rh2m', "w10m_M": 'w10m'})
    df_ane.insert(0, 'ANEN_data', np.array(obs_dates))
    df_ane.to_csv(output_path+ane_name)

# 删除训练集第一天的数据，因为其超算都是缺失
def delete_train_day1(data):
    item = set(train.dates)
    item.remove(2015030103)
    return data[data.dates.isin(item)]
       
# 利用均值填充第一天的超算数据       
def fill_train_day1(data):
    for foretime in range(37):
        if any(data[data['foretimes'] == foretime] == -9999): # 有缺失
            value = data[data['foretimes'] == foretime].mean().values[SUPER_START:OBS_START]
            data.iloc[foretime, SUPER_START:OBS_START]= value
    return data

def fill_with_dup(data, attr_need):
    dup_time_fol = set(range(24,36+1)) # 和后面12小时重复
    dup_time_pre = set(range(0,12+1)) # 和前面12小时重复
    foretime = data['foretimes'].values
    value = data[attr_need].values
    num = 0
    for i in range(12, value.shape[0] - 12):
        for j in range(value.shape[1]):
            if np.isnan(value[i, j]):
                if foretime[i] in dup_time_pre and not np.isnan(value[i-12, j]):
                    value[i, j] = value[i-12, j]
                elif foretime[i] in dup_time_fol and not np.isnan(value[i+12, j]):
                    value[i, j] = value[i+12, j]
                if (foretime[i] in dup_time_pre and np.isnan(value[i-12, j])) or (foretime[i] in dup_time_fol and np.isnan(value[i+12, j])):
                    num += 1
    data.loc[:, attr_need] = value
    print("unfilled dup_time data: ", num)
    return data

# 缺失值只有1,2,或者3个，线性填充
def fill_with_linear(data, attr_need):
    num = 0
    values1 = data[attr_need].values
    for i in range(1, values1.shape[0] - 1):
        for j in range(values1.shape[1]):
            if np.isnan(values1[i, j]):
                if not np.isnan(values1[i - 1, j]) and not np.isnan(values1[i + 1, j]):
                    values1[i, j] = (values1[i - 1, j] + values1[i + 1, j]) / 2
                    num += 1
                    continue
                if i < 2:
                    continue
                if not np.isnan(values1[i - 2, j]) and not np.isnan(values1[i + 1, j]):
                    values1[i, j] = (values1[i - 2, j] + values1[i + 1, j] * 2) / 3
                    values1[i - 1, j] = (values1[i - 2, j] * 2 + values1[i + 1, j]) / 3
                    num += 2
                    continue
                if i >= values1.shape[0] - 2:
                    continue
                if not np.isnan(values1[i - 1, j]) and not np.isnan(values1[i + 2, j]):
                    values1[i, j] = (values1[i - 1, j] * 2 + values1[i + 2, j]) / 3
                    values1[i + 1, j] = (values1[i - 1, j] + values1[i + 2, j] * 2) / 3
                    num += 2
                    continue
                if not np.isnan(values1[i - 2, j]) and not np.isnan(values1[i + 2, j]):
                    values1[i - 1, j] = (values1[i - 2, j] * 3 + values1[i + 2, j]) / 4
                    values1[i, j] = (values1[i - 2, j] * 2 + values1[i + 2, j] * 2) / 4
                    values1[i + 1, j] = (values1[i - 2, j] + values1[i + 2, j] * 3) / 4
                    num += 3
                    continue
                    # print np.isnan(values1).sum()
    data.loc[:, attr_need] = values1
    # print "group.values.shape: ", group.values.shape
    print("filled num: ", num)
    return data

    
# 填充缺失值
def fill_missing_data(data):
    key_list = ['stations', 'dates']
    obs_list = ['psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs',
        'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
    M_list = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M',
        'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M',
        'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', 'wspd925_M',
        'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
        'Q700_M', 'Q500_M']
    tar_list = ['t2m_obs', 'rh2m_obs', 'w10m_obs']
    # attr_need = ['t2m_M', 'psfc_M', 'q2m_M', 'w10m_M', 'd10m_M', 'SWD_M', 'GLW_M', 'LH_M', 'HFX_M', 'RAIN_M', 'PBLH_M', 'psur_obs','t2m_obs','q2m_obs', 'w10m_obs', 'd10m_obs', 'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
    attr_need = obs_list
    data = fill_with_dup(data, attr_need) # 利用重复时段的值进行填充
    # data = fill_train_day1(data) # 利用均值填充第一天的超算数据   
    # attr_need = obs_list + M_list
    # data = fill_with_linear(data, attr_need) # 缺失值只有1,2,或者3个,线性填充
    return data