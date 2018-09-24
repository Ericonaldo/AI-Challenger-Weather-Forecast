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

from data_utils import *
    
test_file = "..\\data\\ai_challenger_wf2018_testa1_20180829-20180924.nc"

if __name__ == '__main__':
    # 转换数据格式
    transfer_data_to_csv(test_file, "..\\data\\test.csv")   
    test_df = load_data(".\\data\\test.csv")
    test_df = fill_missing_data(train_df)
        
# -------------------        
'''
    loss_data_process_main(pre_train_flag=True)
    pre_main(city="bj")
    pre_main(city='ld')
    loss_data_process_main(pre_train_flag=False)
'''
    '''
    获取全部的数据
    利用前三预测后一个值来提交结果，迭代预测
    '''
    # post_data(city="bj")
    #     model_1(city='bj')
    # post_data(city="ld")
    #     model_1(city='ld')

    # ans = "test_id,PM2.5,PM10,O3\n"
    # ans1 = model_1(city="bj")
    # ans2 = model_1(city="ld")
    # ans_file = base_path_3 + "ans.csv"
    # f_to = open(ans_file, 'wb')
    # f_to.write(ans + ans1 + ans2)
    # f_to.close()
    # city = 'bj'
    # df = load_data_process(city=city)
    # for station, group in df.groupby("station_id"):
    #     print station, group.values.shape
    # # print df.values.shape
    #
    # city = 'ld'
    # df = load_data_process(city=city)
    # for station, group in df.groupby("station_id"):
    #     print station, group.values.shape
    # # print df.values.shape
