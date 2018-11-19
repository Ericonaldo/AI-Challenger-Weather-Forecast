import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='training for ai challenger weather forecast')
    parser.add_argument('--test_day', type=str, default='2018-10-28', help='which day to forecast (default: 2018-10-28)')
    args = parser.parse_args()
    return args

def get_df(path):
    df = pd.read_csv(path)
    date = pd.to_datetime(df.dates, format='%Y%m%d%H')
    df['dates'] = date + df.foretimes.map(lambda x: pd.Timedelta(x, unit='h'))
    # 以观测值减去睿图值作为预测目标
    df['t2m_obj'] = df.t2m_obs - df.t2m_M
    df['rh2m_obj'] = df.rh2m_obs - df.rh2m_M
    df['w10m_obj'] = df.w10m_obs - df.w10m_M
    min_date = date.min()
    # 以最早一天为0，每过一天+1，每天有0到36共37个foretimes
    df.insert(0, 'nday', (date - min_date).apply(lambda x: x.days))
    return df.drop('station_date_time', axis=1)

# 找到某一天的nday，一般用来定位测试样本的nday
def get_nday_from_date(df, date):
    return df[lambda x: (x.dates.dt.date.astype(str) == date) & (x.foretimes == 0)].nday.unique()[0]

# 用days天前的同站点同foretime的观测值、睿图值、目标值作为特征
def get_feature_before_per_station_and_foretime(df, days):
    df = df.drop(['dates'], axis=1)
    df['nday'] = df['nday'] + days
    return df.rename(columns=dict(zip(df.columns[3:], [f'{col}_{days}d_sf' for col in df.columns[3:]])))

# 用days天前的同站点的所有foretimes的观测值、睿图值、目标值的平均值作为特征
def get_feature_before_per_station_mean(df, days):
    df = df.drop(['dates', 'foretimes'], axis=1)
    df['nday'] = df['nday'] + days
    return df.groupby(['nday', 'stations'], as_index=True).mean().rename(
        columns=dict(zip(df.columns[2:], [f'{col}_{days}d_f_mean' for col in df.columns[2:]]))).reset_index()

# 用days天前的同站点的所有foretimes的观测值、睿图值、目标值的标准差作为特征
def get_feature_before_per_station_std(df, days):
    df = df.drop(['dates', 'foretimes'], axis=1)
    df['nday'] = df['nday'] + days
    return df.groupby(['nday', 'stations'], as_index=True).std().rename(
        columns=dict(zip(df.columns[2:], [f'{col}_{days}d_f_std' for col in df.columns[2:]]))).reset_index()

# 用days天前的同foretime的所有stations的观测值、睿图值、目标值的平均值作为特征
def get_feature_before_per_foretime_mean(df, days):
    df = df.drop(['stations', 'dates'], axis=1)
    df['nday'] = df['nday'] + days
    return df.groupby(['nday', 'foretimes'], as_index=True).mean().rename(
        columns=dict(zip(df.columns[2:], [f'{col}_{days}d_s_mean' for col in df.columns[2:]]))).reset_index()

# 用days天前的同foretime的所有stations的观测值、睿图值、目标值的标准差作为特征
def get_feature_before_per_foretime_std(df, days):
    df = df.drop(['stations', 'dates'], axis=1)
    df['nday'] = df['nday'] + days
    return df.groupby(['nday', 'foretimes'], as_index=True).std().rename(
        columns=dict(zip(df.columns[2:], [f'{col}_{days}d_s_std' for col in df.columns[2:]]))).reset_index()

def get_feature_before(df, days):
    df_sf = get_feature_before_per_station_and_foretime(df, days)
    df_s_mean = get_feature_before_per_station_mean(df, days)
    df_s_std = get_feature_before_per_station_std(df, days)
    df_f_mean = get_feature_before_per_foretime_mean(df, days)
    df_f_std = get_feature_before_per_foretime_std(df, days)
    df_sf = pd.merge(df_sf, df_s_mean, left_on=['nday', 'stations'], right_on=['nday', 'stations'])
    df_sf = pd.merge(df_sf, df_s_std, left_on=['nday', 'stations'], right_on=['nday', 'stations'])
    df_sf = pd.merge(df_sf, df_f_mean, left_on=['nday', 'foretimes'], right_on=['nday', 'foretimes'])
    df_sf = pd.merge(df_sf, df_f_std, left_on=['nday', 'foretimes'], right_on=['nday', 'foretimes'])
    return df_sf

def train_gbdt(X_tr, y_tr, X_val, y_val, cat_val_list):
    params = {
        'boosting_type': 'gbdt', 
        'num_leaves': 31, 
        'objective': 'regression_l2',
        'min_data_in_leaf': 300,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'metric': 'rmse',
        'num_threads': 12, # 这个按机器来配置
        'min_data_in_bin': 1, 
        #'device': 'gpu', 
    }

    MAX_ROUNDS = 100000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS, categorical_feature=cat_val_list, 
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    return bst

def train_dart(X_tr, y_tr, X_val, y_val, cat_val_list):
    params = {
        'boosting_type': 'dart', 
        'num_leaves': 31, 
        'objective': 'regression_l2',
        'min_data_in_leaf': 300,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'metric': 'rmse',
        'num_threads': 12, 
        'min_data_in_bin': 1, 
        #'device': 'gpu', 
    }

    MAX_ROUNDS = 10000
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, 
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS, categorical_feature=cat_val_list, 
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100
    )
    return bst

def train(test_day):
    print('Reading data...')
    df = get_df('all_data.csv')
    cat_var_list = ['stations', 'foretimes']
    print('Extracting features...')
    df_fea = df.copy()
    df_fea_2d = get_feature_before(df, 2)
    df_fea_7d = get_feature_before(df, 7)
    df_fea_1y = get_feature_before(df, 365)
    df_fea = pd.merge(df_fea, df_fea_2d)
    df_fea = pd.merge(df_fea, df_fea_7d)
    df_fea = pd.merge(df_fea, df_fea_1y)
    print('Processing for training...')
    # 所有字符串里包含d10m的列，都作为类别变量处理
    d10m_var_list = [col for col in df_fea.columns if 'd10m' in col]
    # 观测值在测试集里没有，所以去掉，不作为特征
    d10m_var_list.remove('d10m_obs')
    for d10m_var in d10m_var_list:
        df_fea[d10m_var].fillna(-1, inplace=True)
        df_fea[d10m_var] = df_fea[d10m_var].round().astype(int)
    cat_var_list += d10m_var_list
    # 将月日小时作为类别变量处理
    df_fea.insert(2, "hour", df_fea.dates.dt.hour)
    df_fea.insert(2, "day", df_fea.dates.dt.day)
    df_fea.insert(2, "month", df_fea.dates.dt.month)
    df_fea.drop('dates', axis=1, inplace=True)
    cat_var_list += ['month', 'day', 'hour']
    # 将类别变量进行编码
    cat_var_lblenc_dict = {}
    for cat_var in cat_var_list:
        lblenc = LabelEncoder().fit(df_fea[cat_var])
        df_fea[cat_var] = lblenc.transform(df_fea[cat_var])
        cat_var_lblenc_dict[cat_var] = lblenc
    test_nday = get_nday_from_date(df, test_day)
    df_test = df_fea[lambda x: x.nday == test_nday]
    df_train = df_fea[lambda x: x.nday < test_nday]
    # 去掉温压湿风睿图值有问题的样本
    df_train = df_train[lambda x: (x.psfc_M >= 850.) & (x.psfc_M <= 1100.) & \
                        (x.t2m_M >= -40.) & (x.t2m_M <= 55.) & \
                        (x.rh2m_M >= 0.) & (x.rh2m_M <= 100.) & \
                        (x.w10m_M >= 0.) & (x.w10m_M <= 30.)
                       ]
    # 去掉要预测的目标值为na的样本（无法建立模型和预测）
    df_train.dropna(subset=['t2m_obj', 'rh2m_obj', 'w10m_obj'], inplace=True)
    df_train_y = df_train[['t2m_obj', 'rh2m_obj', 'w10m_obj']]
    # 去掉测试集里看不到的观测值和目标值的列
    obj_start = 35
    obj_end = 47
    df_train = df_train.drop(df_train.columns[obj_start:obj_end], axis=1)
    df_test = df_test.drop(df_train.columns[obj_start:obj_end], axis=1)
    # 用最近的val_days作为验证集，一般可设为30天，也就是近一个月，因为这样会比较接近于测试集的天气情况
    train_last_nday = df_train.nday.max()
    val_days = 30
    df_val = df_train[lambda x: (x.nday >= train_last_nday-val_days+1) & (x.nday <= train_last_nday)]
    df_tr = df_train[lambda x: x.nday < train_last_nday-val_days+1]
    df_tr_X = df_tr[df_train.columns[1:]]
    df_val_X = df_val[df_val.columns[1:]]
    df_test_X = df_test[df_test.columns[1:]]
    df_tr_y = df_train_y.loc[df_tr_X.index]
    df_val_y = df_train_y.loc[df_val_X.index]
    X_tr, y_tr, X_val, y_val = df_tr_X, df_tr_y.values, df_val_X, df_val_y.values
    print('Training GBDT...')
    gbdt_t2m = train_gbdt(X_tr, y_tr[:, 0], X_val, y_val[:, 0], cat_var_list)
    gbdt_rh2m = train_gbdt(X_tr, y_tr[:, 1], X_val, y_val[:, 1], cat_var_list)
    gbdt_w10m = train_gbdt(X_tr, y_tr[:, 2], X_val, y_val[:, 2], cat_var_list)
    df_gbdt_submit = pd.DataFrame([
        [f'{item[0]}_{int(item[1]):02d}' for item in zip(cat_var_lblenc_dict['stations'].inverse_transform(df_test.stations), 
                                                         cat_var_lblenc_dict['foretimes'].inverse_transform(df_test.foretimes))], 
        (df_test.t2m_M + gbdt_t2m.predict(df_test_X)).tolist(), 
        (df_test.rh2m_M + gbdt_rh2m.predict(df_test_X)).tolist(), 
        (df_test.w10m_M + gbdt_w10m.predict(df_test_X)).tolist(), 
    ]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
    df_gbdt_submit.to_csv('forecast-{}03_gbdt.csv'.format(test_day.replace('-', '')), index=False)
    print('Training DART...')
    dart_t2m = train_dart(X_tr, y_tr[:, 0], X_val, y_val[:, 0], cat_var_list)
    dart_rh2m = train_dart(X_tr, y_tr[:, 1], X_val, y_val[:, 1], cat_var_list)
    dart_w10m = train_dart(X_tr, y_tr[:, 2], X_val, y_val[:, 2], cat_var_list)
    df_dart_submit = pd.DataFrame([
        [f'{item[0]}_{int(item[1]):02d}' for item in zip(cat_var_lblenc_dict['stations'].inverse_transform(df_test.stations), 
                                                         cat_var_lblenc_dict['foretimes'].inverse_transform(df_test.foretimes))], 
        (df_test.t2m_M + dart_t2m.predict(df_test_X)).tolist(), 
        (df_test.rh2m_M + dart_rh2m.predict(df_test_X)).tolist(), 
        (df_test.w10m_M + dart_w10m.predict(df_test_X)).tolist(), 
    ]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
    df_dart_submit.to_csv('forecast-{}03_dart.csv'.format(test_day.replace('-', '')), index=False)
    print('Averaging the results of GBDT and DART...')
    df_submit = df_gbdt_submit.copy()
    df_submit['t2m'] = (df_gbdt_submit['t2m'] + df_dart_submit['t2m']) / 2
    df_submit['rh2m'] = (df_gbdt_submit['rh2m'] + df_dart_submit['rh2m']) / 2
    df_submit['w10m'] = (df_gbdt_submit['w10m'] + df_dart_submit['w10m']) / 2
    df_submit.to_csv('forecast-{}03.csv'.format(test_day.replace('-', '')), index=False)
    
if __name__ == "__main__":
    args = parse_args()
    test_day = args.test_day
    print(f'Forecast the weather of {test_day}')
    train(test_day)