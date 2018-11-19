import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

def get_df(path):
    df = pd.read_csv(path)
    date = pd.to_datetime(df.dates, format='%Y%m%d%H')
    df['dates'] = date + df.foretimes.apply(lambda x: pd.Timedelta(x, unit='h'))
    df['t2m_obj'] = df.t2m_obs - df.t2m_M
    df['rh2m_obj'] = df.rh2m_obs - df.rh2m_M
    df['w10m_obj'] = df.w10m_obs - df.w10m_M
    min_date = date.min()
    df.insert(0, 'nday', (date - min_date).apply(lambda x: x.days))
    return df.drop('station_date_time', axis=1)

str_today = '1103'

df = get_df(f'all_data.csv')
cat_var_list = ['stations', 'foretimes']

# 既包含当时的，也包含均值
def get_fea(df, postfix):
    df_fea = pd.merge(
        df, 
        df.drop('stations', axis=1).groupby(['dates', 'foretimes'], as_index=False).mean(), 
        left_on=['dates', 'foretimes'], 
        right_on=['dates', 'foretimes']
    )
    return df_fea.rename(columns=dict(zip(df_fea.columns[4:], [f'{col}_{postfix}' for col in df_fea.columns[4:]])))

df_all = df.copy()
df_fea = df.copy().drop(['nday'], axis=1)
df_fea['dates'] = df_fea['dates'] + pd.Timedelta('2 days')
df_all = pd.merge(
    df_all, 
    get_fea(df_fea, '2d'), 
    left_on=['stations', 'dates', 'foretimes'], 
    right_on=['stations', 'dates', 'foretimes']
)
df_fea = df.copy().drop(['nday'], axis=1)
df_fea['dates'] = df_fea['dates'] + pd.Timedelta('7 days')
df_all = pd.merge(
    df_all, 
    get_fea(df_fea, '7d'), 
    left_on=['stations', 'dates', 'foretimes'], 
    right_on=['stations', 'dates', 'foretimes']
)
df_fea = df.copy().drop(['nday'], axis=1)
df_fea['dates'] = df_fea['dates'] + pd.Timedelta('365 days')
df_all = pd.merge(
    df_all, 
    get_fea(df_fea, '1y'), 
    left_on=['stations', 'dates', 'foretimes'], 
    right_on=['stations', 'dates', 'foretimes']
)

# 所有columns里包含d10m的列，都这么处理
d10m_var_list = [col for col in df_all.columns if 'd10m' in col]
d10m_var_list.remove('d10m_obs')
for d10m_var in d10m_var_list:
    df_all[d10m_var].fillna(-1, inplace=True)
    df_all[d10m_var] = df_all[d10m_var].round().astype(int)
cat_var_list += d10m_var_list

df_all.insert(2,"hour",df_all.dates.dt.hour)
df_all.insert(2,"week",df_all.dates.dt.week)
df_all.insert(2,"day",df_all.dates.dt.day)
df_all.insert(2,"month",df_all.dates.dt.month)
df_all.drop('dates', axis=1, inplace=True)
cat_var_list += ['month', 'day', 'week', 'hour']

cat_var_lblenc_dict = {}
for cat_var in cat_var_list:
    lblenc = LabelEncoder().fit(df_all[cat_var])
    df_all[cat_var] = lblenc.transform(df_all[cat_var])
    cat_var_lblenc_dict[cat_var] = lblenc
    
df_test = df_all[lambda x: x.nday == x.nday.max()]
df_train = df_all[lambda x: x.nday < x.nday.max()]

df_train = df_train[lambda x: (x.psfc_M >= 850.) & (x.psfc_M <= 1100.) & \
                    (x.t2m_M >= -40.) & (x.t2m_M <= 55.) & \
                    (x.rh2m_M >= 0.) & (x.rh2m_M <= 100.) & \
                    (x.w10m_M >= 0.) & (x.w10m_M <= 30.)
                   ]

df_train.dropna(subset=['t2m_obj', 'rh2m_obj', 'w10m_obj'], inplace=True)
df_train_y = df_train[['t2m_obj', 'rh2m_obj', 'w10m_obj']]
df_train.drop(df_train.columns[36:48], axis=1, inplace=True)
df_test.drop(df_train.columns[36:48], axis=1, inplace=True)

split_end = df_train.nday.max()
val_days = 30
df_val = df_train[lambda x: (x.nday >= split_end-val_days+1) & (x.nday <= split_end)]
df_tr = df_train[lambda x: x.nday < split_end-val_days+1]

df_tr_X = df_tr[df_train.columns[1:]]
df_val_X = df_val[df_val.columns[1:]]
df_test_X = df_test[df_test.columns[1:]]

df_val_y = df_train_y.loc[df_val_X.index]
df_tr_y = df_train_y.loc[df_tr_X.index]

def train_bst(X_tr, y_tr, X_val, y_val, cat_val_list):
    params = {
        'num_leaves': 31, 
        'objective': 'regression_l2',
        'min_data_in_leaf': 300,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'metric': 'rmse',
        'num_threads': 8, 
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

X_tr, y_tr, X_val, y_val = df_tr_X, df_tr_y.values, df_val_X, df_val_y.values
bst_t2m = train_bst(X_tr, y_tr[:, 0], X_val, y_val[:, 0], cat_var_list)
bst_t2m.save_model('bst_t2m_gbdt_{}_new.txt'.format(str_today))
bst_rh2m = train_bst(X_tr, y_tr[:, 1], X_val, y_val[:, 1], cat_var_list)
bst_rh2m.save_model('bst_rh2m_gbdt_{}_new.txt'.format(str_today))
bst_w10m = train_bst(X_tr, y_tr[:, 2], X_val, y_val[:, 2], cat_var_list)
bst_w10m.save_model('bst_w10m_gbdt_{}_new.txt'.format(str_today))
df_submit = pd.DataFrame([
    [f'{item[0]}_{int(item[1]):02d}' for item in zip(cat_var_lblenc_dict['stations'].inverse_transform(df_test.stations), 
                                                     cat_var_lblenc_dict['foretimes'].inverse_transform(df_test.foretimes))], 
    (df_test.t2m_M + bst_t2m.predict(df_test_X)).tolist(), 
    (df_test.rh2m_M + bst_rh2m.predict(df_test_X)).tolist(), 
    (df_test.w10m_M + bst_w10m.predict(df_test_X)).tolist(), 
]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
df_submit.to_csv(f'forecast-2018{str_today}03_gbdt_new.csv', index=False)

def train_bst(X_tr, y_tr, X_val, y_val, cat_val_list):
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
        'num_threads': 8, 
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

bst_t2m = train_bst(X_tr, y_tr[:, 0], X_val, y_val[:, 0], cat_var_list)
bst_t2m.save_model('bst_t2m_dart_{}_new.txt'.format(str_today))
bst_rh2m = train_bst(X_tr, y_tr[:, 1], X_val, y_val[:, 1], cat_var_list)
bst_rh2m.save_model('bst_rh2m_dart_{}_new.txt'.format(str_today))
bst_w10m = train_bst(X_tr, y_tr[:, 2], X_val, y_val[:, 2], cat_var_list)
bst_w10m.save_model('bst_w10m_dart_{}_new.txt'.format(str_today))
df_submit = pd.DataFrame([
    [f'{item[0]}_{int(item[1]):02d}' for item in zip(cat_var_lblenc_dict['stations'].inverse_transform(df_test.stations), 
                                                     cat_var_lblenc_dict['foretimes'].inverse_transform(df_test.foretimes))], 
    (df_test.t2m_M + bst_t2m.predict(df_test_X)).tolist(), 
    (df_test.rh2m_M + bst_rh2m.predict(df_test_X)).tolist(), 
    (df_test.w10m_M + bst_w10m.predict(df_test_X)).tolist(), 
]).T.rename(columns={0: 'FORE_data', 1: 't2m', 2: 'rh2m', 3: 'w10m'})
df_submit.to_csv(f'forecast-2018{str_today}03_dart_new.csv', index=False)

df_submit1 = pd.read_csv(f'forecast-2018{str_today}03_gbdt_new.csv')
df_submit2 = pd.read_csv(f'forecast-2018{str_today}03_dart_new.csv')

df_submit1['t2m'] = (df_submit1['t2m'] + df_submit2['t2m']) / 2
df_submit1['rh2m'] = (df_submit1['rh2m'] + df_submit2['rh2m']) / 2
df_submit1['w10m'] = (df_submit1['w10m'] + df_submit2['w10m']) / 2

df_submit1.to_csv(f'forecast-2018{str_today}03_new.csv', index=False)