import pandas as pd
import datetime as dt
from conf.conf import logging,settings


def get_test_max_date(df:pd.DataFrame,column:str,test_interval_days=14):
    MAX_DATE = df[column].max()
    MIN_DATE = df[column].min()
    TEST_INTERVAL_DAYS = test_interval_days
    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)
    
    return TEST_MAX_DATE
    


def get_global_train_test(df:pd.DataFrame,column:str,TEST_MAX_DATE:dt.timedelta):
    
    global_train = df.loc[df[column] < TEST_MAX_DATE]
    global_test = df.loc[df[column] >= TEST_MAX_DATE]

    global_train = global_train.dropna().reset_index(drop = True)
    
    return global_test,global_train

def get_local_train_test(global_train:pd.DataFrame,column:str):
    local_train_thresh = global_train['last_watch_dt'].quantile(q = .7, interpolation = 'nearest')
    local_train = global_train.loc[global_train['last_watch_dt'] < local_train_thresh]
    local_test = global_train.loc[global_train['last_watch_dt'] >= local_train_thresh]
    return local_train,local_test

    