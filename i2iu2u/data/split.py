import pandas as pd
import datetime as dt
from conf.conf import logging,settings
<<<<<<< HEAD
from data.data import save_data_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')
=======

>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f

def get_test_max_date(df:pd.DataFrame,column:str,test_interval_days=14):
    MAX_DATE = df[column].max()
    MIN_DATE = df[column].min()
    TEST_INTERVAL_DAYS = test_interval_days
    TEST_MAX_DATE = MAX_DATE - dt.timedelta(days = TEST_INTERVAL_DAYS)
    
    return TEST_MAX_DATE
    


<<<<<<< HEAD
def get_global_train_test(df:pd.DataFrame,column:str,TEST_MAX_DATE:dt.timedelta,train_path=settings.PATH.global_train,test_path=settings.PATH.global_test):
    
    logging.info('spliting dfs')
=======
def get_global_train_test(df:pd.DataFrame,column:str,TEST_MAX_DATE:dt.timedelta):
>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f
    
    global_train = df.loc[df[column] < TEST_MAX_DATE]
    global_test = df.loc[df[column] >= TEST_MAX_DATE]

    global_train = global_train.dropna().reset_index(drop = True)
    
<<<<<<< HEAD
    logging.info('split is ready')
    
    save_data_csv(train_path,global_train)
    save_data_csv(test_path,global_test)
    
    return global_test,global_train

def get_local_train_test(global_train:pd.DataFrame,column:str,train_path=settings.PATH.local_train,test_path=settings.PATH.local_test):
    
    logging.info('spliting dfs')
    
=======
    return global_test,global_train

def get_local_train_test(global_train:pd.DataFrame,column:str):
>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f
    local_train_thresh = global_train[column].quantile(q = .7, interpolation = 'nearest')
    local_train = global_train.loc[global_train[column] < local_train_thresh]
    local_test = global_train.loc[global_train[column] >= local_train_thresh]
    local_test = local_test.loc[local_test['user_id'].isin(local_train['user_id'].unique())]
<<<<<<< HEAD
    
    logging.info('split is ready')
    
    save_data_csv(train_path,local_train)
    save_data_csv(test_path,local_test)
    
    return local_train,local_test

def get_positive_negative_preds(local_test_preds, local_test,positive_preds_path=settings.PATH.positive_preds,negative_preds_path=settings.PATH.negative_preds):
    
    logging.info('spliting dfs')
    positive_preds = pd.merge(local_test_preds, local_test, how = 'inner', on = ['user_id', 'movie_id'])
    positive_preds['target'] = 1
    negative_preds = pd.merge(local_test_preds, local_test, how = 'left', on = ['user_id', 'movie_id'])
    negative_preds['target'] = 0
    logging.info('split is ready')
    save_data_csv(positive_preds_path,positive_preds)
    save_data_csv(negative_preds_path,negative_preds)
    
    return positive_preds,negative_preds


def get_train_test_users(local_test):
    logging.info('spliting dfs')
    train_users, test_users = train_test_split(
    local_test['user_id'].unique(),
    test_size = .2,
    random_state = 13
    )
    logging.info('split is ready')
    
    return train_users, test_users

def get_cbm_train_test_set(positive_preds,negative_preds,train_users,test_users,movies_metadata,ITEM_FEATURES=settings.VARIABLES.ITEM_FEATURES,cbm_train_set_path=settings.PATH.cbm_train_set,cbm_test_set_path=settings.PATH.cbm_test_set):
    logging.info('spliting dfs')
    cbm_train_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(train_users)],
    negative_preds.loc[negative_preds['user_id'].isin(train_users)]]
    )
)
    cbm_test_set = shuffle(
    pd.concat(
    [positive_preds.loc[positive_preds['user_id'].isin(test_users)],
    negative_preds.loc[negative_preds['user_id'].isin(test_users)]]
    )  
)
    
    movies_metadata=movies_metadata.drop('watch_duration_minutes',axis=1)
    cbm_train_set = pd.merge(cbm_train_set, movies_metadata[['movie_id'] + ITEM_FEATURES],
                         how = 'left', on = ['movie_id'])
    cbm_test_set = pd.merge(cbm_test_set, movies_metadata[['movie_id'] + ITEM_FEATURES],
                        how = 'left', on = ['movie_id'])
    logging.info('split is ready')
    
    save_data_csv(cbm_train_set_path,cbm_train_set)
    save_data_csv(cbm_test_set_path,cbm_test_set)
    
    return cbm_train_set,cbm_test_set
    
    
def get_x_y_split(cbm_train_set,cbm_test_set,ID_COLS=settings.VARIABLES.ID_COLS,DROP_COLS=settings.VARIABLES.DROP_COLS,TARGET=settings.VARIABLES.TARGET):
    logging.info('spliting dfs')
    X_train, y_train = cbm_train_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_train_set[TARGET]
    X_test, y_test = cbm_test_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_test_set[TARGET]
    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])
    logging.info('split is ready')
    return X_train, y_train,X_test, y_test
=======
    return local_train,local_test

    
>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f
