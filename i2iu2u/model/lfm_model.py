import pandas as pd
import numpy as np
import datetime as dt
from conf.conf import logging,settings
from conf.conf import dataset
from util.util import save_model, load_model
from tqdm import tqdm
from lightfm import LightFM
from data.data import get_data_parquet
from data.split import  get_test_max_date,get_global_train_test,get_local_train_test
from data.update_data import get_last_watch_dt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')





def compute_popularity(df: pd.DataFrame, item_id: str, max_candidates: int):
    """
    calculates mean rating to define popular titles
    """
    popular_titles = df.groupby(item_id).agg({'rating': np.mean})\
                     .sort_values(['rating'], ascending=False).head(max_candidates).index.values

    return popular_titles

def get_lightfm_mapping(dataset, df:pd.DataFrame):
    dataset.fit(df['user_id'].unique(), df['movie_id'].unique())
    lightfm_mapping = dataset.mapping()
    lightfm_mapping = {
        'users_mapping': lightfm_mapping[0],
        'user_features_mapping': lightfm_mapping[1],
        'items_mapping': lightfm_mapping[2],
        'item_features_mapping': lightfm_mapping[3],
    }
    
    lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
    lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

    return dataset, lightfm_mapping


def get_item_name_mapper(df):
    item_name_mapper = dict(zip(df['movie_id'], df['title']))
    
    return item_name_mapper


def df_to_tuple_iterator(df: pd.DataFrame):
    '''
    :df: pd.DataFrame, interactions dataframe
    returns iterator
    '''
    
    return zip(*df.values.T)


def get_train_mat_and_weights(dataset,local_train):
    train_mat, train_mat_weights = dataset.build_interactions(df_to_tuple_iterator(local_train[['user_id', 'movie_id']]))
    
    return train_mat, train_mat_weights


def model_fit_partial(model,train_mat,EPOCHS = 20):
    for _ in tqdm(range(EPOCHS), total = EPOCHS):
            model.fit_partial(
            train_mat,
            num_threads = 4
        )
            
    return model


def predict(row_id, all_cols, num_threads, model_path):
    model = load_model(model_path)
    
    return model.predict(row_id, all_cols, num_threads, model_path)


def generate_lightfm_recs_mapper(
        model: object,
        item_ids: list,
        known_items: dict,
        user_features: list,
        item_features: list,
        N: int,
        user_mapping: dict,
        item_inv_mapping: dict,
        num_threads: int = 4
        ):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features = user_features,
            item_features = item_features,
            num_threads = num_threads)
        
        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]
        
        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
            
        return final_recs[:N]
    
    return _recs_mapper



def get_local_preds(local_test,lightf_mapping,lfm_model,item_name_mapper):
    local_test_preds = pd.DataFrame({'user_id': local_test['user_id'].unique()})
    all_cols = list(lightf_mapping['items_mapping'].values())
    mapper = generate_lightfm_recs_mapper(
    lfm_model, 
    item_ids = all_cols, 
    known_items = dict(),
    N = 10,
    user_features = None, 
    item_features = None, 
    user_mapping = lightf_mapping['users_mapping'],
    item_inv_mapping = lightf_mapping['items_inv_mapping'],
    num_threads = 20)
    local_test_preds['movie_id'] = local_test_preds['user_id'].map(mapper)
    local_test_preds = local_test_preds.explode('movie_id')
    local_test_preds['rank'] = local_test_preds.groupby('user_id').cumcount() + 1 
    local_test_preds['item_name'] = local_test_preds['movie_id'].map(item_name_mapper)
    print(f'Data shape{local_test_preds.shape}')
    local_test_preds.head(50)
    return local_test_preds


def get_positive_negative_preds(local_test_preds, local_test):
    positive_preds = pd.merge(local_test_preds, local_test, how = 'inner', on = ['user_id', 'movie_id'])
    positive_preds['target'] = 1
    negative_preds = pd.merge(local_test_preds, local_test, how = 'left', on = ['user_id', 'movie_id'])
    negative_preds['target'] = 0
    return positive_preds,negative_preds

def get_train_test_users(local_test):
    train_users, test_users = train_test_split(
    local_test['user_id'].unique(),
    test_size = .2,
    random_state = 13
    )
    return train_users, test_users

def get_cbm_train_test_set(positive_preds,negative_preds,train_users,test_users,movies_metadata,ITEM_FEATURES):
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
    return cbm_train_set,cbm_test_set
    
    
def get_x_y_split(cbm_train_set,cbm_test_set,ID_COLS,DROP_COLS,TARGET):
    X_train, y_train = cbm_train_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_train_set[TARGET]
    X_test, y_test = cbm_test_set.drop(ID_COLS + DROP_COLS + TARGET, axis = 1), cbm_test_set[TARGET]
    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])
    return X_train, y_train,X_test, y_test

def get_cbm(X_train, y_train,X_test, y_test):
    X_train=X_train.drop('watch_duration_minutes',axis=1)
    cbm_classifier = CatBoostClassifier(
    loss_function = 'CrossEntropy',
    iterations = 5000,
    learning_rate = .1,
    depth = 6,
    random_state = 1234,
    verbose = True
)
    cbm_classifier.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds = 100,
    verbose = False
)
    return cbm_classifier
    
def get_global_cbm_test_preds(local_train,top_k,lightf_mapping,movies_metadata,ITEM_FEATURES):
    
    global_test_predictions = pd.DataFrame({
    'user_id': global_test['user_id'].unique()
        }
    )
    
    global_test_predictions = global_test_predictions.loc[global_test_predictions['user_id'].isin(local_train.user_id.unique())]
    
    watched_movies = local_train.groupby('user_id')['movie_id'].apply(list).to_dict()

    mapper = generate_lightfm_recs_mapper(
        lfm_model, 
        item_ids = all_cols, 
        known_items = watched_movies,
        N = top_k,
        user_features = None, 
        item_features = None, 
        user_mapping = lightf_mapping['users_mapping'],
        item_inv_mapping = lightf_mapping['items_inv_mapping'],
        num_threads = 10
    )

    global_test_predictions['movie_id'] = global_test_predictions['user_id'].map(mapper)
    global_test_predictions = global_test_predictions.explode('movie_id').reset_index(drop=True)
    global_test_predictions['rank'] = global_test_predictions.groupby('user_id').cumcount() + 1 
    
    cbm_global_test = pd.merge(global_test_predictions, movies_metadata[['movie_id'] + ITEM_FEATURES],
                         how = 'left', on = ['movie_id'])
    cbm_global_test = cbm_global_test.fillna(cbm_global_test.mode().iloc[0])
    
    return global_test_predictions,cbm_global_test

def get_global_ranks(cbm_global_test,cbm,X_train):
    cbm_global_test = cbm_global_test.drop('watch_duration_minutes')
    cbm_global_test['cbm_preds'] = cbm.predict_proba(cbm_global_test[X_train.columns])[:, 1]
    cbm_global_test = cbm_global_test.sort_values(by = ['user_id', 'cbm_preds'], ascending = [True, False])
    cbm_global_test['cbm_rank'] = cbm_global_test.groupby('user_id').cumcount() + 1
    
    return cbm_global_test


    
def calc_metrics(df_true, df_pred, k: int = 10, target_col = 'rank'):
    """
    calculates confusion matrix based metrics
    :df_true: pd.DataFrame
    :df_pred: pd.DataFrame
    :k: int, 
    """
    # prepare dataset
    df = df_true.set_index(['user_id', 'movie_id']).join(df_pred.set_index(['user_id', 'movie_id']))
    df = df.sort_values(by = ['user_id', target_col])
    df['users_watch_count'] = df.groupby(level = 'user_id')[target_col].transform(np.size)
    df['cumulative_rank'] = df.groupby(level = 'user_id').cumcount() + 1
    df['cumulative_rank'] = df['cumulative_rank'] / df[target_col]
    
    # params to calculate metrics
    output = {}
    num_of_users = df.index.get_level_values('user_id').nunique()

    # calc metrics
    df[f'hit@{k}'] = df[target_col] <= k
    output[f'Precision@{k}'] = (df[f'hit@{k}'] / k).sum() / num_of_users
    output[f'Recall@{k}'] = (df[f'hit@{k}'] / df['users_watch_count']).sum() / num_of_users
    output[f'MAP@{k}'] = (df["cumulative_rank"] / df["users_watch_count"]).sum() / num_of_users
    print(f'Calculated metrics for top {k}')
    return output

    
def implify_metrics(global_test,global_test_predictions,cbm_global_test):
    lfm_metrics = calc_metrics(global_test, global_test_predictions)
    full_pipeline_metrics = calc_metrics(global_test, cbm_global_test, target_col = 'cbm_rank')
    metrics_table = pd.concat(
    [pd.DataFrame([lfm_metrics]),
    pd.DataFrame([full_pipeline_metrics])],
    ignore_index = True
)
    metrics_table.index = ['LightFM', 'FullPipeline']
    metrics_table = metrics_table.append(metrics_table.pct_change().iloc[-1].mul(100).rename('lift_by_ranker, %'))
    print(metrics_table)
    


# nakonets to ne ebanii functions

interactions = get_data_parquet('data/interactions.parquet')
movies_md = get_data_parquet('data/movies_md_upd.parquet')

interactions = get_last_watch_dt(interactions,['day','month','year'])

TEST_MAX_DATE = get_test_max_date(interactions,'last_watch_dt')

print(f"test max date to split:: {TEST_MAX_DATE}")

global_train,global_test = get_global_train_test(interactions,'last_watch_dt',TEST_MAX_DATE)

local_train,local_test=get_local_train_test(global_train,'last_watch_dt')

dataset,lightf_mapping = get_lightfm_mapping(dataset, local_train)

item_name_mapper = get_item_name_mapper(movies_md)

train_mat, train_mat_weights = get_train_mat_and_weights(dataset,local_train)

lfm_model = LightFM(
    no_components = 64,
    learning_rate = .03,
    loss = 'warp',
    max_sampled = 5,
    random_state = 42
    )

lfm_model = model_fit_partial(lfm_model, train_mat)

save_model('model/lfm.pkl',lfm_model)

all_cols = list(lightf_mapping['items_mapping'].values())

mapper = generate_lightfm_recs_mapper(
    lfm_model, 
    item_ids = all_cols, 
    known_items = dict(),
    N = 10,
    user_features = None, 
    item_features = None, 
    user_mapping = lightf_mapping['users_mapping'],
    item_inv_mapping = lightf_mapping['items_inv_mapping'],
    num_threads = 20
)

local_test_preds = get_local_preds(local_test,lightf_mapping,lfm_model,item_name_mapper)

print(local_test_preds)
print(local_test)
positive_preds,negative_preds = get_positive_negative_preds(local_test_preds, local_test)

train_users, test_users = get_train_test_users(local_test)

print(movies_md)
ITEM_FEATURES = ['age_rating','duration','num_users_watched','time_watched','avg_watch_coef','rating','score']
cbm_train_set,cbm_test_set = get_cbm_train_test_set(positive_preds,negative_preds,train_users,test_users,movies_md,ITEM_FEATURES)

ID_COLS = ['user_id', 'movie_id']
TARGET = ['target']
DROP_COLS = ['item_name', 'year', 'month', 'day','last_watch_dt']

X_train, y_train,X_test, y_test = get_x_y_split(cbm_train_set,cbm_test_set,ID_COLS,DROP_COLS,TARGET)

cbm = get_cbm(X_train, y_train,X_test, y_test)

save_model('model/cbm.pkl',lfm_model)

global_test_predictions,cbm_global_test =  get_global_cbm_test_preds(local_train,100,lightf_mapping,movies_md,ITEM_FEATURES)

cbm_global_test = get_global_ranks(cbm_global_test,cbm,X_train)

implify_metrics(global_test,global_test_predictions,cbm_global_test)