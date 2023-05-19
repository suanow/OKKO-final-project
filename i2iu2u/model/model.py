import pandas as pd
import numpy as np
import datetime as dt
from conf.conf import logging,settings
from conf.conf import dataset
from util.util import save_model, load_model
from data.data import save_data_csv,get_data_csv
from tqdm import tqdm
from lightfm import LightFM
from data.data import get_data_parquet
from data.split import  get_test_max_date,get_global_train_test,get_local_train_test,get_positive_negative_preds,get_train_test_users,get_cbm_train_test_set,get_x_y_split
from data.update_data import get_last_watch_dt
from catboost import CatBoostClassifier


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
    """
    creates a lightfm mapping
    """
    
    logging.info('Creating lightfm mapping')
    
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
    
    logging.info('Lightfm mapping is ready')
    
    return dataset, lightfm_mapping


def get_item_name_mapper(df:pd.DataFrame)-> dict:
    """
    creates a item name mapper
    """
    logging.info('Creating item name mapper')
    item_name_mapper = dict(zip(df['movie_id'], df['title']))
    logging.info('item name mapper is ready')
    return item_name_mapper


def df_to_tuple_iterator(df: pd.DataFrame):
    '''
    :df: pd.DataFrame, interactions dataframe
    returns iterator
    '''
    return zip(*df.values.T)


def get_train_mat_and_weights(dataset,local_train):
    """
    creates a train mat and train weights mat
    """
    logging.info('Creating train mat and train weights mat')
    
    train_mat, train_mat_weights = dataset.build_interactions(df_to_tuple_iterator(local_train[['user_id', 'movie_id']]))
    
    logging.info('train mat and train weights mat is ready')
    
    return train_mat, train_mat_weights


def model_fit_partial(model,train_mat,EPOCHS = 20):
    """
    creates a train mat and train weights mat
    """
    logging.info('fitting model progress')
    for _ in tqdm(range(EPOCHS), total = EPOCHS):
            model.fit_partial(
            train_mat,
            num_threads = 4
        )
    logging.info('model is ready')        
    return model


def predict(row_id, all_cols, num_threads, model_path):
    """
    returns a prediction for chosen model
    """
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



def get_local_preds(local_test,lightf_mapping,lfm_model,item_name_mapper,local_test_preds=settings.PATH.local_test_preds):
    """
    generates local preds 
    """
    
    logging.info('generating local preds')
    
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
    
    logging.info('local preds ready')
    
    return local_test_preds



def get_cbm(X_train, y_train,X_test, y_test):
    logging.info('fitting cbm model')
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
    logging.info('fitting is ready')
    return cbm_classifier
    
def get_global_cbm_test_preds(local_train,top_k,lightf_mapping,movies_metadata,ITEM_FEATURES=settings.VARIABLES.ITEM_FEATURES):
    
    logging.info('getting global preds')
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
    
    logging.info('global preds ready')
    
    return global_test_predictions,cbm_global_test

def get_global_ranks(cbm_global_test,cbm,X_train):
    logging.info('getting global ranks')
    
    cbm_global_test['cbm_preds'] = cbm.predict_proba(cbm_global_test[X_train.columns])[:, 1]
    cbm_global_test = cbm_global_test.sort_values(by = ['user_id', 'cbm_preds'], ascending = [True, False])
    cbm_global_test['cbm_rank'] = cbm_global_test.groupby('user_id').cumcount() + 1
    
    logging.info('global ranks ready')
    
    return cbm_global_test




interactions = get_data_parquet('data/interactions.parquet')
movies_md = get_data_parquet('data/movies_md_upd.parquet')

interactions = get_last_watch_dt(interactions,['day','month','year'])

TEST_MAX_DATE = get_test_max_date(interactions,'last_watch_dt')

global_train,global_test = get_global_train_test(interactions,'last_watch_dt',TEST_MAX_DATE)

local_train,local_test=get_local_train_test(global_train,'last_watch_dt')

dataset,lightf_mapping = get_lightfm_mapping(dataset, local_train)

item_name_mapper = get_item_name_mapper(movies_md)

train_mat, train_mat_weights = get_train_mat_and_weights(dataset,local_train)

lfm_model = LightFM(
    no_components = settings.PARAMS.NO_COMPONENTS,
    learning_rate = settings.PARAMS.LEARNING_RATE,
    loss = settings.PARAMS.LOSS,
    max_sampled = settings.PARAMS.MAX_SAMPLED,
    random_state = settings.PARAMS.RANDOM_STATE
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

positive_preds,negative_preds = get_positive_negative_preds(local_test_preds, local_test)

train_users, test_users = get_train_test_users(local_test)

cbm_train_set,cbm_test_set = get_cbm_train_test_set(positive_preds,negative_preds,train_users,test_users,movies_md)

X_train, y_train,X_test, y_test = get_x_y_split(cbm_train_set,cbm_test_set)

cbm = get_cbm(X_train, y_train,X_test, y_test)

save_model('model/cbm.pkl',lfm_model)

global_test_predictions,cbm_global_test =  get_global_cbm_test_preds(local_train,100,lightf_mapping,movies_md)

cbm_global_test = get_global_ranks(cbm_global_test,cbm,X_train)

save_data_csv(settings.PATH.cbm_global_test,cbm_global_test)