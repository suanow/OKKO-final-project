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


# nakonets to ne ebanii functions
interactions = get_data_parquet('data/interactions.parquet')
movies_md = get_data_parquet('data/movies_metdata.parquet')

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

save_model('model/model.pkl',lfm_model)




