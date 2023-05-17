import pandas as pd
import numpy as np
from conf.conf import logging,settings
from conf.conf import dataset
from tqdm import tqdm

def compute_popularity(df: pd.DataFrame, item_id: str, max_candidates: int):
    """
    calculates mean rating to define popular titles
    """
    popular_titles = df.groupby(item_id).agg({'rating': np.mean})\
                     .sort_values(['rating'], ascending=False).head(max_candidates).index.values

    return popular_titles

def get_lightfm_mapping(dataset,df:pd.DataFrame):
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

    return dataset,lightfm_mapping

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

def model_fit_patial(model,train_mat,EPOCHS = settings.PARAMS.EPOCHS):
    for _ in tqdm(range(EPOCHS), total = EPOCHS):
            model.fit_partial(
            train_mat,
            num_threads = 4
        )
            
    return model