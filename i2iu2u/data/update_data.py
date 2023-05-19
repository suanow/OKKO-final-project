import numpy as np
import pandas as pd
from conf.conf import logging
import warnings
warnings.filterwarnings('ignore')


def get_rating(movies_md: pd.DataFrame,interactions: pd.DataFrame)-> pd.DataFrame:
    """
    Calculates how long a certain user have watched the film
    """
    
    logging.info('Calculating how long a user wacthed a certain film')
    
    interactions = interactions.merge(movies_md,how='left',on='movie_id')
    interactions['watched'] = interactions['watch_duration_minutes']/interactions['duration']
    interactions['watched'] = np.clip(interactions['watched'],0,1)
    interactions = interactions.replace(np.nan,interactions['watched'].mean())
    interactions =  interactions[['year','month','day','user_id','movie_id','watch_duration_minutes','watched']]
    interactions = interactions.rename(columns={'watched':'rating'})
    logging.info('Calculation is done')
    
    return interactions

def get_last_watch_dt(df:pd.DataFrame,date_columns_list:list):
    for column in date_columns_list:
        df[column] = [str(x) for x in df[column]]
    
    df['last_watch_dt']  = df['day'] + '-' + df['month'] + '-' + df['year']
    df['last_watch_dt'] = pd.to_datetime(df['last_watch_dt'])
    
    return df