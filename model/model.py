import pandas as pd
import numpy as np
from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler,RobustScaler

from connector.data import get_data
from conf.conf import logging, settings



def preprocess_movies_rd(movies_rd: pd.DataFrame) -> pd.DataFrame:
    """
    Makes preprocessing of parsed dataframe with ranking and description data for movies.
    Returns usable df for further caclulations
    """
    movies_rd['rating'] = [str(x) for x in movies_rd['rating'] .values]
    movies_rd['rating'] = [x.replace('dict_values([','') for x in movies_rd['rating'].values]
    movies_rd['rating'] = [x.replace('])','') for x in movies_rd['rating'].values]
    movies_rd['rating'] = [float(x) for x in movies_rd['rating'] .values]
    movies_rd['description'] = [str(x) for x in movies_rd['description'] .values]
    movies_rd = movies_rd.drop('Unnamed: 0', axis=1)
    movies_rd.columns = ['rating', 'description', 'title']
    
    return movies_rd


def get_int_coef(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the coefficient of interactions as number of interactions with the film
    """
    interactions['count'] = interactions['count']= [1 for x in interactions['user_id']]
    int_coef = interactions.groupby(['user_id']).sum()
    int_coef['interactions_coefficient'] = np.clip(int_coef['count'],1,20)
    int_coef['interactions_coefficient'] = np.log(int_coef['interactions_coefficient'])
    int_coef = int_coef.drop('count',axis=1)
    int_coef = int_coef[['interactions_coefficient','day']]
    int_coef = int_coef.drop('day',axis=1)
    scaler = MinMaxScaler()
    int_coef['interactions_coefficient'] = scaler.fit_transform(np.array(int_coef['interactions_coefficient']).reshape(-1,1))
    
    return int_coef


def get_popularity(interactions:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the popularity as number of users who watched the film
    """
    interactions['count']= [1 for x in interactions['movie_id']]
    popularity = interactions.groupby(['movie_id']).sum()
    popularity['num_users_watched'] = popularity['count']
    popularity = popularity.drop('count',axis=1)
    popularity = popularity[['watch_duration_minutes','num_users_watched']]
    
    return popularity


def get_avg(interactions:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average watch coefficient as time watched divided by length of the film
    """
    avg = interactions.groupby(['movie_id']).mean()
    avg = avg[['watch_duration_minutes','day']]
    avg['avg_watch_coef'] = avg['watch_duration_minutes']
    avg = avg.drop(['watch_duration_minutes','day'],axis=1)
    
    return avg


def get_movies_upd(movies_md:pd.DataFrame, popularity:pd.DataFrame, avg:pd.DataFrame) -> pd.DataFrame:
    """
    Merges the movies dataframe with the popularity and average watch coefficient dataframes
    """
    movies_md = movies_md.merge(popularity, how='right', on= 'movie_id')
    movies_md['time_watched'] = np.clip(movies_md['watch_duration_minutes']/movies_md['duration'],0,1)
    movies_md = movies_md.merge(avg, how='right', on= 'movie_id')
    
    return movies_md


def convert_columns(df:pd.DataFrame, columns_list:list) -> pd.DataFrame:
    """
    Deleted unnecessary characters from text
    """
    for column in columns_list:
        df[column] = [str(x) for x in df[column].values]
        df[column] = [x.replace('"','') for x in df[column].values]
        df[column] = [x.replace('[','') for x in df[column].values]
        df[column] = [x.replace(']','') for x in df[column].values]
        df[column] = [x.replace(',',' ') for x in df[column].values]
        
    return df


def get_desc(movies_upd:pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the description of the film to single text containing all the necessary information
    """
    movies_upd['desc_md'] = movies_upd['title'] + ' ' + movies_upd['entity_type'] + ' ' + movies_upd['genres'] + ' ' + movies_upd['director']
    movies_upd = movies_upd.dropna(subset=['desc_md'])
    
    return movies_upd


def get_cosine_mat(df:pd.DataFrame, desc:str) -> pd.DataFrame:
    """
    Calculates the cosine matrix by df and column name
    """
    tfidf = TfidfVectorizer(min_df=5)
    tfidf_matrix = tfidf.fit_transform(df[desc])
    cos_sim = cosine_similarity(tfidf_matrix)
    
    return cos_sim


def get_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns final score of the movie
    """
    df['rating'] = df['rating'].replace(-1,0)
    df['time_watched'] = df['time_watched'].replace(np.nan,0)
    scaler = MinMaxScaler()
    df['time_watched'] = scaler.fit_transform(np.array(df['time_watched']).reshape(-1,1))
    df['num_users_watched'] = scaler.fit_transform(np.array(df['num_users_watched']).reshape(-1,1))
    df['avg_watch_coef'] = scaler.fit_transform(np.array(df['avg_watch_coef']).reshape(-1,1))
    df['rating'] = scaler.fit_transform(np.array(df['rating']).reshape(-1,1))
    df['score'] = df['time_watched']*0.1 + df['num_users_watched']*0.4+df['avg_watch_coef']*0.1 + df['rating']*0.4
    #df['score'] = np.clip(df['score'],0,100)
    df['score'] = scaler.fit_transform(np.array(df['score']).reshape(-1,1))
    
    return df