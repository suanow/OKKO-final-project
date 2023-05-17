import requests
from tqdm import tqdm
import pandas as pd 
from conf.conf import logging

def get_rating(name:str,APIkey:str)->pd.DataFrame:
    """
    Accepts name and API key of film and returns a dataframe with film name,rating and description
    """
    
    logging.info(f'Parsing data for {name}')
    
    headers = {'X-API-KEY': APIkey}
    
    response = requests.get('https://api.kinopoisk.dev/v1.3/movie', params={'selectFields':['rating.kp','description'],'name':name,'limit':1}, headers=headers)
    rating = response.json()
    try:
        if len(rating['docs']) != 0:
            rating = pd.DataFrame(rating['docs'])
            rating['rating'] = [dict(x).values() for x in rating['rating']]
            rating['name'] = name
            
            logging.info(f'Parsing data for {name} complete')
            
            return rating
        
        else:
            logging.info(f'Parsing data for {name} complete, but it contains errors')
            
            return pd.DataFrame({'rating':-1, 'description':' ', 'name':name}, index=[0])
        
    except:
        logging.info(f'Parsing data for {name} complete, but it contains errors')
        return pd.DataFrame({'rating':-1, 'description':' ', 'name':name}, index=[0])
    

def parse_all_movies_rd(interactions:pd.DataFrame,movies_md:pd.DataFrame,APIkey:str)->pd.DataFrame:
    """ 
    Accepts interactions df and movies metadata df and returns a df with rating and descriptions for films that appear in both df's
    """
    
    logging.info('Parsing started')
    interactions_new = interactions.join(movies_md[['movie_id', 'title']].set_index('movie_id'), how='left', on='movie_id')
    ratings_n_descrs = pd.DataFrame()
    for name in tqdm(interactions_new['title'].unique().tolist()):
        ratings_n_descrs = pd.concat([ratings_n_descrs, get_rating(name,APIkey)])
    
    logging.info('Parsing finished')
    return ratings_n_descrs