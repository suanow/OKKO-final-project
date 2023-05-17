import pandas as pd
from conf.conf import logging


def get_data() -> pd.DataFrame:
    """ 
    Getting table from csv 
    """
    logging.info('Extracting interactions')
    interactions = pd.read_parquet('interactions.parquet', engine='pyarrow')
    logging.info('Interactions is extracted')
    
    logging.info('Extracting movies metadata')
    movies_md = pd.read_parquet('movies_metdata.parquet',engine='pyarrow')
    logging.info('Metadata is extracted')
    
    logging.info('Extracting parsed ranks')
    movies_rd = pd.read_excel('movies_rd.xlsx')
    logging.info('Parsed ranks are extracted')
    
    return interactions, movies_md, movies_rd
