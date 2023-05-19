import pandas as pd
from item2item.conf.conf import logging


def get_data_parquet(link:str) -> pd.DataFrame:
    """ 
    Getting table from parquet 
    """
    logging.info('Extracting DataFrame')
    df = pd.read_parquet(link, engine='pyarrow')
    logging.info('DataFrame is extracted')
    
    return df

def gget_data_csv(link:str) -> pd.DataFrame:
    """ 
    Getting table from xslx 
    """
    logging.info('Extracting DataFrame')
    df = pd.read_csv(link)
    logging.info('DataFrame is extracted')
    
    return df

def get_data_xslx(link:str) -> pd.DataFrame:
    """ 
    Getting table from xslx 
    """
    logging.info('Extracting DataFrame')
    df = pd.read_excel(link)
    logging.info('DataFrame is extracted')
    
    return df

def save_data_parquet(dir:str,df:pd.DataFrame) -> None:
    """ 
    Saving table to parquet
    """
    logging.info('Saving DataFrame')
    df.to_parquet(dir)
    logging.info('DataFrame is saved')
    
    return df


def save_data_csv(dir:str,df:pd.DataFrame) -> None:
    """ 
    Saving table to csv
    """
    logging.info('Saving DataFrame')
    df.to_csv(dir)
    logging.info('DataFrame is saved')
    
    return df