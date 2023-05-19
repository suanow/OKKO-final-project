import pandas as pd
<<<<<<< HEAD
from item2item.conf.conf import logging
=======
from conf.conf import logging
>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f


def get_data_parquet(link:str) -> pd.DataFrame:
    """ 
    Getting table from parquet 
    """
    logging.info('Extracting DataFrame')
    df = pd.read_parquet(link, engine='pyarrow')
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
