from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from conf.conf import logging

def get_cosine_mat(df:pd.DataFrame, desc:str):
    """
    Calculates the cosine matrix by df and column name
    """
    
    logging.info('Creating Cosine Matrix')
    
    tfidf = TfidfVectorizer(min_df=5)
    tfidf_matrix = tfidf.fit_transform(df[desc])
    cos_sim = cosine_similarity(tfidf_matrix)
    
    logging.info('Cosine Matrix is Ready')
    
    return cos_sim