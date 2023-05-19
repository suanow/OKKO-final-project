import pickle
from item2item.conf.conf import logging

def save_matrix(dir:str,matrix)->None:
    """
    Saving Model
    """
    logging.info('Saving Matrix')
    pickle.dump(matrix,open(dir,'wb'))
    
def load_matrix(dir:str)->any:
    """
    Loading Model
    """
    logging.info('Loading Matrix')
    matrix = pickle.load(open(dir,'rb'))
    return matrix