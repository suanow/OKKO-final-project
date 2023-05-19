import pickle
<<<<<<< HEAD
from item2item.conf.conf import logging
=======
from conf.conf import logging
>>>>>>> 9596f3448f3d2d7f6ca4650b3c1f4dc77fd4902f

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