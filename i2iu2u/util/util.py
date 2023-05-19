import pickle
from conf.conf import logging,settings
from data.data import get_data_csv

def save_model(dir: str, model) -> None:
    """
    Saving model to .pkl file
    """
    pickle.dump(model, open(dir, 'wb'))
    logging.info('Model saved')


def load_model(dir: str) -> None:
    """
    Loading model from .pkl file
    """
    logging.info('Loading model')
    return pickle.load(open(dir, 'rb'))

def get_user_pred(user_id,top_k,pred_link='i2iu2u/data/cbm_global_test.csv'):
    cbm_global_test = get_data_csv(pred_link)
    preds = cbm_global_test.loc[cbm_global_test['user_id']== user_id][:top_k]
    preds = preds[['user_id','movie_id','rank']]
    print(preds)
    return preds
