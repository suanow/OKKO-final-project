import pickle
from conf.conf import logging

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