import pandas as pd
from i2iu2u.util.util import get_user_pred
import argparse
from i2iu2u.data.data import save_data_csv


def get_user_pred_i2iu2u(user_id,top_n):
    preds = get_user_pred(user_id, top_n)
    pred_link = 'pred/pred('+str(user_id)+').csv'
    save_data_csv(pred_link,preds)