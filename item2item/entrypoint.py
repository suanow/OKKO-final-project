import pandas as pd
import numpy as np
from item2item.model.model import get_user_pred
from item2item.data.data import get_data_parquet,save_data_csv
from item2item.util.util import load_matrix
from item2item.conf.conf import settings

def get_pred_i2i(user_id,top_n):
    interactions = get_data_parquet(settings.PATH.interactions_upd_link)
    movies_md = get_data_parquet(settings.PATH.movies_md_upd_link)
    int_coef = get_data_parquet(settings.PATH.int_coef_link)
    cos_sim_md = load_matrix(settings.PATH.cos_sim_md_link)
    cos_sim_desc = load_matrix(settings.PATH.cos_sim_desc_link)
    pred = get_user_pred(movies_md,interactions,cos_sim_md,cos_sim_desc,int_coef,user_id,top_n)
    print(pred)
    pred_link = 'pred/pred('+str(user_id)+').csv'
    save_data_csv(pred_link,pred)

