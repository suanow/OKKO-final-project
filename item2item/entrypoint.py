import pandas as pd
import numpy as np
import argparse
from model.model import predict,get_user_pred
from data.data import get_data_parquet,save_data_csv
from data.update_data import get_int_coef
from util.util import load_matrix

parser = argparse.ArgumentParser()
parser.add_argument('user_id', type=int)
parser.add_argument('top_n', type=int)
parser.add_argument('interactions_upd_link', type=str)
parser.add_argument('movies_md_upd_link', type=str)
parser.add_argument('int_coef_link', type=str)
parser.add_argument('cos_sim_md_link', type=str)
parser.add_argument('cos_sim_desc_link', type=str)
args = parser.parse_args()

'''
movies_md_upd_link = 'data/movies_md_upd.parquet'
interactions_upd_link = 'data/interactions_upd.parquet'
int_coef_link = 'data/int_coef.parquet'
cos_sim_md_link = 'matrix/cos_sim_md.pkl'
cos_sim_desc_link = 'matrix/cos_sim_desc.pkl'
'''


interactions = get_data_parquet(args.interactions_upd_link)
movies_md = get_data_parquet(args.movies_md_upd_link)
int_coef = get_data_parquet(args.int_coef_link)
cos_sim_md = load_matrix(args.cos_sim_md_link)
cos_sim_desc = load_matrix(args.cos_sim_desc_link)
pred = get_user_pred(movies_md,interactions,cos_sim_md,cos_sim_desc,int_coef,args.user_id,args.top_n)
print(pred)
pred_link = 'pred/pred('+str(args.user_id)+').csv'
save_data_csv(pred_link,pred)
