from parse.parse import parse_all_movies_rd
from data.data import get_data_parquet
from data.data import save_data_parquet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('APIkey', type=int)
parser.add_argument('interactions_link', type=str)
parser.add_argument('movies_md_link', type=str)
parser.add_argument('movies_rd_link', type=str)
args = parser.parse_args()

"""
movies_md_upd_link = 'data/movies_md_upd.parquet'
interactions_upd_link = 'data/interactions_upd.parquet'
movies_rd_link = 'data/movies_rd.parquet
'"""

interactions = get_data_parquet(args.interactions_link)
movies_md = get_data_parquet(args.movies_md_link)

movies_rd = parse_all_movies_rd(interactions,movies_md,args.APIkey)

save_data_parquet(args.movies_rd_link,movies_rd)

