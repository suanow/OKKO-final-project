import pandas as pd 
import numpy as np
from conf.conf import settings
from data.data import get_data_parquet,get_data_xslx,save_data_parquet
from util.util import save_matrix
from data.update_data import preprocess_movies_rd,get_int_coef,get_popularity,get_avg,get_movies_upd,convert_columns,get_desc,get_score,get_dur
from matrix.matrix import get_cosine_mat


interactions = get_data_parquet(settings.PATH.interactions_link)
movies_md = get_data_parquet(settings.PATH.movies_md_link)
movies_rd = get_data_xslx(settings.PATH.movies_rd_link)

movies_rd = preprocess_movies_rd(movies_rd)

save_data_parquet(settings.PATH.movies_rd_upd_link,movies_rd)

int_coef = get_int_coef(interactions)

save_data_parquet(settings.PATH.int_coef_link,int_coef)

populatiry = get_popularity(interactions)

avg = get_avg(interactions)

movies_upd = get_movies_upd(movies_md,populatiry,avg,movies_rd)

movies_upd = convert_columns(movies_upd,settings.VARIABLES.columns_list)

movies_upd = get_desc(movies_upd)

movies_upd = get_score(movies_upd)

interactions= get_dur(movies_upd,interactions)

save_data_parquet(settings.PATH.movies_md_upd_link,movies_upd)
save_data_parquet(settings.PATH.interactions_upd_link,interactions)

cos_sim_md = get_cosine_mat(movies_upd,'desc_md')
save_matrix(settings.PATH.cos_sim_md_link,cos_sim_md)


cos_sim_desc = get_cosine_mat(movies_upd,'description')
save_matrix(settings.PATH.cos_sim_desc_link,cos_sim_desc)



