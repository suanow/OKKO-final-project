from data.data import get_data_parquet, save_data_parquet
from conf.conf import logging,settings


interactions = get_data_parquet(settings.PATH.interactions)
print(interactions)

movies_md = get_data_parquet(settings.PATH.movies_md)
print(movies_md)
