import pandas as pd
from model.model import compute_popularity, get_lightfm_mapping, get_item_name_mapper, df_to_tuple_iterator, get_train_mat_and_weights, model_fit_partial, predict
from conf.conf import logging, settings
from data.data import get_data_parquet, get_data_xslx, save_data_parquet, save_data_csv
import argparse

parser = argparse.ArgumentParser("Getting model params")

parser.add_argument('--values', 
                    nargs='+', 
                    type=float, 
                    required=True, 
                    help='Provide input values')

parser.add_argument('--model_path', 
                    type=str, 
                    help='Provide path to .pkl file with model')

args = parser.parse_args()

# values = args.values
# default_model_path = settings.MODEL.ranf_conf
# model_path = args.model_path if args.model_path else default_model_path

print(predict(values, model_path))