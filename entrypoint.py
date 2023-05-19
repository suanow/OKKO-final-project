from i2iu2u.entrypoint import get_user_pred_i2iu2u
from item2item.entrypoint import get_pred_i2i
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('func', type=int)
parser.add_argument('user_id', type=int)
parser.add_argument('top_n', type=int)
args = parser.parse_args()

if args.func == 1:
    get_user_pred_i2iu2u(args.user_id,args.top_n)
elif args.func == 2:
    get_pred_i2i(args.user_id,args.top_n)