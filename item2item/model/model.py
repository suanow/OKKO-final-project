import pandas as pd
import numpy as np
import re
from conf.conf import logging


def get_highest_value(row:pd.DataFrame)->pd.DataFrame:
    """
    Accepts a DataFrame row, finds and returns 
    the best similarity between the values in a row
    """
    if row['similarity_md'] - row['similarity_desc'] <0.2:
        return max(row['similarity_md'], row['similarity_desc'])
    elif row['similarity_md'] ==0:
        return row['similarity_desc']
    else:
        return row['similarity_desc']
    
    
def predict(movies_md:pd.DataFrame,interactions:pd.DataFrame,cos_sim_md,cos_sim_desc,title:str,userid:int, similarity_weight=0.7, top_n=10)->pd.DataFrame:
    """
    Accepts all of the necessary info for making a prediction and returns a dataframe with best recomandations
    """
    
    logging.info('Making a prediction')
    
    data = movies_md.reset_index()
    index_movie = data[data['movie_id'] == title].index
    similarity_md = cos_sim_md[index_movie].T
    similarity_desc = cos_sim_desc[index_movie].T
    dur_coef = float(interactions.loc[(interactions['movie_id'] == title)&(interactions['user_id'] == userid)]['watched'])
    nan = float(np.nan)
    if dur_coef > 0 :
        dur = dur_coef
    else: 
        dur = 0.5
    sim_df_md = pd.DataFrame(similarity_md, columns=['similarity_md'])
    sim_df_desc = pd.DataFrame(similarity_desc, columns=['similarity_desc'])
    final_df = pd.concat([data, sim_df_md,sim_df_desc], axis=1)
    final_df['similarity'] = final_df.apply(get_highest_value,axis=1) 
    final_df['final_score'] = (final_df['score']*(1-similarity_weight) +   final_df['similarity'] *similarity_weight)*dur
    
    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    final_df_sorted = final_df_sorted.loc[final_df_sorted['similarity'] < 1]
    
    logging.info('Prediction is ready')
    
    return final_df_sorted[['title','score','similarity', 'similarity_md','similarity_desc','final_score']]

def get_movies_list(userID:int,df:pd.DataFrame)->list:
    """
    Accepts user id and interactions dataframe and returns a list of films watched by user
    """
    ml = df.loc[df['user_id']==userID]['movie_id'].to_list()
    return ml


def get_user_pred(movies_md:pd.DataFrame,interactions:pd.DataFrame,cos_sim_md,cos_sim_desc,int_coef:pd.DataFrame,user_id:int,top_n:int):
    """
     Accepts all of the necessary info for making a prediction and returns a dataframe with best recomandations
    """
    
    logging.info(f'Creating Predictions for user {user_id}')
    
    coef = float(int_coef.loc[int_coef.index == user_id]['interactions_coefficient'].values)
    z=pd.DataFrame()
    ml = set(get_movies_list(user_id,interactions))
    for id in ml:
        try:
            title = str(movies_md.loc[movies_md['movie_id']== id]['title'])
            title = title.split("\n")[0]
            title = re.sub(r'\d+', '', title)
            df = predict(movies_md,interactions,cos_sim_md,cos_sim_desc,id,user_id,similarity_weight=coef)
            df['based_on'] = title
            z = pd.concat([z,df])
            z = z.loc[z['similarity']<1]
            z = z.sort_values('final_score',ascending=False)
            if len(z)> top_n:
                z = z.head(top_n)
            elif len(z) <= top_n:
                z = z
        except:
            z = z
            
    logging.info('Prediction is ready')
    
    return z