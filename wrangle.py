########## Functions used for CNY - BTC Price Movements ##########

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler() 

import statsmodels.api as sm


def prep_cny(df):
    
    '''
    This function takes in a dataframe and drops unneccessary columns, then renames
    the price column. From there the date column is converted to datetime and the clean df
    is returned.
    '''
    
    # dropping unneeded columns
    df.drop(columns = ['Open',
                  'High',
                  'Low',
                   'Change %'], inplace = True)
    
    #renameing the price column for simplistic exporation
    df.rename(columns={'Price':'cny_price'}, inplace=True)
    
    # converting the date column to datetime
    df.Date = pd.to_datetime(df.Date)
    
    return df


def prep_btc(df):
    
    '''
    This function takes in a dataframe and drops unneccessary columns, then renames
    the price column. From there the date column is converted to datetime and the clean df
    is returned. Two functions are needed as the btc df has different information than the cny df.
    '''
    
    # dropping columns
    df.drop(columns = ['Open',
                  'High',
                  'Low',
                    'Vol.',
                   'Change %'], inplace = True)
    
    # converting to datetime
    df.Date = pd.to_datetime(df.Date)
    
    # changing datatype to float from object.
    df['Price'].astype(float)
    
    return df


def scale_minmax(df):
    
    '''
    This function takes in a dataframe and calls in the minmax scaler, fits it,
    then transforms the df. From there it is turned to a dataframe with
    relabeled columns.
    '''
    
    # calling in the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    # fitting the scaler to the df
    scaler.fit(df)
    
    # transforming the df with the scaler
    df_scaled = scaler.transform(df)
    
    # creating a new dataframe with the scaled data
    df_s = pd.DataFrame(df_scaled)
    
    # assigning labels to the data
    df_s.columns=['BTC','CNY']
    
    return df_s


def evaluate(target_var):
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_and_eval(target_var):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()