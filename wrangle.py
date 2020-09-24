import pandas as pd
import numpy as np
import scipy as sp 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

###################### Acquire Telco Churn Data ######################

def train_valid_test(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.total_charges)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.total_charges)
    return train, validate, test

def wrangle_telco(df):
    ''' 
    This function preforms 3 operations:
    1. Reads in the telco data from a csv file
    2. Changes total_charges to a numeric variable and replaces any NaN values with a 0
    3. Splits data in to train, validate, test and returns as dataframes     
    '''
    # Changes total_charges to numeric variable
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    return df
    # Splits dataframe in to train, validate, test
    train, validate, test = train_valid_test(df)
    return train, validate, test



