import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

###################### Scale Telco Churn Data ######################

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    '''
    This function takes in train, validate, test dataframes, a scaler, and a list of colums and returns
    a scaled datafram for train, validate, and test 
    '''
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)

    return train, validate, test

def scale_telco_data(train, validate, test):
    '''
    This function defines the paramters for add_scaled_columns
    Items to Modity:
      1. Scaler
      2. Columns_to_scale
    '''
    train, validate, test = add_scaled_columns(
        train,
        validate,
        test,
        scaler= MinMaxScaler(),
        columns_to_scale=['total_charges', 'monthly_charges', 'tenure'],
    )
    return train, validate, test

###################### Acquire Telco Churn Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = '''
                Select
                customer_id,
                monthly_charges,
                tenure,
                total_charges
                From customers
                Where contract_type_id = 3;
                '''
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco_churn.csv')
    return df

def get_telco_data(cached = False):
    '''
    This function reads in telco data from Codeup database if cached == False
    or if cached == True reads in telco df from a csv file, returns df
    '''
    if cached or os.path.isfile('telco_churn.csv') == False:
        df = new_telco_data()
    else:
        df = pd.read_csv('telco_churn.csv', index_col=0)
    return df

###################### Inverse Scale Telco Churn Data ######################
  

def scale_inverse(train, validate, test, scaler, columns_to_scale, columns_to_inverse):
    '''
    This function takes in scaled train, validate, test dataframes, a scaler, and a list of colums and returns
    a scaled datafram for train, validate, and test 
    '''

    new_column_names = [c + '_inverse' for c in columns_to_inverse]
    scaler.fit(train[columns_to_scale])
    train = pd.concat([
        train,
        pd.DataFrame(scaler.inverse_transform(train[columns_to_inverse]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.inverse_transform(validate[columns_to_inverse]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.inverse_transform(test[columns_to_inverse]), columns=new_column_names, index=test.index),
    ], axis=1)
    return train, validate, test

###################### Prepared Telco Churn Data ######################

def wrangle_telco():
    ''' 
    This function preforms 4 operations:
    1. Reads in the telco data from a csv file
    2. Changes total_charges to a numeric variable and replaces any NaN values with a 0 
    3. Splits prepared data in to train, validate, test  
    4. Scales and returns dataframes
    '''
    df = get_telco_data(cached = False)
    # Changes total_charges to numeric variable
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    # Split the data in to train, validate, test
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    # Inverse scale
    #train, validate, test = scale_inverse(train, validate, test)
    # Scales data and return scaled data dataframes
    return scale_telco_data(train, validate, test)
