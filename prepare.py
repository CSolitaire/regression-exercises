import pandas as pd
import numpy as np
import scipy as sp 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

###################### Split Telco Churn Data ######################

def train_valid_test(df):
    '''
    This function takes in a prepared dataframe and splits that data in to train, valid, split
    ** NOTE:**  No stratify in regression, only in used in classificaiton on labeled data
    '''
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test

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
    return scaler, train, validate, test

##################### Inverse Scale Telco Churn Data ######################
  

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

