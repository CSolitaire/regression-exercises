import pandas as pd
import numpy as np
import scipy as sp 
import os
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


def standard_scaler(train, validate, test):
    '''
    This function scales data using a standard sclaler. 
    which scales to a standard normal distribution (mean = 0, sd = 1)
    '''
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    train[['monthly_charges','tenure','total_charges']] = scaler.fit_transform(train[['monthly_charges','tenure','total_charges']])
    validate[['monthly_charges','tenure','total_charges']] = scaler.transform(validate[['monthly_charges','tenure','total_charges']])
    test[['monthly_charges','tenure','total_charges']] = scaler.transform(test[['monthly_charges','tenure','total_charges']])
    return scaler, train, validate, test


def uniform_scaler(train, validate, test):
    '''
    This function scales data using a uniform sclaler. It smooths out unusual distributions
    and it spreads out the most frequent values and reduces the impact of (marginal) outliers 
    '''
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True)
    train[['monthly_charges','tenure','total_charges']] = scaler.fit_transform(train[['monthly_charges','tenure','total_charges']])
    validate[['monthly_charges','tenure','total_charges']] = scaler.transform(validate[['monthly_charges','tenure','total_charges']])
    test[['monthly_charges','tenure','total_charges']] = scaler.transform(test[['monthly_charges','tenure','total_charges']])
    return scaler, train, validate, test


def gaussian_scaler(train, validate, test):
    '''
    This function scales data using a gaussian sclaler. This uses either the Box-Cox(positive data) 
    or Yeo-Johnson(negative and positibe data) method to transform to resemble normal or standard normal distrubtion.
    '''
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
    train[['monthly_charges','tenure','total_charges']] = scaler.fit_transform(train[['monthly_charges','tenure','total_charges']])
    validate[['monthly_charges','tenure','total_charges']] = scaler.transform(validate[['monthly_charges','tenure','total_charges']])
    test[['monthly_charges','tenure','total_charges']] = scaler.transform(test[['monthly_charges','tenure','total_charges']])
    return scaler, train, validate, test


def min_max_scaler(train, validate, test):
    '''
    This function scales data using a min_max sclaler. This is a linear transformation since it is derived from a linear function.
    Values will lie between a given minimum and maximum value, such as 0 and 1. 
    '''
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[['monthly_charges','tenure','total_charges']] = scaler.fit_transform(train[['monthly_charges','tenure','total_charges']])
    validate[['monthly_charges','tenure','total_charges']] = scaler.transform(validate[['monthly_charges','tenure','total_charges']])
    test[['monthly_charges','tenure','total_charges']] = scaler.transform(test[['monthly_charges','tenure','total_charges']])
    return scaler, train, validate, test


def iqr_robust_scaler(train, validate, test):
    '''
    This function scales data using robust sclaler. For data with a lot of outliers, Using RobustScaler, the median is removed 
    (instead of mean) and data is scaled according to a quantile range (the IQR is default).
    '''
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True)
    train[['monthly_charges','tenure','total_charges']] = scaler.fit_transform(train[['monthly_charges','tenure','total_charges']])
    validate[['monthly_charges','tenure','total_charges']] = scaler.transform(validate[['monthly_charges','tenure','total_charges']])
    test[['monthly_charges','tenure','total_charges']] = scaler.transform(test[['monthly_charges','tenure','total_charges']])
    return scaler, train, validate, test

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



