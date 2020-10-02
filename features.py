import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

from evaluate import linear_model

# This is the code for the Linear Model
from statsmodels.formula.api import ols

###################### Feature Selection (Linear Models) ######################

def select_kbest(k, X_train_scaled, y_train):
    '''
    This function takes in k, x train and y train data, and produces a k list of features
    as well as a Data Frame for modeling
    '''
    # Defining KBest object
    f_selector = SelectKBest(f_regression, k=k)

    # Fitting data to model
    f_selector = f_selector.fit(X_train_scaled, y_train)

    # Transforming dataset
    X_train_reduced = f_selector.transform(X_train_scaled)
    print(X_train.shape)
    print(X_train_reduced.shape)

    # Making a boolean mask
    f_support = f_selector.get_support()

    # List of Columns to keep
    f_feature = X_train_scaled.iloc[:,f_support].columns.tolist()

    # Output list as DF for Modeling
    X_reduced_scaled = X_train_scaled.iloc[:,f_support]

    print(str(len(f_feature)), 'selected features')
    print(f_feature)
    return X_reduced_scaled



def select_rfe(k, X_train_scaled, y_train):
    '''
    This function takes in k, x train and y train data, and produces a k list of features
    as well as a Data Frame for modeling
    '''
    # Initalize regression object
    lm = LinearRegression()

    # Initialize the RFE object, setting the hyperparameters to be our linear model above (lm), and the number of features we want returned.
    rfe = RFE(lm, k)

    # Fit model to data
    X_rfe = rfe.fit_transform(X_train_scaled, y_train)
    print(X_train.shape)                        # The origional shape of the data frame
    print(X_rfe.shape)                      # The shape of the df after feature selection

    mask = rfe.support_                         # boolean mask of features and if they should be used

    X_reduced_scaled_rfe = X_train_scaled.iloc[:,mask] # returns df ready for modeling

    rfe_feature = X_reduced_scaled_rfe.columns.tolist()       # returns list of selected features
    print(str(len(rfe_feature)), 'selected features')
    print(rfe_feature) 
    return X_reduced_scaled_rfe