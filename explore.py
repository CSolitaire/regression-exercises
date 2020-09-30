import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

###################### Explore Telco Data ######################
def plot_variable_pairs(df):
    '''
    Function that plots all of the pairwise relationships along with the regression line for each pair
    '''
    g = sns.PairGrid(df)    
    g.map_diag(sns.distplot)  
    g.map_offdiag(sns.regplot)
    plt.show()


def plot_variable_pairs_alt(df):
    '''
    Function that plots all of the pairwise relationships along with the regression line for each pair
    '''
    y = df.iloc[:,3]
    X = df.iloc[:,1]
    data = df
    return sns.regplot(X, y, ci = None)


def months_to_years(df):
    '''
    Function that returns your dataframe with a new feature tenure_years, in complete years as a customer
    '''
    df['tenure_years'] = round(df.tenure / 12, 0)
    return df


def months_to_years_alt(tenure_months, df):
    '''
    Function that returns your dataframe with a new feature tenure_years, in complete years as a customer
    '''
    df['tenure_years'] = round(tenure_months / 12, 0)
    return df


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    '''
    Function that outputs 3 different plots for plotting a categorical variable with a continuous variable
    '''
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13, 7))
    sns.boxplot(data = df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.violinplot(data = df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data = df, y=continuous_var, x=categorical_var)
    plt.show()