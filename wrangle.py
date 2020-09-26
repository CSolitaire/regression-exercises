import pandas as pd
import numpy as np
import scipy as sp 
from sklearn.impute import SimpleImputer

###################### Wrangle Telco Churn Data ######################

def wrangle_telco(df):
    ''' 
    This function preforms 2 operations:
    1. Reads in the telco data from a csv file
    2. Changes total_charges to a numeric variable and replaces any NaN values with a 0    
    '''
    # Changes total_charges to numeric variable
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    # Replaces NaN values with 0 for new customers with no total_charges
    df["total_charges"].fillna(0, inplace = True) 
    return df



