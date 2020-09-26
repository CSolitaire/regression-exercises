import pandas as pd
import numpy as np
import scipy as sp 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

###################### Split Telco Churn Data ######################

def train_valid_test(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123, stratify = df.total_charges)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123, stratify = train_validate.total_charges)
    return train, validate, test

###################### Scale Telco Churn Data ######################