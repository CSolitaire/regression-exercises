import pandas as pd
import numpy as np
import os
from env import host, user, password

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