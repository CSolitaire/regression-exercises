import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


#################### Prepare ##################

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
    

def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

    
def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
        
    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                columns=numeric_cols).\
                                set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                    columns=numeric_cols).\
                                    set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                columns=numeric_cols).\
                                set_index([X_test.index.values])

        
    return X_train_scaled, X_validate_scaled, X_test_scaled


def wrangle_zillow(path):
    df = pd.read_csv(path)

    # New Feature (Ratio of bedroomcnt and bathroomcnt)
    df['bedbathratio'] = round(df['bedroomcnt'] / df['bathroomcnt'],2)

    # Rename columns for clarity
    df.rename(columns={"hashottuborspa":"hottub_spa","fireplacecnt":"fireplace","garagecarcnt":"garage"}, inplace = True)

    # Replaces NaN values with 0
    df['garage'] = df['garage'].replace(np.nan, 0)
    df['hottub_spa'] = df['hottub_spa'].replace(np.nan, 0)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].replace(np.nan, 0)
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    df['fireplace'] = df['fireplace'].replace(np.nan, 0)
    
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    df["zip"] = df["regionidzip"].astype('category')
    df["zip"] = df["zip"].cat.codes

    df.drop(columns= ['parcelid','id','airconditioningtypeid','architecturalstyletypeid','basementsqft','buildingclasstypeid','buildingqualitytypeid'], inplace = True)
    df.drop(columns= ['calculatedbathnbr','decktypeid','finishedfloor1squarefeet','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15'], inplace = True)
    df.drop(columns= ['finishedsquarefeet50','finishedsquarefeet6','fips','fullbathcnt','heatingorsystemtypeid','poolsizesum','pooltypeid10','pooltypeid2'], inplace = True)
    df.drop(columns= ['pooltypeid7','propertycountylandusecode','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood'], inplace = True)
    df.drop(columns= ['storytypeid','threequarterbathnbr','typeconstructiontypeid','unitcnt','yardbuildingsqft17','yardbuildingsqft26','numberofstories'], inplace = True)
    df.drop(columns= ['fireplaceflag','structuretaxvaluedollarcnt','assessmentyear','landtaxvaluedollarcnt','taxamount','taxdelinquencyflag','taxdelinquencyyear'], inplace = True)
    df.drop(columns= ['censustractandblock','logerror','transactiondate','garagetotalsqft'], inplace = True)

    # drop any nulls
    df = df.dropna()

    # get object column names
    object_cols = get_object_cols(df)
        
    # create dummy vars
    df = create_dummies(df, object_cols)
        
    # split data (taxvaluedollarcnt is target)
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'taxvaluedollarcnt')
        
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
        
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

  
def run():
    print("Model: Running models...")
    # Write code here
    print("Model: Completed!")