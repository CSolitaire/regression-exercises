import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# This is the code for the Linear Model
from statsmodels.formula.api import ols

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from math import sqrt
from sklearn.model_selection import train_test_split

###################### Feature Selection For Regression Models ######################

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


###################### Model ######################




###################### Evaluating Regression Models ######################

#linear_model(ols('tip ~ total_bill', train).fit(), train.tip, train.total_bill)

def eval_linear_model(linear_model, target, feature):
    '''
    This function evaluates linear models
    1. It compares model SSE to baseline SEE
      - If baseline SEE > Model - Program Ends
      - If Model SSE < Baseline - Program Continues
    2. It creates an evaulation matrix and calculates 
      - SSE
      - MSE
      - RMSE
    3. It evaluates model significance by caluculating
      - $r^2$
      - F-Test to return p-value (looking for value under 0.05)
    4. Plots residuals
    '''
    baseline = target.mean()                              
    model = linear_model           
    evaluate = pd.DataFrame()
    evaluate["x"] = feature                          
    
    # Our y is our dependent variable
    evaluate["y"] = target                                
    evaluate["baseline"] = target.mean()                  

    # y-hat is shorthand for "predicted y" values
    evaluate["yhat"] = model.predict()

    # Calculate the baseline residuals 
    evaluate["baseline_residual"] = evaluate.baseline - evaluate.y

    # Calculate the model residuals
    evaluate["model_residual"] = evaluate.yhat - evaluate.y
    
    # Calculate if the model beats the baseline
    baseline_sse = ((evaluate.baseline_residual**2).sum())
    model_sse = ((evaluate.model_residual**2).sum())

    if model_sse > baseline_sse:
        print("Our baseline is better than the model.")
    else:
        print("Our model beats the baseline")
        print("It makes sense to evaluate this model more deeply.")
        print("Baseline SSE", baseline_sse)
        print("Model SSE", model_sse)
        metrics = pd.DataFrame()

        # Sum the squares of the baseline errors
        model_sse = ((evaluate.model_residual**2).sum())

        # Take the average of the Sum of squared errors
        # mse = model_sse / len(evaluate)

        # Or we could calculate this using sklearn's mean_squared_error function
        mse = mean_squared_error(evaluate.y, evaluate.yhat)

        # Now we'll take the Square Root of the Sum of Errors
        # Taking the square root is nice because the units of the error 
        # will be in the same units as the target variable.
        rmse = sqrt(mse)

        print("SSE is", model_sse, " which is the sum sf squared errors")
        print("MSE is", mse, " which is the average squared error")
        print("RMSE is", rmse, " which is the square root of the MSE")

        # The model commented below is our model
        # model = ols('sales ~ flyers', df).fit()

        r2 = model.rsquared
        print('R-squared = ', round(r2,3))

        # F-Test for p value
        f_pval = model.f_pvalue
        print("p-value for model significance = ", f_pval)

        if f_pval < 0.05:
          print('Reject Null: The model bulit on indipendent variables explains the relationship, validates $r^2$')
        else:
          print('Accept Null: A model not built w/ the independent variables explains the relationship, ie Baseline')
        
        def plot_residuals(actual, predicted):
            residuals = actual - predicted
            plt.hlines(0, actual.min(), actual.max(), ls=':')
            plt.scatter(actual, residuals)
            plt.ylabel('residual ($y - \hat{y}$)')
            plt.xlabel('actual value ($y$)')
            plt.title('Actual vs Residual')
            return plt.gca()
        
        actual = evaluate.y 
        predicted = evaluate.yhat
        plot_residuals(evaluate.y, evaluate.yhat)



def run():
    print("Model: Running models...")
    # Write code here
    print("Model: Completed!")