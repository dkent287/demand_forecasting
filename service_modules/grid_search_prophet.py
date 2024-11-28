import os
import pandas as pd
from warnings import catch_warnings
from warnings import filterwarnings
from itertools import product
from prophet import Prophet
from statistics import mean
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import gc

from service_modules.ancillary_functions import mean_absolute_percentage_error
from external_modules.pca_module import pca_fun
from service_modules.predictions_module import predictions_insample
from service_modules.target_transform_module import target_transform_fun2
from service_modules.target_untransform_module import target_untransform_fun

def param_summary():
    cfg_list = {  
                'growth': ['linear', 'logistic'],
              }
        
    param_list = ['growth']
    return cfg_list, param_list

def create_param_combinations(**param_dict):
    param_iter = product(*param_dict.values())
    params =[]
    for param in param_iter:
        params.append(param) 
    params = pd.DataFrame(params, columns=list(param_dict.keys()))
    params = params.values.tolist()
    return params

# N-step prophet forecast
def prophet_forecast(history, helper, first_step, last_step, param_list, cfg, **kwargs):  
    
    # prepare objects needed for modelling
    cfg_dict = dict(zip(param_list,cfg))
    
    # build model and make predictions
    if kwargs['ext_df_full'] is None:           
                    
        history = history.reset_index()
        
        history.columns = ['ds','y']
        
        if cfg[0] == 'logistic':
            history['cap'] = max(history['y'])*1.5
            history['floor'] = 1
        
        model = Prophet(**cfg_dict, interval_width=0.95)
        
        model = model.fit(history)
        
        yhat = predictions_insample('FB Prophet', model, first_step, last_step, history,
                                    cfg = cfg, **kwargs)
            
    elif len(kwargs['ext_df_full'].columns) < 3:
                
        history = history.reset_index()
        
        history.columns = ['ds','y']
        
        if cfg[0] == 'logistic':
            history['cap'] = max(history['y'])*1.5
            history['floor'] = 1
        
        history_exog = kwargs['history_exog'].reset_index(drop=True)
        full_exog_final = kwargs['full_exog'].reset_index(drop=True)        

        history = pd.concat([history, history_exog], axis=1)
        
        model = Prophet(**cfg_dict, interval_width=0.95)
        for i1 in range(0,len(history_exog.columns)):
            model.add_regressor(history_exog.iloc[:,i1].name)
        
        model = model.fit(history)
                
        yhat = predictions_insample('FB Prophet', model, first_step, last_step, history,
                                    cfg = cfg, full_exog_final = full_exog_final, **kwargs)
              
    else:
        
        history, helper2 = target_transform_fun2(history, helper)
        
        history = history.reset_index()
        
        history.columns = ['ds','y']
        
        if cfg[0] == 'logistic':
            history['cap'] = max(history['y'])*1.5
            if helper[2] == ('None' or 'Failed'):
                history['floor'] = 1
            else:
                history['floor'] = min(history['y'])*1.5
        
        history_exog = kwargs['history_exog'].reset_index(drop=True)
        full_exog_final = kwargs['full_exog'].reset_index(drop=True)        
        
        # trimming history_exog to reflect differencing treatment given to history object
        while len(history) < len(history_exog):
            history_exog = history_exog.iloc[1:,:].reset_index(drop=True)
            full_exog_final = full_exog_final.iloc[1:,:].reset_index(drop=True)

        history_exog, full_exog_final, pca_result = pca_fun(history_exog,full_exog_final)         
        
        history = pd.concat([history, history_exog], axis=1)
        
        model = Prophet(**cfg_dict, interval_width=0.95)
        for i1 in range(0,len(history_exog.columns)):
            model.add_regressor(history_exog.iloc[:,i1].name)
        
        model = model.fit(history)
                
        yhat = predictions_insample('FB Prophet', model, first_step, last_step, history,
                                    cfg = cfg, helper2 = helper2, full_exog_final = full_exog_final, **kwargs)
        
        yhat = target_untransform_fun(yhat, helper2)
 
    return yhat

# split a univariate dataset into train/test sets
def train_test_split_fun(data, test_length):
    return data[:-test_length], data[-test_length:]

# walk-forward validation
def walk_forward_validation_fun(trimac_series, helper, first_step, last_step, test_length, test_readings,
                                param_list, cfg, **kwargs):
    mape_list = []
    rmse_list = []

    # split dataset
    train, test = train_test_split_fun(trimac_series, test_length)
    if kwargs['ext_df_full'] is not None:
        X_train, X_test = train_test_split_fun(kwargs['ext_df_full'], test_length)        
    # seed history with training dataset
    history = train
    if kwargs['ext_df_full'] is not None:
        history_exog = X_train
        full_exog = pd.concat([X_train, X_test[0:last_step]], axis=0)   
    # step over each time-step in the test set
    possible_readings = test_length - last_step + 1
    block = max(floor((possible_readings)/test_readings),1)
    for i1 in range(0,min(possible_readings,test_readings)*block,block):
        try:
            # fit model and make forecast for history; convert the resut to a list
            if kwargs['ext_df_full'] is None:
                yhat = prophet_forecast(history, helper, first_step, last_step, param_list, cfg, **kwargs)
            else:
                yhat = prophet_forecast(history, helper, first_step, last_step, param_list, cfg,
                                        history_exog = history_exog, full_exog = full_exog, **kwargs)              
            yhat = yhat[first_step - 1:last_step]  
            test_segment = test[first_step + i1 - 1:last_step + i1]
            # calculate MAPE and RMSE 
            mape_err = mean_absolute_percentage_error(yhat,test_segment)
            mape_list.append(mape_err)
            rmse_err = sqrt(mean_squared_error(yhat,test_segment))
            rmse_list.append(rmse_err)
        except:
            pass
        # add observations for the next loop
        history = pd.concat([train, test[0:i1 + block]], axis=0)
        if kwargs['ext_df_full'] is not None:
            history_exog = pd.concat([X_train, X_test[0:i1 + block]], axis=0)
            full_exog = pd.concat([X_train, X_test[0:i1 + block + last_step]], axis=0)   
    # estimate prediction error
    mape_result = mean(mape_list)
    rmse_result = mean(rmse_list)
    
    return mape_result, rmse_result

# run TS CV on a single set of hyperparameters
def score_model(trimac_series_trojan, helper, first_step, last_step, test_length, test_readings, param_list, cfg,
                debug=False):
    
    # set initial parameters
    mape_result = None
    rmse_result = None
    
    if len(trimac_series_trojan.columns) < 2: 
        # single variable case
        trimac_series = pd.Series(trimac_series_trojan.iloc[:,0])
        ext_df_full = None
        if debug:
            mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper, first_step, last_step,
                                             test_length, test_readings, param_list, cfg, ext_df_full = ext_df_full)
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper, first_step, last_step,
                                             test_length, test_readings, param_list, cfg, ext_df_full = ext_df_full)
            except:
                pass         
    else: 
        # multiple variable case
        trimac_series = pd.Series(trimac_series_trojan.iloc[:,0])
        ext_df_full = trimac_series_trojan.iloc[:,1:]
        if debug:
            mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper, first_step, last_step,
                                             test_length, test_readings, param_list, cfg, ext_df_full = ext_df_full)
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper, first_step, last_step,
                                             test_length, test_readings, param_list, cfg, ext_df_full = ext_df_full)
            except:
                pass
    
    return [cfg, mape_result, rmse_result]


'''
def param_summary():
    cfg_list = {  
                'changepoint_prior_scale': [0.005, 0.05, 0.5, 5],
                'changepoint_range': [0.8, 0.9],
                'growth': ['linear', 'logistic'],
                'seasonality_mode': ['multiplicative', 'additive'],
                'seasonality_prior_scale':[0.01, 0.1, 1, 10.0]
              }    
    param_list = ['changepoint_prior_scale', 'changepoint_range','growth',
                  'seasonality_mode','seasonality_prior_scale']
    return cfg_list, param_list
'''