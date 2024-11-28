import os
import numpy as np
import pandas as pd
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statistics import mean
from math import sqrt, floor
from sklearn.metrics import mean_squared_error
import gc

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()
from service_modules.ancillary_functions import mean_absolute_percentage_error
from external_modules.pca_module import pca_fun
from service_modules.predictions_module import predictions_insample
from service_modules.bias_correction_module import bias_correction_fun
from service_modules.diagnostics_module import diagnostics_fun
from service_modules.target_transform_module import target_transform_fun2
from service_modules.target_untransform_module import target_untransform_fun


# create a set of sarima configs to try
def sarima_configs():
    models = list()
    # define config lists
    p_params = [1]
    d_params = [0]
    q_params = [2]
    t_params = ['n']
    P_params = [0]
    D_params = [0]
    Q_params = [0]
    m_params =[52]
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    # models = models[1:]
    return models

# N-step sarima forecast
def sarima_forecast(history, helper, first_step, last_step, cfg, **kwargs):
    
    # get ready: (1) extract hyperparameters and (2) transform 'history' object
    order, sorder, trend = cfg
    history, helper2 = target_transform_fun2(history, helper)
    history = history.to_numpy()
    
    # build model and make predictions
    if kwargs['ext_df_full'] is None:    
        
        model = SARIMAX(history,
                        order=order,
                        seasonal_order=sorder,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False, method = 'powell')
        
        yhat = predictions_insample('SARIMAX',
                                    model_fit,
                                    first_step,
                                    last_step,
                                    history,
                                    **kwargs)
        
        yhat = pd.Series(yhat) 
        yhat = target_untransform_fun(yhat, helper2)
        
    else:
        
        history_exog = kwargs['history_exog'].reset_index(drop=True)
        future_exog_mod = kwargs['future_exog'].reset_index(drop=True)
            
        # trimming history_exog to reflect differencing treatment given to history object
        while np.size(history,0) < len(history_exog):
            history_exog = history_exog.iloc[1:,:].reset_index(drop=True)
        
        history_exog, future_exog_mod, pca_result = pca_fun(history_exog,future_exog_mod)         
    
        history_exog = history_exog.to_numpy()
        future_exog_mod = future_exog_mod.to_numpy()
        
        model = SARIMAX(history,
                        order=order,
                        seasonal_order=sorder,
                        trend=trend,
                        exog = history_exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False, method = 'powell')
        
        yhat = predictions_insample('SARIMAX',
                                    model_fit,
                                    first_step,
                                    last_step,
                                    history,
                                    future_exog_mod = future_exog_mod,
                                    **kwargs)
        
        yhat = pd.Series(yhat) 
        yhat = target_untransform_fun(yhat, helper2)
        
    return yhat

# split a univariate dataset into train/test sets
def train_test_split_fun(data, test_length):
    return data[:-test_length], data[-test_length:]

# walk-forward validation
def walk_forward_validation_fun(trimac_series, helper, first_step, last_step, test_length, test_readings, cfg, **kwargs):
    
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
        future_exog = X_test[0:last_step]  
    # step over each time-step in the test set
    possible_readings = test_length - last_step + 1
    block = max(floor((possible_readings)/test_readings),1)
    for i1 in range(0,min(possible_readings,test_readings)*block,block):
        try:
            # fit model and make forecast for history; convert the resut to a list
            if kwargs['ext_df_full'] is None:
                yhat = sarima_forecast(history, helper, first_step, last_step, cfg, **kwargs)
            else:
                yhat = sarima_forecast(history, helper, first_step, last_step, cfg,
                                       history_exog = history_exog, future_exog = future_exog, **kwargs)              
            yhat = yhat[first_step - 1:last_step]  
            test_segment = test[first_step + i1 - 1:last_step + i1]
            # calculate MAPE and RMSE 
            mape_err = mean_absolute_percentage_error(yhat,test_segment)
            mape_list.append(mape_err)
            rmse_err = sqrt(mean_squared_error(yhat,test_segment))
            rmse_list.append(rmse_err)
        except:
            pass
        # add actual observation to history for the next loop
        history = pd.concat([train, test[0:i1 + block]], axis=0)
        if kwargs['ext_df_full'] is not None:
            history_exog = pd.concat([X_train, X_test[0:i1 + block]], axis=0)
            future_exog = X_test[i1 + block:i1 + block + last_step]
    # estimate prediction error
    mape_result = mean(mape_list)
    rmse_result = mean(rmse_list)
    
    return mape_result, rmse_result

# run TS CV on a single set of hyperparameters
def score_model(trimac_series_trojan, helper, first_step, last_step, test_length, test_readings,
                diag_sig_level, cfg, debug=False):
    
    # set initial parameters
    mape_result = None
    rmse_result = None
    
    if len(trimac_series_trojan.columns) < 2:
        # single variable case
        trimac_series = pd.Series(trimac_series_trojan.iloc[:,0])
        ext_df_full = None        
        if debug:
            mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper,
                                                                   first_step, last_step,
                                                                   test_length, test_readings,
                                                                   cfg, ext_df_full = ext_df_full)
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper,
                                                                           first_step, last_step,
                                                                           test_length, test_readings,
                                                                           cfg, ext_df_full = ext_df_full)
            except:
                pass         
    else: 
        # multiple variable case
        trimac_series = pd.Series(trimac_series_trojan.iloc[:,0])
        ext_df_full = trimac_series_trojan.iloc[:,1:]
        if debug:
            mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper,
                                                                   first_step, last_step,
                                                                   test_length, test_readings,
                                                                   cfg, ext_df_full = ext_df_full)
        else:
            try:
                with catch_warnings():
                    filterwarnings("ignore")
                    mape_result, rmse_result = walk_forward_validation_fun(trimac_series, helper,
                                                                           first_step, last_step,
                                                                           test_length, test_readings,
                                                                           cfg, ext_df_full = ext_df_full)
            except:
                pass
    
    # retrain model on full data set and run diagnostics
    order, sorder, trend = cfg    
    trimac_series, helper2 = target_transform_fun2(trimac_series, helper)
    
    if ext_df_full is None:
                
        model = SARIMAX(trimac_series,
                        order=order,
                        seasonal_order=sorder,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
    
        model_fit = model.fit(disp=False, method = 'powell')
    
        residuals_bias_corrected, cutoff = bias_correction_fun(trimac_series, model_fit,'corrected_residuals')
        diagnostics_details, diagnostics_result = diagnostics_fun(trimac_series, residuals_bias_corrected,
                                              cutoff, model, diag_sig_level)
        bias_correction, cutoff = bias_correction_fun(trimac_series, model_fit, 'bias_adjustment')
        pca_result = [None, None]
            
    else:
                
        # trimming history_exog to reflect differencing treatment given to history object
        while len(trimac_series) < len(ext_df_full):
            ext_df_full = ext_df_full.iloc[1:,:]
          
        ext_df_full.index = trimac_series.index
        
        ext_df_full, pca_result = pca_fun(ext_df_full)
        
        model = SARIMAX(trimac_series,
                        order=order,
                        seasonal_order=sorder,
                        exog = ext_df_full,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
    
        model_fit = model.fit(disp=False, method = 'powell')
        
        residuals_bias_corrected, cutoff = bias_correction_fun(trimac_series, model_fit, 'corrected_residuals')
        diagnostics_details, diagnostics_result = diagnostics_fun(trimac_series, residuals_bias_corrected,
                                                                  cutoff, model, diag_sig_level,
                                                                  exog_1 = ext_df_full)
        bias_correction, cutoff = bias_correction_fun(trimac_series, model_fit, 'bias_adjustment')
          
    # save model in Model_Holder folder and run memory sweep
    name = 'mod - '
    for i1 in range (0,len(order)):
        name = name + str(order[i1]) 
    for i1 in range (0,len(sorder)):
        name = name + str(sorder[i1])   
    for i1 in range (0,len(trend)):
        name = name + str(trend[i1])
    name = name + '.pickle'
    os.chdir(dir_string + '/Data/Temp_Object_Holder/Model_Holder')
    model_fit.save(name)
    os.chdir(dir_string)

    del model_fit
    gc.collect()

    return ([order, sorder, trend], mape_result, rmse_result, bias_correction, diagnostics_details,
            diagnostics_result, name, helper2, pca_result)

