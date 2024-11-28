import os
import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from prophet import Prophet
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
import gc
from sklearn.preprocessing import StandardScaler
# from memory_profiler import profile

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()
from service_modules.ancillary_functions import resample_fun, start_trim_fun
from service_modules.stationary_module import make_stationary_fun2
from service_modules.grid_search_prophet import param_summary, create_param_combinations, score_model
from external_modules.pca_module import pca_fun
from service_modules.json_converter import json_create
from service_modules.target_transform_module import target_transform_fun2


def prophet_fun(trimac_series, Branch, helper, test_length, test_readings, holidays_main_extended,
                parallel=False, **kwargs):

    # step 1: create model summary object
    model_summary = []
        
    # step 2: create supporting objects; update helper object to remove Box-Cox and Min-Max transformations  
    test_start = len(trimac_series) - test_length
    train = trimac_series.iloc[:test_start]
    test = trimac_series.iloc[test_start:]
            
    trimac_series_initial = trimac_series # used to reset trimac_series at step 5
    
    if holidays_main_extended is None:
        holidays_main_extended_copy = None
    else:
        holidays_main_extended_copy = holidays_main_extended.copy()
        
    max_steps = test_length
    
    # step 3: start main external df object (to be used later on) as None
    external_df_extended = None
    external_df_full = None

    # step 4: build list of macro / markets variables if indicated
    
    if 'GC_Matrix' in kwargs:
        GC_Matrix = kwargs['GC_Matrix']
        GC_Matrix = GC_Matrix[Branch]   
        GC_Matrix = GC_Matrix.reset_index()
        GC_Matrix = GC_Matrix[GC_Matrix[Branch] != 'NULL']   
        GC_Matrix = GC_Matrix.reset_index()   
        GC_Matrix = GC_Matrix.drop('index',1)
        
        var_list = []
        
        for i1 in range(0,len(GC_Matrix)):
            
            Title = GC_Matrix.iloc[i1,0]
            
            Gap, Holder, te_help = GC_Matrix.iloc[i1,1]
            
            for i2 in range(0,len(Holder)):
                
                Lag, pvalue = Holder[i2]
                
                Steps = Lag - Gap
                
                var_list.append([Title, Lag, pvalue, Steps, te_help])
                
        var_list = pd.DataFrame(var_list, columns=['Title','Lag','pvalue','Steps','Stationary Helper'])
        
        max_steps = max(var_list['Steps']) - (max(var_list['Steps']) % 5)
        
    # step 5: build n-1 FB Prophet Models, where n = len(models_map)
        
    models_map = [0,5,10,15,20,25,30,35,39]
    
    models_map_len1 = len(models_map)
    
    counter = 8
    while counter > 0:
        if models_map[counter] > max_steps:
            models_map = models_map[:-1]
            counter = counter - 1
        else:
            counter = 0
    
    models_map_len2 = len(models_map)
    
    add_rows = models_map_len1 - models_map_len2
            
    for i1 in range(0,len(models_map)-1):
        
        # reset initial version of trimac_series
        trimac_series = trimac_series_initial

        # set first step, last step and label
        first_step = models_map[i1] + 1
        last_step = models_map[i1 + 1]
        label = 'Steps ' + str(first_step) + ' - ' + str(last_step) + ': '
        
        # build exogenous variable dataframe if indicated
        
        if 'GC_Matrix' in kwargs:
        
            var_list_filt = var_list[var_list['Steps'] >= last_step]
            
            var_list_filt = var_list_filt.reset_index()
            
            var_list_filt = var_list_filt.drop('index', axis = 1)
            
            df_index = pd.date_range(start='2018-03-25', periods=(len(trimac_series) + last_step), freq='W')
            external_df_extended = pd.DataFrame(index=df_index)
    
            for i2 in range(0,len(var_list_filt)):
                
                # bring the relevant external series back in
                os.chdir(dir_string + '/Data/Trading Economics/External Data - C')
                te_series = pd.read_csv(var_list_filt.iloc[i2,0] + '.csv', header=0, index_col=None, squeeze=False)
                os.chdir(dir_string)
                te_series['Date'] = pd.to_datetime(te_series['Date'])
                te_series = te_series.set_index('Date')
                
                # resample the data to a weekly series
                te_series = resample_fun(te_series, var_list_filt.iloc[i2,0])
                te_series.index.freq = 'W'
                
                # make series stationary
                te_series, helper_te_series = make_stationary_fun2(te_series, var_list_filt.iloc[i2,4])
                
                # shift te_series given the lag suggestion from GC_Matrix
                te_series.index = te_series.index.shift(var_list_filt.iloc[i2,1])
                
                # run triming operation
                te_series = start_trim_fun(trimac_series, te_series)
                te_series = te_series.iloc[:(len(trimac_series) + last_step)]
                
                # update series name and add to external_df_extended
                te_series.name = str(var_list_filt.iloc[i2,0]) + ' - Lag ' + str(var_list_filt.iloc[i2,1])
                external_df_extended = pd.concat([external_df_extended, te_series], axis=1) 
            
            # create full version of external_df_extended needed for downstream operations
            external_df_full = external_df_extended.iloc[:-(last_step)]
        
        # add holidays if indicated
        holidays_main_extended = holidays_main_extended_copy.copy()                
        if holidays_main_extended is None:
            pass
        else:
            if 'GC_Matrix' in kwargs:
                # make same stationary transform as for target variable
                if helper[1] == '1 Step Differencing':
                    for i2 in range(0,len(holidays_main_extended.columns)):
                        holidays_main_extended.iloc[:,i2] = holidays_main_extended.iloc[:,i2].diff()
                elif helper[1] == '2 Step Differencing':
                    for i2 in range(0,len(holidays_main_extended.columns)):
                        holidays_main_extended.iloc[:,i2] = holidays_main_extended.iloc[:,i2].diff()
                        holidays_main_extended.iloc[:,i2] = holidays_main_extended.iloc[:,i2].diff()
                elif helper[1] == 'Log Transform':
                    for i2 in range(0,len(holidays_main_extended.columns)):
                        holidays_main_extended.iloc[:,i2] = np.log(holidays_main_extended.iloc[:,i2])
                # combine holiday an exogenous variables
                holidays_main_full = holidays_main_extended.iloc[:len(train) + len(test)]
                external_df_full = pd.concat([external_df_full, holidays_main_full], axis=1)
                holidays_main_extended_short = holidays_main_extended.iloc[:len(train) + len(test) + last_step]
                external_df_extended = pd.concat([external_df_extended, holidays_main_extended_short], axis=1)
            else:
                external_df_full = holidays_main_extended.iloc[:len(train) + len(test)]
                external_df_extended = holidays_main_extended.iloc[:len(train) + len(test) + last_step]
        
        # run grid search to select best model; trimac_series_trojan deals with joblib probem
        
        cfg_list, param_list = param_summary()
        cfg_list = create_param_combinations(**cfg_list)
        
        if external_df_full is None:
            trimac_series_trojan = pd.DataFrame(trimac_series)
        else:
            external_df_full_copy = external_df_full.copy()
            trimac_series_trojan = pd.concat([trimac_series, external_df_full_copy], axis=1)
            trimac_series_trojan = trimac_series_trojan.dropna()

        if parallel:
            scores = Parallel(n_jobs=cpu_count(),verbose=50)\
                (delayed(score_model)(trimac_series_trojan,
                                      helper,
                                      first_step,
                                      last_step,
                                      test_length,
                                      test_readings,
                                      param_list,
                                      cfg) for cfg in cfg_list)
        else:
            scores = [score_model(trimac_series_trojan,
                                  helper,
                                  first_step,
                                  last_step,
                                  test_length,
                                  test_readings,
                                  param_list,
                                  cfg) for cfg in cfg_list]   
         
        scores = pd.DataFrame(scores, columns = ['best_params','mape_result','rmse_result']) 
        scores['rmse_result'] = scores['rmse_result'].replace(np.inf, np.nan)
        scores = scores[scores['rmse_result'].notna()]
        scores = scores.sort_values(by='rmse_result', ascending=True).head(1)
        
        best_params = scores.iloc[0,0]
        mape_err = scores.iloc[0,1]
        rmse_err = scores.iloc[0,2]
        
        # use best hyperparameters to fit a final model
        
        cfg_dict = dict(zip(param_list,best_params))
        
        if external_df_full is None:
            
            trimac_series = trimac_series.reset_index()
        
            trimac_series.columns = ['ds','y']
            
            if best_params[0] == 'logistic':
                trimac_series['cap'] = max(trimac_series['y'])*1.5
                trimac_series['floor'] = min(trimac_series['y'])*1.5
            
            mod = Prophet(**cfg_dict, interval_width=0.95)
            
            model = mod.fit(trimac_series)
            
            model_final = json_create(model)
            
            helper2 = [None, None, [None, None], None]
               
        elif len(external_df_full.columns) < 3:
            
            trimac_series = trimac_series.reset_index()
        
            trimac_series.columns = ['ds','y']
            
            if best_params[0] == 'logistic':
                trimac_series['cap'] = max(trimac_series['y'])*1.5
                trimac_series['floor'] = min(trimac_series['y'])*1.5
            
            external_df_full = external_df_full.reset_index(drop=True)
            
            trimac_series = pd.concat([trimac_series, external_df_full], axis=1)
            
            mod = Prophet(**cfg_dict, interval_width=0.95)
            for i1 in range(0,len(external_df_full.columns)):
                mod.add_regressor(external_df_full.iloc[:,i1].name)
            
            model = mod.fit(trimac_series)
            
            model_final = json_create(model)
            
            helper2 = [None, None, [None, None], None]

        else:
            
            trimac_series, helper2 = target_transform_fun2(trimac_series, helper)
            
            trimac_series = trimac_series.reset_index()
            
            trimac_series.columns = ['ds','y']
            
            if best_params[0] == 'logistic':
                trimac_series['cap'] = max(trimac_series['y'])*1.5
                if helper2[2] == ('None' or 'Failed'):
                    trimac_series['floor'] = 1
                else:
                    trimac_series['floor'] = min(trimac_series['y'])*1.5
                        
            external_df_full = external_df_full.dropna()
            
            external_df_full = external_df_full.reset_index(drop=True)
            external_df_full, external_df_extended, pca_result = pca_fun(external_df_full,external_df_extended)
            feature_description = pca_result[1]
            
            trimac_series = pd.concat([trimac_series, external_df_full], axis=1)
            
            mod = Prophet(**cfg_dict, interval_width=0.95)
            for i1 in range(0,len(external_df_full.columns)):
                mod.add_regressor(external_df_full.iloc[:,i1].name)
            
            model = mod.fit(trimac_series)
            
            model_final = json_create(model)
        
        # update model_summary object
        
        if external_df_full is None:
            model_summary.append([label, first_step, last_step, 'FB Prophet',
                                  'Uni', best_params, 'n/a',
                                  mape_err, rmse_err,'n/a', 'n/a',
                                  'n/a', 'n/a', model_final, helper2])
        elif len(external_df_full.columns) < 3:
            model_summary.append([label, first_step, last_step, 'FB Prophet',
                                  'Uni', best_params, 'n/a',
                                  mape_err, rmse_err,'n/a', 'n/a',
                                  'n/a', external_df_extended, model_final, helper2])
        else:
            model_summary.append((label, first_step, last_step, 'FB Prophet',
                                  'Multi', best_params, feature_description,
                                  mape_err, rmse_err, 'n/a', 'n/a',
                                  'n/a', external_df_extended, model_final, helper2))
            
        # sweep memory
        del scores, trimac_series_trojan, model_final
        if 'GC_Matrix' in kwargs:
            external_df_extended = None
            external_df_full = None
            del te_series
        gc.collect()
        
    # step 6: convert model_summary object from a list to a dataframe
        
    model_summary = pd.DataFrame(model_summary, columns=['Label','First Step','Last Step','Model Type',
                                                          'Uni or Multi','Model_Parameters','Feature_Description',
                                                          'Mape_Error','RMSE_Error','Diagnostic_Details',
                                                          'Diagnostic_Overall','Bias_Correction',
                                                          'External-DataFrame','Model - Final', 'Helper'])
    
    # step 7: adding rows to model_summary where we don't have external data that goes out 'test_length' steps
    if add_rows > 0:
        for i1 in range(0,add_rows):
            first_step2 = models_map[-1] + 1
            if first_step2 == 36:
                last_step2 = models_map[-1] + 4
            else:
                last_step2 = models_map[-1] + 5
            label = 'Steps ' + str(first_step2) + ' - ' + str(last_step2) + ': '
            new_row = {'Label':label, 'First Step':first_step2, 'Last Step':last_step2, 'Model Type':'n/a', 'Uni or Multi':'n/a', 
                   'Model_Parameters':'n/a', 'Feature_Description':'n/a', 'Mape_Error':1000000, 'RMSE_Error':1000000,
                   'Diagnostic_Details':'n/a','Diagnostic_Overall':'n/a', 'Bias_Correction':'n/a',
                   'External-DataFrame':'n/a', 'Model - Final':'n/a', 'Helper':'n/a'}
            model_summary = model_summary.append(new_row,ignore_index=True)            
    
    return model_summary