# Import Needed Libraries and Functions
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

#---------------------------------------------------------------------------------------------------------------#
# Insample Prediction Functions                                                                                 #
#---------------------------------------------------------------------------------------------------------------#


def p_sarimax_insample(model, first_step, last_step, trimac_series, **kwargs):
       
    if kwargs['ext_df_full'] is None:
        start = len(trimac_series)
        end = start + last_step - 1
        predictions = model.predict(start=start,
                                end=end,
                                dynamic=False,
                                typ='levels')   
    
    else:
        start = len(trimac_series)
        end = start + last_step - 1
        predictions = model.predict(start = start,
                                    end = end,
                                    exog = kwargs['future_exog_mod'],
                                    dynamic=False,
                                    typ='levels')
    
    return predictions


def p_prophet_insample(model, first_step, last_step, trimac_series, **kwargs):
    
    cfg =  kwargs['cfg']
    
    if kwargs['ext_df_full'] is None:
        
        predictions_df = model.make_future_dataframe(periods=last_step,freq='W')
        
        if cfg[0] == 'logistic':
            predictions_df['cap'] = max(trimac_series['y'])*1.5
            predictions_df['floor'] = 1
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series):]
        predictions = predictions[['ds','yhat']]
        predictions = predictions.set_index('ds')
        
    elif len(kwargs['ext_df_full'].columns) < 3:
                        
        predictions_df = model.make_future_dataframe(periods=last_step,freq='W')
        
        if cfg[0] == 'logistic':
            predictions_df['cap'] = max(trimac_series['y'])*1.5
            predictions_df['floor'] = 1
        
        full_exog_final = kwargs['full_exog_final'].reset_index(drop=True)
        
        predictions_df = pd.concat([predictions_df, full_exog_final], axis=1)
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series):]
        predictions = predictions[['ds','yhat']]
        predictions = predictions.set_index('ds')
        
    else:
        
        helper2 =  kwargs['helper2']
        
        predictions_df = model.make_future_dataframe(periods=last_step,freq='W')
        
        if cfg[0] == 'logistic':
            predictions_df['cap'] = max(trimac_series['y'])*1.5
            if helper2[2] == ('None' or 'Failed'):
                predictions_df['floor'] = 1
            else:
                predictions_df['floor'] = min(trimac_series['y'])*1.5
        
        full_exog_final = kwargs['full_exog_final'].reset_index(drop=True)
        
        predictions_df = pd.concat([predictions_df, full_exog_final], axis=1)
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series):]
        predictions = predictions[['ds','yhat']]
        predictions = predictions.set_index('ds')
    
    return predictions


def predictions_insample(model_type, model, first_step, last_step, trimac_series, **kwargs):
    return {
        'SARIMAX': lambda: p_sarimax_insample(model, first_step, last_step, trimac_series, **kwargs),        
        'FB Prophet': lambda: p_prophet_insample(model, first_step, last_step, trimac_series, **kwargs)
    }.get(model_type, lambda: 'Not a valid operation')()


#---------------------------------------------------------------------------------------------------------------#
# Out of Sample Prediction Functions                                                                            #
#---------------------------------------------------------------------------------------------------------------#


def p_sarimax_outsample(Branch, model, model_parameters, steps_start, steps_end,
                                               trimac_series_trans, helper, external_df_future):
    
    if type(external_df_future) == str:
        forecast_obj = model.get_forecast(steps = steps_end)
        forecast = forecast_obj.predicted_mean
        ci_95 = forecast_obj.conf_int(alpha=.05)
        predictions = pd.concat([forecast, ci_95], axis=1)
        predictions = predictions.rename(columns = {predictions.columns[0]: Branch + ': ' + 'Forecast'})    
        predictions = predictions.rename(columns = {predictions.columns[1]: Branch + ': ' + 'CI-Lower'})
        predictions = predictions.rename(columns = {predictions.columns[2]: Branch + ': ' + 'CI-Upper'})
    
    else:
        forecast_obj = model.get_forecast(steps = steps_end, exog=external_df_future)
        forecast = forecast_obj.predicted_mean
        ci_95 = forecast_obj.conf_int(alpha=.05)
        predictions = pd.concat([forecast, ci_95], axis=1)
        predictions = predictions.rename(columns = {predictions.columns[0]: Branch + ': ' + 'Forecast'})    
        predictions = predictions.rename(columns = {predictions.columns[1]: Branch + ': ' + 'CI-Lower'})
        predictions = predictions.rename(columns = {predictions.columns[2]: Branch + ': ' + 'CI-Upper'})
    
    return predictions


def p_prophet_outsample(Branch, model, model_parameters, steps_start,
                        steps_end, trimac_series_trans, helper, external_df_future):
        
    if type(external_df_future) == str:
        
        predictions_df = model.make_future_dataframe(periods=steps_end,freq='W')
        
        if model_parameters[0]== 'logistic':
            predictions_df['cap'] = max(trimac_series_trans)*1.5
            predictions_df['floor'] = 1
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series_trans):]
        predictions = predictions[['ds','yhat','yhat_lower','yhat_upper']]
        predictions = predictions.set_index('ds')
        predictions = predictions.rename(columns = {predictions.columns[0]: Branch + ': ' + 'Forecast'})    
        predictions = predictions.rename(columns = {predictions.columns[1]: Branch + ': ' + 'CI-Lower'})
        predictions = predictions.rename(columns = {predictions.columns[2]: Branch + ': ' + 'CI-Upper'})
        
    elif len(external_df_future.columns) < 3:
                
        predictions_df = model.make_future_dataframe(periods=steps_end,freq='W')
        
        if model_parameters[0] == 'logistic':
            predictions_df['cap'] = max(trimac_series_trans)*1.5
            predictions_df['floor'] = 1
        
        external_df_future = external_df_future.reset_index(drop=True)
        
        predictions_df = pd.concat([predictions_df, external_df_future], axis=1)
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series_trans):]
        predictions = predictions[['ds','yhat', 'yhat_lower','yhat_upper']]
        predictions = predictions.set_index('ds')
        predictions = predictions.rename(columns = {predictions.columns[0]: Branch + ': ' + 'Forecast'})    
        predictions = predictions.rename(columns = {predictions.columns[1]: Branch + ': ' + 'CI-Lower'})
        predictions = predictions.rename(columns = {predictions.columns[2]: Branch + ': ' + 'CI-Upper'})
        
    else:
                
        predictions_df = model.make_future_dataframe(periods=steps_end,freq='W')
        
        if model_parameters[0] == 'logistic':
            predictions_df['cap'] = max(trimac_series_trans)*1.5
            if helper[2] == ('None' or 'Failed'):
                predictions_df['floor'] = 1
            else:
                predictions_df['floor'] = min(trimac_series_trans)*1.5
        
        external_df_future = external_df_future.reset_index(drop=True)
        
        while len(predictions_df) < len(external_df_future):
            external_df_future = external_df_future.iloc[1:,:]
            external_df_future = external_df_future.reset_index(drop=True)
        
        predictions_df = pd.concat([predictions_df, external_df_future], axis=1)
        
        predictions = model.predict(predictions_df)
        predictions = predictions[len(trimac_series_trans):]
        predictions = predictions[['ds','yhat', 'yhat_lower','yhat_upper']]
        predictions = predictions.set_index('ds')
        predictions = predictions.rename(columns = {predictions.columns[0]: Branch + ': ' + 'Forecast'})    
        predictions = predictions.rename(columns = {predictions.columns[1]: Branch + ': ' + 'CI-Lower'})
        predictions = predictions.rename(columns = {predictions.columns[2]: Branch + ': ' + 'CI-Upper'})
      
    return predictions


def predictions_outsample(model_type, Branch, model, model_parameters, steps_start, steps_end,
                              trimac_series_trans, helper, external_df_future):
    return {
        'FB Prophet': lambda: p_prophet_outsample(Branch, model, model_parameters, steps_start, steps_end,
                                                  trimac_series_trans, helper, external_df_future),
        'SARIMAX': lambda: p_sarimax_outsample(Branch, model, model_parameters, steps_start, steps_end,
                                               trimac_series_trans, helper, external_df_future)
    }.get(model_type, lambda: 'Not a valid operation')()


