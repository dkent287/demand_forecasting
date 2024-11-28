# Import Needed Libraries and Functions
import pandas as pd

from service_modules.predictions_module import predictions_outsample
from service_modules.json_converter import json_unwind
from service_modules.target_transform_module import target_transform_fun2
from service_modules.target_untransform_module import target_untransform_fun


def forecast_fun(models_master_best, Branch, trimac_series):
    
    forecast_holder2 = pd.DataFrame()
   
    for i1 in range(0,len(models_master_best)):
        
        forecast_holder1 = pd.DataFrame()
        
        # assemble forecast inputs
        helper = models_master_best.iloc[i1,14]
        trimac_series_trans, helper2 = target_transform_fun2(trimac_series, helper)
        model_type = models_master_best.iloc[i1,3]
        model = models_master_best.iloc[i1,13]
        if model_type == 'FB Prophet':
            model = json_unwind(model)
        model_parameters = models_master_best.iloc[i1,5]
        steps_start = int(models_master_best.iloc[i1,1])
        steps_end = int(models_master_best.iloc[i1,2])
        if type(models_master_best.iloc[i1,12]) == str:
            external_df_future = 'n/a'
        elif model_type == 'FB Prophet':
            external_df_future = models_master_best.iloc[i1,12]
            index_holder = external_df_future.iloc[-(steps_end):]
            forecast_index = index_holder.index
        elif model_type == 'SARIMAX':
            external_df_future = models_master_best.iloc[i1,12]
            external_df_future = external_df_future.iloc[-(steps_end):]
            forecast_index = external_df_future.index
       
        # generate forcast
        forecast_trans = predictions_outsample(model_type, Branch, model, model_parameters, steps_start,
                                                   steps_end, trimac_series_trans, helper2, external_df_future)
        
        # store column values; untransform the result; clean-up index and column headings
        col_names = list(forecast_trans.columns.values)
        
        for i2 in range(0,len(forecast_trans.columns)):
            forecast_untrans = target_untransform_fun(forecast_trans.iloc[:,i2], helper2)
            forecast_holder1 = pd.concat([forecast_holder1,forecast_untrans], axis=1)
        
        forecast_holder1.index = forecast_index
        forecast_holder1.columns = col_names
        
        # trim forcast given values for steps_start and steps_end and add to forecast_holder object     
        forecast_holder1 = forecast_holder1.iloc[steps_start-1:steps_end,:]
        forecast_holder2 = pd.concat([forecast_holder2, forecast_holder1], axis=0)

    forecast_holder2.index.name = 'index'

    return forecast_holder2