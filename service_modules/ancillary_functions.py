import numpy as np
import pandas as pd
import requests
from urllib.parse import urlencode
from datetime import date, timedelta
import math
import os

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


# datacheck function to find problematic lat-long data
def datacheck(starting,zip,city_state):
    check = ((starting > 200) | (starting == 0)) or ((zip < 200) and (abs(starting - zip) > 5)) or ((city_state < 200) and (abs(city_state - zip) > 5))
    return check


# Google Maps Api Call Function
def extract_lat_lng(address_or_postalcode, country_code, data_type = 'json'):
    api_key = "AIzaSyA6vycgsRoAKQdgm2ZGK63itAvslQIK7qE"
    endpoint = f"https://maps.googleapis.com/maps/api/geocode/{data_type}"
    country = f"country:{country_code}"
    params = {"address": address_or_postalcode, "key": api_key, "components": country}
    url_params = urlencode(params)
    url = f"{endpoint}?{url_params}"
    r = requests.get(url)
    if r.status_code not in range(200, 299): 
        return {}
    latlng = {}
    try:
        latlng = r.json()['results'][0]['geometry']['location']
    except:
        pass
    return latlng.get("lat"), latlng.get("lng")


# MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    value = (np.mean(np.abs((y_true - y_pred)/(y_true)))*100)
    return value


# function to select the best set of models
def model_set_selector_best(models_master):            
    models_master_filter = models_master[models_master['Diagnostic_Overall'] != 'fail']
    models_master_filter = models_master_filter[['Label','RMSE_Error']]
    models_master_filter = models_master_filter.groupby(['Label']).min(['RMSE Error'])
    models_master_filter = models_master_filter.reset_index()
    models_master_filter = pd.merge(models_master_filter,models_master,
                                    how='left',on=['Label','RMSE_Error'])
    models_master = models_master_filter[['Label','First Step','Last Step','Model Type','Uni or Multi',
                                              'Model_Parameters','Feature_Description','Mape_Error','RMSE_Error',
                                              'Diagnostic_Details','Diagnostic_Overall', 'Bias_Correction',
                                              'External-DataFrame','Model - Final', 'Helper']]
    models_master_best = models_master.sort_values('First Step')
    models_master_best = models_master_best.reset_index(drop = True)

    return models_master_best


# function to select the best univariate
def model_set_selector_best_uni(models_master):

    models_master_filter = models_master[models_master['Diagnostic_Overall'] != 'fail']    
    models_master_filter = models_master_filter[models_master_filter['Uni or Multi'] != 'Multi']
    models_master_filter = models_master_filter[models_master_filter['Uni or Multi'] != 'n/a']

    sarimax_result = models_master_filter[models_master_filter['Model Type'] == 'SARIMAX'].loc[:,'RMSE_Error'].sum()
    prophet_result = models_master_filter[models_master_filter['Model Type'] == 'FB Prophet'].loc[:,'RMSE_Error'].sum()
    if sarimax_result < prophet_result and len(models_master_filter[models_master_filter['Model Type'] == 'SARIMAX']) == 8:
        filter_value = "SARIMAX"
    else:
        filter_value = "FB Prophet"

    models_master_filter = models_master_filter[models_master_filter['Model Type'] == filter_value]

    models_master_filter = models_master_filter.sort_values('First Step')

    models_master_bestuni = models_master_filter
    
    return models_master_bestuni


def resample_fun(series,series_name):
    
    # creating needed look-up tables
    days_in_month_leap = {1:31,2:29,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    days_in_month_reg = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    month_to_quarter = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
    days_in_quarter_leap = {1:91,2:91,3:92,4:92}
    days_in_quarter_reg = {1:90,2:91,3:92,4:92}
    days_in_year_leap = 366
    days_in_year_reg = 365
    
    # bring in Trading Economics database description
    os.chdir(dir_string + '/Data/Trading Economics')
    te_library_desc = pd.read_csv('te_library_desc_working.csv')
    os.chdir(dir_string)
    te_library_desc_frequency = te_library_desc[['Title','Frequency']]    
    te_library_desc_frequency = te_library_desc.set_index('Title')['Frequency'].to_dict()
    te_library_desc_stockflow = te_library_desc[['Title','Stock_or_Flow']]    
    te_library_desc_stockflow = te_library_desc.set_index('Title')['Stock_or_Flow'].to_dict()
    
    frequency = te_library_desc_frequency[series_name]
    stock_or_flow = te_library_desc_stockflow[series_name]
    
    # resample as a daily series
    upsampled = series.resample('D')
    daily_interpolated = upsampled.interpolate(method='cubic', order = 3)
    
    # for flow series, divide by number of days in frequency (366 for an annual series in leap year)
    if stock_or_flow == 'Flow':
        
        # adding 3 helper columns - month, quarter, is_leap_year?
        daily_interpolated = daily_interpolated.reset_index()
        daily_interpolated['month'] = daily_interpolated['Date'].dt.month
        daily_interpolated['quarter'] = ''
        for i1 in range (0,len(daily_interpolated)):
            daily_interpolated.iloc[i1,3] = month_to_quarter[daily_interpolated.iloc[i1,2]]
        daily_interpolated['is_leap_year'] = daily_interpolated['Date'].dt.is_leap_year
        daily_interpolated = daily_interpolated.set_index('Date')

        # computing the indicated daily value based on the number of days in the relevant frequency      
        daily_interpolated['updated'] = ''
        for i1 in range (0,len(daily_interpolated)):
            leap_year = daily_interpolated.iloc[i1,3]
            if frequency == 'Yearly' and leap_year:
                days_in_period = 366
            elif frequency == 'Yearly' and not leap_year:
                days_in_period = 365
            elif frequency == 'Quarterly' and leap_year:
                days_in_period = days_in_quarter_leap[daily_interpolated.iloc[i1,2]]
            elif frequency == 'Quarterly' and not leap_year:
                days_in_period = days_in_quarter_reg[daily_interpolated.iloc[i1,2]]
            elif frequency == 'Monthly' and leap_year:
                days_in_period = days_in_month_leap[daily_interpolated.iloc[i1,1]]
            elif frequency == 'Monthly' and not leap_year:
                days_in_period = days_in_month_reg[daily_interpolated.iloc[i1,1]]
            elif frequency == 'Weekly':
                days_in_period = 7
            daily_interpolated.iloc[i1,4] = daily_interpolated.iloc[i1,0] / days_in_period
        daily_interpolated = daily_interpolated['updated'].rename('Value')
              
    # resample as a weeky series 
    if stock_or_flow == 'Flow':
        series = daily_interpolated.resample('W-SUN', label='left', closed='left').sum()[1:]
    elif stock_or_flow == 'Stock':
        series = daily_interpolated.resample('W-SUN', label='left', closed='left').mean()[1:]
        
    return series


# function to round down decimmals
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


# run trimming operation
def start_trim_fun(series_a, series_b):            
    
    series_a = series_a.reset_index()    
    series_b = series_b.reset_index()        
    
    diff = series_a.iloc[0,0] - series_b.iloc[0,0]
        
    if diff.days > 0:
        series_b = series_b[series_b['Date'] >= series_a.iloc[0,0]]
        series_b = series_b.set_index('Date')
        series_b = pd.Series(series_b.iloc[:,0])
    
    return series_b
    

# function to prepare strings needed for PC Miler function
def string_prep(val):
    val = val.astype(str)
    for i in range(0,len(val)):
       if val[i][0] == '-':
           val[i] = val[i][1:]
    val = val.str[:11]
    val = val.str.pad(width=11, side='right', fillchar='0')
    for i in range(0,len(val)):
       if 'Lat' in val.name:
           val[i] = val[i] + 'N'
       else:
           val[i] = val[i] + 'W'
    return val


# run trimming operation
def trim_fun(series_a, series_b):            
    
    series_a = series_a.reset_index()
    series_b = series_b.reset_index()        
    
    # run trimming operation at beginning of time series
    
    diff = series_a.iloc[0,0] - series_b.iloc[0,0]     
    if diff.days < 0:
        series_a = series_a[series_a['index'] >= series_b.iloc[0,0]]
    else:
        series_b = series_b[series_b['Date'] >= series_a.iloc[0,0]]
    
    # run trimming operation at end of time series
    
    diff = series_a.iloc[-1,0] - series_b.iloc[-1,0]      
    if diff.days > 0:
        series_a = series_a[series_a['index'] <= series_b.iloc[-1,0]]
    else:
        series_b = series_b[series_b['Date'] <= series_a.iloc[-1,0]] 
    series_b = series_b.set_index('Date')
    series_a = series_a.set_index('index')
    
    return series_a, series_b


