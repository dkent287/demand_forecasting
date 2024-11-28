'''
NOTES:
    
Improvements over previous work:

- creates different features for Canada vs US
- creates features specific to a state / province
- makes sure to use the "day observed' to assign holiday to the propoer week
- creates separate features for all holidays and xmas holidays
- creates the correct extended versions of the feature series we can use for the various forecasts
    
'''

import warnings
warnings.filterwarnings("ignore")
import holidays
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import *

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


def holidays_fun(trimac_series, holiday_type, test_length):
     
    # collect initial terminal information (country and state) to be used to get correct holidays
    
    os.chdir(dir_string + '/Data/Trimac Terminals')
    terminals_input = pd.read_csv('terminal_input.csv', header=0, index_col=None)
    os.chdir(dir_string)
    
    Branch = trimac_series.name
    
    terminals_input_filter = terminals_input[terminals_input['Terminal'] == Branch].reset_index()
    
    state_prov = terminals_input_filter.loc[0,'State']
    country = terminals_input_filter.loc[0,'Country']
     
    # extend the trimac_series object to cover dates for which we need to forecast
        
    trimac_series_ext = np.arange(0,test_length)
    trimac_series_ext = pd.Series(trimac_series_ext)
    
    rng = pd.date_range(start = trimac_series.index[-1] + pd.Timedelta('14 day'), periods=test_length, freq='W') + pd.Timedelta('-7 day')
    
    trimac_series_ext = trimac_series_ext.reindex(rng)
    
    trimac_series = pd.concat([trimac_series, trimac_series_ext])
    
    trimac_series.index = trimac_series.index.rename('index')
    
    trimac_series.name = Branch
    
    # preparing initial data from Holidays library
    
    holidays_df = pd.DataFrame([], columns=['date','holiday'])
    dates = []
    names = []
    
    if country == 'US':
        for date, name in sorted(holidays.CountryHoliday('US', prov=None, state=state_prov, years=np.arange(2016, 2023 + 1)).items()):
            dates.append(date)
            names.append(name)
    
    if country == 'CAN':
        for date, name in sorted(holidays.CountryHoliday('CAN', prov=state_prov, state=None, years=np.arange(2016, 2023 + 1)).items()):
            dates.append(date)
            names.append(name)
    
    holidays_df.loc[:,'date'] = dates
    holidays_df.loc[:,'holiday'] = names
            
    holidays_df.loc[:,'holiday_trim'] = holidays_df.loc[:,'holiday']
    holidays_df.loc[:,'holiday_trim'] = holidays_df.loc[:,'holiday_trim'].apply(lambda x : x.replace(' (Observed)',''))
    holidays_df.loc[:,'check'] = ''
    
    # getting rid of holidays falling on Sat or Sun that are "Observed" on other days - we keep only the "Observed"
    
    for i1 in range(0,len(holidays_df)-1):
        if holidays_df.iloc[i1,2][0:5] == holidays_df.iloc[i1+1,2][0:5]:
            if 'Observed' not in holidays_df.iloc[i1,1]:
                holidays_df.iloc[i1,3] = 'x'
            if 'Observed' not in holidays_df.iloc[i1+1,1]:
                holidays_df.iloc[i1+1,3] = 'x'
                
    holidays_df = holidays_df[holidays_df['check'] != 'x']
    
    holidays_df = holidays_df.drop('check', axis = 1)
    
    holidays_df = holidays_df.drop('holiday_trim', axis = 1)
    
    # attend clean-up and formatting
    
    holidays_df.loc[:,'holiday'] = holidays_df.loc[:,'holiday'].apply(lambda x : x.replace(' (Observed)',''))
    
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    holidays_df['week_starting'] = holidays_df['date'].where(holidays_df['date'] == ((holidays_df['date'] + Week(weekday=6)) - Week()), holidays_df['date'] - Week(weekday=6))
    
    holidays_df = holidays_df.rename(columns={"week_starting": "index"})
    
    holidays_df = holidays_df.set_index('index')
    
    holidays_df = holidays_df.drop('date', axis = 1)
    
    holidays_df.loc[:,'holidays_dummy'] = int(1)
    
    holidays_df = holidays_df.groupby(holidays_df.index).first() # getting rid of duplicated index values

    # prepare final datasets (ie all_holidays vs xmas_only)
    
    if holiday_type == 'all_holidays':
    
        holidays_df_final = trimac_series
        
        holidays_df_final = pd.merge(holidays_df_final,holidays_df,how='left',on=['index'])
        
        holidays_df_final = holidays_df_final.drop([Branch,'holiday'], axis = 1)
        
        holidays_df_final['holidays_dummy'] = holidays_df_final['holidays_dummy'].fillna(0)
        
        holidays_df_final.columns = ['all_holidays_dummy']
        
        holidays_df_final.iloc[:,0] = holidays_df_final.iloc[:,0].astype(int)
        
    if holiday_type == 'xmas_only':
    
        holidays_df_final = trimac_series
        
        holidays_df = holidays_df[(holidays_df['holiday'] == 'Christmas Day') | (holidays_df['holiday'] == 'Boxing Day') | (holidays_df['holiday'] == "New Year's Day")]
        
        holidays_df_final = pd.merge(holidays_df_final,holidays_df,how='left', on=['index'])
        
        holidays_df_final = holidays_df_final.drop([Branch,'holiday'], axis = 1)
        
        holidays_df_final['holidays_dummy'] = holidays_df_final['holidays_dummy'].fillna(0)
        
        holidays_df_final.columns = ['xmas_only_dummy']
        
        holidays_df_final.iloc[:,0] = holidays_df_final.iloc[:,0].astype(int)
        
    return holidays_df_final