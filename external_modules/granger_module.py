import pandas as pd
import os, re, os.path
from statsmodels.tsa.stattools import grangercausalitytests
import tradingeconomics as te
import time
from datetime import date, timedelta
from statistics import mean

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()
from service_modules.ancillary_functions import resample_fun, trim_fun
from service_modules.stationary_module import make_stationary_fun1


def gc_fun(production_train, production_full, te_library_desc, gc_data, gc_sig_level):
    
    #-------------------------------------------------------------------------------#
    # Step 1: Initialize the Trading Ecnomics service; Bring In Updated CCF Summary #
    #-------------------------------------------------------------------------------#
    
    # logging in to the trading economics api
    te.login('12A12ED2953A479:8A145F804D7049E')
    
    # set today's date
    today = date.today()
    today = today.strftime('%Y-%m-%d')
    
    # clearing out External Data - C directory; adding a placeolder for git
    mypath = dir_string + '/Data/Trading Economics/External Data - C'
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
    os.chdir(dir_string + '/Data/Trading Economics/External Data - C')
    f = open('gitplaceholderB.txt','w')
    f.close()
    del f
    os.chdir(dir_string)
    
    # bring in Updated CCF Summary; Remove Manually Deleted Items 
    os.chdir(dir_string + '/Data/CCF_GC_Data')
    CCF_Summary = pd.read_csv('CCF_Summary.csv', header=0, index_col=None, squeeze=True)
    os.chdir(dir_string)
    CCF_Summary = CCF_Summary[CCF_Summary['Delete'] != 'x']
    CCF_Summary = CCF_Summary.drop('Delete',1)

    #-------------------------------------------------------------------------------#
    # Step 2: bring in external data; build the GC_Matrix                           #
    #-------------------------------------------------------------------------------#
    
    GC_Matrix = pd.DataFrame(index=CCF_Summary['Title'], columns = production_train.columns)
    
    for i1 in range(0,len(CCF_Summary)):
        
        try:
        
            if gc_data == 'folderA':
                os.chdir(dir_string + '/Data/Trading Economics/External Data - A')
                te_series = pd.read_csv(CCF_Summary.iloc[i1,2] + '.csv', header=0, index_col=None, squeeze=False)
                os.chdir(dir_string)
                te_series['Date'] = pd.to_datetime(te_series['Date'])
                te_series = te_series.set_index('Date')

            elif gc_data == 'folderB':
                os.chdir(dir_string + '/Data/Trading Economics/External Data - B')
                te_series = pd.read_csv(CCF_Summary.iloc[i1,2] + '.csv', header=0, index_col=None, squeeze=False)
                os.chdir(dir_string)
                te_series['Date'] = pd.to_datetime(te_series['Date'])
                te_series = te_series.set_index('Date')
                
            else:
                if te_library_desc.iloc[i1,4] == "ECO":
                    te_series = te.getHistoricalData(country = CCF_Summary.iloc[i1,0],
                                                      indicator = CCF_Summary.iloc[i1,1],
                                                      initDate='2015-01-01',
                                                      endDate=today,
                                                      output_type = 'df')
                    time.sleep(10)
                    te_series = te_series[['DateTime', 'Value']]
                    te_series.columns = ['Date', 'Value']
                    te_series['Date'] = pd.to_datetime(te_series['Date'])
                    te_series = te_series.set_index('Date')
                    te_series = te_series.sort_index()
    
                    
                if te_library_desc.iloc[i1,4] != "ECO":
                    te_series = te.fetchMarkets(symbol = CCF_Summary.iloc[i1,1],
                                                initDate = '2015-01-01',
                                                endDate = today,
                                                output_type='df')
                    time.sleep(10)
                    te_series = te_series.reset_index()
                    te_series = te_series[['index','close']]
                    te_series.columns = ['Date','Value']
                    te_series['Date'] = pd.to_datetime(te_series['Date'])
                    te_series = te_series.set_index('Date')
                    te_series = te_series.sort_index()
            
            # creating a version of the series that we can save and use again in the future
            te_series_raw = te_series
            
            # resample the data to a weekly series
            te_series = resample_fun(te_series,CCF_Summary.iloc[i1,2])
    
            # make te_series stationary
            te_series, help_te = make_stationary_fun1(te_series)         
            
            # calculating te_series component of lag trim operation
            te_series = te_series.reset_index()
            te_series_latest = te_series['Date'].max()
            te_series = te_series.set_index('Date')
            
            # create the GC Matrix
                    
            flag1 = 'not_saved'
            
            for i2 in range(0,len(production_train.columns)):
                
                # remove nan values reulting from target_transform_fun operation
                prod_series = production_train.iloc[:,i2]
                
                # calculating lag trim value
                production_full = production_full.reset_index()
                production_full_latest =  production_full['index'].max()
                production_full = production_full.set_index('index')
                
                lag_trim_dt = production_full_latest - te_series_latest
                lag_trim = int((lag_trim_dt.days) / 7)
                lag_trim = max(lag_trim,0)
                
                # run triming operation
                prod_series, te_series = trim_fun(prod_series, te_series)
                
                # produce granger causality
                
                df_gc = pd.concat([prod_series, te_series], axis = 1)
                
                flag2 = 0
                maximum_lag = 53
                
                while flag2 == 0:
                    try:
                        gc = grangercausalitytests(df_gc, maxlag = maximum_lag)
                        flag2 = 1
                    except:
                        maximum_lag = maximum_lag - 1
                
                # extract relevant lags
                
                lag_list = []
                
                list1 = list(gc.items())
                
                for i3 in range(0,len(gc)):
                    
                    list2 = list(list1[i3])
                    
                    list3 = list(list2[1])
                    
                    list4 = list(list3[0].items())
                    
                    list5 = list(list4[0])
                    
                    list6 = list(list5[1])
                    
                    pvalue = list6[1]
                    
                    if pvalue < gc_sig_level:
                        lag_list.append([(i3 + 1), pvalue])
                
                # udate GC_Matrix
                
                if len(lag_list) == 0:
                    GC_Matrix.iloc[i1,i2] = 'NULL'
                else:
                    GC_Matrix.iloc[i1,i2] = [lag_trim, lag_list, help_te]
                    
                    if flag1 == 'not_saved':
                        os.chdir(dir_string + '/Data/Trading Economics/External Data - C')
                        te_series_raw.to_csv(CCF_Summary.loc[i1,'Title'] + '.csv')
                        os.chdir(dir_string)
                        flag1 = 'saved'
    
        except:
            GC_Matrix.iloc[i1,0] = 'Error'

    # cleaning up any cells missed in the operation above
    GC_Matrix = GC_Matrix.fillna('NULL')
    
    # saving GC_Matrix for future reference
    
    os.chdir(dir_string + '/Data/CCF_GC_Data')
    pd.to_pickle(GC_Matrix,'GC_Matrix.pkl')
    os.chdir(dir_string)
    
    return GC_Matrix