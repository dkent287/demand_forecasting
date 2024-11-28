import pandas as pd
import numpy as np
import os, os.path
import statsmodels.tsa.stattools as smt
import tradingeconomics as te
import time
from datetime import date
from scipy import stats

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()
from service_modules.ancillary_functions import resample_fun, trim_fun
from service_modules.stationary_module import make_stationary_fun1


def ccf_fun(production, te_library_desc, ccf_data, ccf_sig_level):
    
    #-------------------------------------------------------------------------------#
    # Step 1: Initialize the Trading Ecnomics service                               #
    #-------------------------------------------------------------------------------#
    
    # logging in to the trading economics api
    te.login('12A12ED2953A479:8A145F804D7049E')
    
    # set today's date
    today = date.today()
    today = today.strftime('%Y-%m-%d')
    
    # clearing out External Data - B directory; adding a placeolder for git
    mypath = dir_string + '/Data/Trading Economics/External Data - B'
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
    os.chdir(dir_string + '/Data/Trading Economics/External Data - B')
    f = open('gitplaceholderA.txt','w')
    f.close()
    del f
    os.chdir(dir_string)
    
    #-------------------------------------------------------------------------------#
    # Step 2: bring in the external data; build the ccf matrix                      #
    #-------------------------------------------------------------------------------#
    
    # failed_series will be used to asseble a list series we cannot make stationary
    stationary_failed_list = []
    irreg_period_list = []
    cutoff_failed_list = []
    
    # create CCF_Matrix to collect main results
    CCF_Matrix = pd.DataFrame(index=te_library_desc['Title'], columns = production.columns)
    
    for i1 in range(0,len(te_library_desc)):
                
        if ccf_data == 'folderA':
            os.chdir(dir_string + '/Data/Trading Economics/External Data - A')
            te_series = pd.read_csv(te_library_desc.iloc[i1,2] + '.csv', header=0, index_col=None, squeeze=False)
            os.chdir(dir_string)
            te_series['Date'] = pd.to_datetime(te_series['Date'])
            te_series = te_series.set_index('Date')
        
        else:
                            
            if te_library_desc.iloc[i1,4] == "ECO":
                te_series = te.getHistoricalData(country = te_library_desc.iloc[i1,0],
                                                  indicator = te_library_desc.iloc[i1,1],
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
                te_series = te.fetchMarkets(symbol = te_library_desc.iloc[i1,1],
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
        te_series = resample_fun(te_series,te_library_desc.iloc[i1,2])
        
        # a check for whether the series is of an irregular length (like BOC rate); if so, delete it
        irreg_period = False
        if te_series.isnull().values.any():
            irreg_period = True
            irreg_period_list.append(te_library_desc.iloc[i1,2])
        
        # make te_series stationary if we can; if not, remove it
        stationary_failed = False
        if irreg_period == False:
            te_series, helper = make_stationary_fun1(te_series)
            if helper[0] == 'Failed':
                stationary_failed = True
                stationary_failed_list.append(te_library_desc.iloc[i1,2])
        
        # run CCF calculations and build CCF Matrix
        
        if stationary_failed == False and irreg_period == False:
        
            flag1 = 'not_saved'
            
            for i2 in range(0,len(production.columns)):
                
                # access a Trimac series
                prod_series = production.iloc[:,i2]
    
                # run triming operation
                prod_series, te_series = trim_fun(prod_series, te_series)
                
                # calculate lag with largest correlation; significance-test same
                                            
                ccf_out = smt.ccf(te_series.iloc[::-1], prod_series.iloc[::-1], adjusted=True)
                
                ccf_postive = np.absolute(ccf_out)
                
                ccf_postive = ccf_postive[0:78]
                
                ccf_max = np.argmax(ccf_postive)
                
                te_series_shifted = te_series.shift(periods = ccf_max).dropna()
                    
                prod_series = prod_series.reset_index()
                prod_series_shifted = prod_series.iloc[ccf_max:,:]
                prod_series = prod_series.set_index('index')
                prod_series_shifted = prod_series_shifted.set_index('index')
                
                prod_series_shifted, te_series_shifted = trim_fun(prod_series_shifted, te_series_shifted)
                
                te_series_shifted = te_series_shifted.iloc[:,0]
                prod_series_shifted = prod_series_shifted.iloc[:,0]
    
                try:
                    corr, p_value = stats.pearsonr(prod_series_shifted, te_series_shifted)
                except:
                    p_value = 1
                    cutoff_failed_list.append(te_library_desc.iloc[i1,2])
                
                # update CCF_Matrix based on the above
                           
                if ccf_max > 0 and p_value < ccf_sig_level:
                    
                    CCF_Matrix.iloc[i1,i2] = np.argmax(ccf_postive)
                    
                    if flag1 == 'not_saved':
                        os.chdir(dir_string + '/Data/Trading Economics/External Data - B')
                        te_series_raw.to_csv(te_library_desc.loc[i1,'Title'] + '.csv')
                        os.chdir(dir_string)
                        flag1 = 'saved'
                else: 
                    CCF_Matrix.iloc[i1,i2] = 'NULL'
                    
        else:
            CCF_Matrix.iloc[i1,] = 'NULL'
            
    # save CCF_Matrix to file
    os.chdir(dir_string + '/Data/CCF_GC_Data')
    CCF_Matrix.to_csv('CCF_Matrix.csv')
    os.chdir(dir_string)
    
    #-------------------------------------------------------------------------------#
    # Step 4: Use CCF_Matrix to Build CCF_Summary                                   #
    #-------------------------------------------------------------------------------#

    CCF_Summary = CCF_Matrix
            
    CCF_Summary = CCF_Summary[(CCF_Summary.T != 'NULL').any()]
    
    CCF_Summary = CCF_Summary.reset_index()
    
    CCF_Summary['Delete'] = ""
        
    CCF_Summary = pd.merge(CCF_Summary,
                                  te_library_desc,
                                  how='left',
                                  on = 'Title')
    
    CCF_Summary = CCF_Summary[['Country','Category','Title','Frequency','Type','Delete']]

    os.chdir(dir_string + '/Data/CCF_GC_Data')
    CCF_Summary.to_csv('CCF_Summary.csv', index =False)
    os.chdir(dir_string)
    
    return CCF_Matrix, CCF_Summary, [irreg_period_list, stationary_failed_list, cutoff_failed_list]