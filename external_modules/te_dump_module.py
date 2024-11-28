import pandas as pd
import os
import tradingeconomics as te
import time
from datetime import date, timedelta

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


def te_dump_fun(te_library_desc):
    
    #-------------------------------------------------------------------------------#
    # Step 1: Initialize the Trading Ecnomics service and Prepare Folder            #
    #-------------------------------------------------------------------------------#
    
    # logging in to the trading economics api
    te.login('12A12ED2953A479:8A145F804D7049E')
    
    # set today's date
    today = date.today()
    today = today.strftime('%Y-%m-%d')
    
    # clearing out External Data - A directory; adding a placeolder for git
    mypath = dir_string + '/Data/Trading Economics/External Data - A'
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
    os.chdir(dir_string + '/Data/Trading Economics/External Data - A')
    f = open('gitplaceholderA.txt','w')
    f.close()
    del f
    os.chdir(dir_string)
    
    # prep te_library_desc for updates
    te_library_desc['Delete'] = ''
    
    #-------------------------------------------------------------------------------#
    # Step 2: bring in the external data and save to External Data - A              #
    #-------------------------------------------------------------------------------#
        
    for i1 in range(0,len(te_library_desc)):
        
        try:
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
                    
            # update te_library_desc if data does not go back far enough; otherwise save to file
            if te_series.first_valid_index() > pd.Timestamp(2016, 1, 2):
                te_library_desc.iloc[i1,5] = 'x'
            elif (te_series.last_valid_index() < (date.today() - timedelta(days=200))):
                te_library_desc.iloc[i1,5] = 'x'
            else:
                os.chdir(dir_string + '/Data/Trading Economics/External Data - A')
                te_series.to_csv(te_library_desc.loc[i1,'Title'] + '.csv')
                os.chdir(dir_string)
            
        except:
            pass
        
    # update te_library_desc; save for future use
    
    te_library_desc = te_library_desc[te_library_desc.iloc[:,5] == '']
    te_library_desc = te_library_desc.reset_index(drop = True)
    te_library_desc = te_library_desc.drop('Delete', axis = 1)

    os.chdir(dir_string + '/Data/Trading Economics')
    te_library_desc.to_csv('te_library_desc_working.csv', index=False)
    os.chdir(dir_string)

    return te_library_desc