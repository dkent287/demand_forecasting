import pandas as pd
import os
import tradingeconomics as te
import time

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()

def te_library_fun():
    
    #-------------------------------------------------------------------------------#
    # Step 1: Initialize the Trading Ecnomics service                               #
    #-------------------------------------------------------------------------------#
    
    # logging in to the trading economics api
    te.login('12A12ED2953A479:8A145F804D7049E')
    
    #-----------------------------------------------------------------------------------------#
    # Step 2: Build Trading Economics Data List                                               #
    #-----------------------------------------------------------------------------------------#
    
    te_library_desc = pd.DataFrame(columns = ['Country','Category','Title','Frequency','Type'])

    # Economic Indicator Data
        
    country_list = ['Canada','United States','Mexico']

    for country in country_list:
        temp = te.getIndicatorData(country = country, output_type = 'df')
        time.sleep(3)
        temp = temp.dropna(subset=['PreviousValue'])
        temp = temp[temp['CategoryGroup'] != 'Markets']
        temp = temp[['Country','Category','Title','Frequency']]
        temp['Type'] = "ECO"
        te_library_desc = pd.concat([te_library_desc,temp], axis=0, ignore_index="false")

    # Currency Markets Data 
    
    currency_types = ['CAD','USD']
    
    temp1 = pd.DataFrame(columns = ['Country','Category','Title','Frequency','Type'])

    for currency in currency_types:
        temp = te.getCurrencyCross(cross = currency, output_type = 'df')
        time.sleep(3)
        temp = temp[['Country','Symbol','Name','frequency']]
        temp = temp.rename(columns = {'Symbol': 'Category'})
        temp = temp.rename(columns = {'Name': 'Title'})
        temp = temp.rename(columns = {'frequency': 'Frequency'})
        temp['Type'] = "Currency"
        temp1 = pd.concat([temp1,temp], axis=0, ignore_index="false")

    curr_list = ['EURUSD','GBPUSD','USDCAD','USDCNY','USDJPY','EURCAD','CADCNY','CADJPY','GBPCAD']    

    temp1 = temp1[temp1['Title'].isin(curr_list)]
    
    temp1 = temp1.drop_duplicates(subset='Title')

    te_library_desc = pd.concat([te_library_desc,temp1], axis=0, ignore_index=False)

    # Other Markets Data (commodities, financial indicies and bond market)    
    
    market_types = ['commodities','index','bond']
    
    temp1 = pd.DataFrame(columns = ['Country','Category','Title','Frequency','Type'])
    
    for market_type in market_types:
        temp = te.getMarketsData(marketsField = market_type, output_type = 'df')
        time.sleep(3)
        temp = temp[['Country','Symbol','Name','frequency']]
        temp = temp.rename(columns = {'Symbol': 'Category'})
        temp = temp.rename(columns = {'Name': 'Title'})
        temp = temp.rename(columns = {'frequency': 'Frequency'})
        temp['Type'] = "Markets"
        temp1 = pd.concat([temp1,temp], axis=0, ignore_index="false") 
        
    commodities = temp1[temp1['Country'] == 'commodity']
    
    other = temp1[(temp1['Country'] == 'Canada') | (temp1['Country'] == 'United States') | (temp1['Country'] == 'united states') ]

    te_library_desc = pd.concat([te_library_desc,commodities,other], axis=0, ignore_index="false")    

    # clean-up list
    
    te_library_desc = te_library_desc.drop_duplicates(subset='Title')
    
    te_library_desc['Title'] = te_library_desc['Title'].str.replace('/',' ')
    
    os.chdir(dir_string + '/Data/Trading Economics')    
    te_library_desc.to_csv('te_library_desc_working.csv', index=False)    
    os.chdir(dir_string)
    
    return te_library_desc