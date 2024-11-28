# Import Needed Libraries and Functions

import numpy as np
import pandas as pd
import csv
import os
from service_modules.ancillary_functions import extract_lat_lng
from service_modules.ancillary_functions import datacheck
from service_modules.ancillary_functions import string_prep

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()

def terminal_assign_A(production,term_assign_scope):
    
    #-----------------------------------------------------------------------------#
    # Step 0: Check for New Data                                                  #
    #-----------------------------------------------------------------------------#
    
    # triming down run if 'top-up' selected in main.py
    
    if term_assign_scope == 'top-up':
        
        production = production[['Order_Move_Key','LegNumber', 'LegEndDate','Shipper_City',
                             'Shipper_State','Shipper_Zip','Consignee_City','Consignee_State','Consignee_Zip',
                             'Shipper_Latitude','Shipper_Longitude','Consignee_Latitude','Consignee_Longitude',
                             'CLASS - 4 ','CLASS - 3','Travel_Miles']]
    
        production['LegEndDate'] = production['LegEndDate'].astype('str')
        production['LegEndDate'] = pd.to_datetime(production['LegEndDate'])
        
        os.chdir(dir_string + '/Data/Trimac Demand/Term_Assign_B_Output')
        production_hist = pd.read_csv('term_assign_b_output.csv', header=0, index_col=None)
        os.chdir(dir_string)
        
        production_hist['LegEndDate'] = production_hist['LegEndDate'].astype('str')
        production_hist['LegEndDate'] = pd.to_datetime(production_hist['LegEndDate'])
        
        last_date = production_hist['LegEndDate'].max()
        
        production = production[production['LegEndDate'] > last_date]
    
    if production.empty:
        pcm_table = "null"
        production = 'null'
        print('terminal_assign_A: No new data to process here. No need to run Batch Pro.')
    
    else:
    
        #-----------------------------------------------------------------------------#
        # Step 1: Attend Initial Data Prep                                            #
        #-----------------------------------------------------------------------------#
        
        # filtering down to needed columns
        
        production = production[['Order_Move_Key','LegNumber', 'LegEndDate','Shipper_City',
                                 'Shipper_State','Shipper_Zip','Consignee_City','Consignee_State','Consignee_Zip',
                                 'Shipper_Latitude','Shipper_Longitude','Consignee_Latitude','Consignee_Longitude',
                                 'CLASS - 4 ','CLASS - 3','Travel_Miles']]
        
        # get rid of "nan" values
        
        production = production.dropna(thresh=6)
        
        L1 = ['Order_Move_Key','LegNumber', 'LegEndDate','Shipper_City','Shipper_State','Shipper_Zip','Consignee_City',
             'Consignee_State','Consignee_Zip','Shipper_Latitude','Shipper_Longitude','Consignee_Latitude','Consignee_Longitude',
             'CLASS - 4 ','CLASS - 3','Travel_Miles']
        
        for i1 in range(0,len(L1)):
            production[L1[i1]] = production[L1[i1]].replace(np.nan,'none')
            production[L1[i1]] = production[L1[i1]].replace('nan','none')
       
        # converting LegEndDate to a datetime object
        
        production['LegEndDate'] = production['LegEndDate'].astype('str')
        production['LegEndDate'] = pd.to_datetime(production['LegEndDate'])
            
        # sort by Order_Move_Key and reindex
        production['Order_Move_Key'] = production['Order_Move_Key'].astype('str') 
        production = production.sort_values(['Order_Move_Key', 'LegNumber'], ascending=[True, True])
        production = production.reset_index()
        production = production.drop('index', 1)
        
        # convert some columns to strings
        production['LegNumber'] = production['LegNumber'].astype('str') 
        production['CLASS - 3'] = production['CLASS - 3'].astype('str') 
        production['CLASS - 4 '] = production['CLASS - 4 '].astype('str') 
        
        # remove "#" added to Shipper_City and Consignee_City
        production['Shipper_City'] = production['Shipper_City'].str.replace('#','')
        production['Consignee_City'] = production['Consignee_City'].str.replace('#','')
        
        # combine city and state into one column; remove the individual columns
        production['Shipper_City_State'] = production['Shipper_City'] + ', ' + production['Shipper_State']
        production['Consignee_City_State'] = production['Consignee_City'] + ', ' + production['Consignee_State']
        production = production.drop('Shipper_City', 1)
        production = production.drop('Consignee_City', 1)
        
        # recoding incorrect state values
        
        L2 = ['Shipper_State','Consignee_State']
        
        for i1 in range(0,len(L2)):
            
            state_list = pd.pivot_table(production, 
                                   index = [production[L2[i1]].name],
                                   values = ['Travel_Miles'],
                                   aggfunc=np.sum)
            
            state_list = state_list.drop('Travel_Miles', 1)
                    
            state_list = state_list.reset_index()
            
            for i2 in range(0,len(state_list)):
                
                if len(state_list.loc[i2,L2[i1]]) > 2:
                    state_list.loc[i2,L2[i1]] = 'XX'
                
                if state_list.loc[i2,L2[i1]].isnumeric():
                    state_list.loc[i2,L2[i1]] = 'XX'
                    
            state_list.columns = [L2[i1] + ' - corrected']
            
            production = pd.merge(production,
                                  state_list,
                                  how='left',
                                  left_on = L2[i1],
                                  right_on = L2[i1] + ' - corrected')
            
            production = production.replace(np.nan, 'none', regex=True)
            
            production = production.drop(L2[i1],axis =  1)
            
            production = production.rename(columns = {L2[i1] + ' - corrected':L2[i1]})
                          
        # add country code columns to the production dataset to help with the Google API call
        
        os.chdir(dir_string + '/Data/Trimac Terminals')
        states = pd.read_csv('states.csv', header=0, index_col=None) # bringing in a state - country converter
        os.chdir(dir_string)
        states.loc[16,'Mexico'] = 'NA-Mex'
        states = states.replace(np.nan, 'none', regex=True)
        states.loc[16,'Mexico'] = 'NA'
        
        production['shipper_country_code'] = ''
        production['consignee_country_code'] = ''
        
        L2 = ['shipper_country_code','consignee_country_code']
    
        for i1 in range(0,len(L2)):
            for i2 in range(0,len(production)):
                for i3 in range(0,len(states)):
                    if production.loc[i2,L2[i1]] in states.loc[i3,'Mexico']:
                        production.loc[i2,L2[i1]] = 'MX'
                for i3 in range(0,len(states)):
                    if production.loc[i2,L2[i1]] in states.loc[i3,'Canada']:
                        production.loc[i2,L2[i1]] = 'CA'
                for i3 in range(0,len(states)):
                    if production.loc[i2,L2[i1]] in states.loc[i3,'US']:
                        production.loc[i2,L2[i1]] = 'US'
                for i3 in range(0,len(states)):
                    if ((production.loc[i2,L2[i1]] in states.loc[i3,'Mexico']) & 
                        (production.loc[i2,L2[i1]] in states.loc[i3,'US'])):
                        production.loc[i2,L2[i1]] = ''
                for i3 in range(0,len(states)):
                    if ((production.loc[i2,L2[i1]] in states.loc[i3,'Mexico']) & 
                        (production.loc[i2,L2[i1]] in states.loc[i3,'Canada'])):
                        production.loc[i2,L2[i1]] = ''    
        
        #-----------------------------------------------------------------------------#
        # Step 2: Pivot Around MoveNumber                                             #
        #-----------------------------------------------------------------------------#
        
        for i in range(1,len(production)):
            if production.loc[i,'Order_Move_Key'] == production.loc[i-1,'Order_Move_Key']:
                production.loc[i-1,'LegEndDate'] = production.loc[i,'LegEndDate']
                production.loc[i,'Shipper_City_State'] = production.loc[i-1,'Shipper_City_State']
                production.loc[i-1,'Consignee_City_State'] = production.loc[i,'Consignee_City_State']
                production.loc[i,'Shipper_Latitude'] = production.loc[i-1,'Shipper_Latitude']
                production.loc[i,'Shipper_Longitude'] = production.loc[i-1,'Shipper_Longitude']
                production.loc[i-1,'Consignee_Latitude'] = production.loc[i,'Consignee_Latitude']
                production.loc[i-1,'Consignee_Longitude'] = production.loc[i,'Consignee_Longitude']
                production.loc[i,'CLASS - 3'] = production.loc[i-1,'CLASS - 3']
                production.loc[i,'CLASS - 4 '] = production.loc[i-1,'CLASS - 4 ']
                production.loc[i,'Shipper_Zip'] = production.loc[i-1,'Shipper_Zip']
                production.loc[i-1,'Consignee_Zip'] = production.loc[i,'Consignee_Zip']
        
        production = pd.pivot_table(production, 
                                    index = ['Order_Move_Key','LegEndDate','Shipper_City_State',
                                             'Consignee_City_State','Shipper_Latitude','Shipper_Longitude',
                                             'Consignee_Latitude', 'Consignee_Longitude','CLASS - 3',
                                             'CLASS - 4 ','Shipper_Zip', 'Consignee_Zip',
                                             'shipper_country_code','consignee_country_code'],
                                    values = ['Travel_Miles'],
                                    aggfunc=np.sum)
        
        production = production.reset_index()
        
        #-----------------------------------------------------------------------------#
        # Step 3: Correct Imported Lat-Long Data                                      #
        #-----------------------------------------------------------------------------#
        
        # convert columns to strings
        production['Shipper_Latitude'] = production['Shipper_Latitude'].astype(str).str[:10]
        production['Shipper_Longitude'] = production['Shipper_Longitude'].astype(str).str[:10]
        production['Consignee_Latitude'] = production['Consignee_Latitude'].astype(str).str[:10]
        production['Consignee_Longitude'] = production['Consignee_Longitude'].astype(str).str[:10]
        
        # add "-" to the Shipper longitude numbers
        for i in range(0,len(production)):
            if production.loc[i,'Shipper_Longitude'][0] != '-':
               production.loc[i,'Shipper_Longitude'] = '-' +  production.loc[i,'Shipper_Longitude']
        
        # add "-" to the Consignee longitude numbers
        for i in range(0,len(production)):
            if production.loc[i,'Consignee_Longitude'][0] != '-':
                production.loc[i,'Consignee_Longitude'] = '-' +  production.loc[i,'Consignee_Longitude']
     
        # convert lat-long data to floats where possible
        Lat_Long_List = ['Shipper_Latitude','Shipper_Longitude','Consignee_Latitude','Consignee_Longitude']
        
        for i1 in range(0,len(Lat_Long_List)):
            for i2 in range(0,len(production)):
                try:
                    production.loc[i2, Lat_Long_List[i1]] = pd.to_numeric(production.loc[i2, Lat_Long_List[i1]])                
                except:
                    pass
        
        # prepare check columns for existing lat and long values that convert strings to "9999"
        
        df1 = pd.DataFrame({'Starting': ['Shipper_Latitude','Shipper_Longitude','Consignee_Latitude','Consignee_Longitude'],
                           'Check': ['Shipper_Latitude_Check','Shipper_Longitude_Check','Consignee_Latitude_Check','Consignee_Longitude_Check']})
        
        for i1 in range(0,len(df1)):
            
            geo_pivot = pd.pivot_table(production, 
                                   index = [df1.iloc[i1,0]],
                                   values = ['Travel_Miles'],
                                   aggfunc=np.sum)
            geo_pivot = geo_pivot.reset_index()
            geo_pivot = geo_pivot.drop('Travel_Miles', axis = 1)
            
            geo_pivot[df1.iloc[i1,1]] = ''
            
            # Check Values
            for i2 in range(0,len(geo_pivot)):
                if type(geo_pivot[df1.iloc[i1,0]][i2]) == np.float64:
                    geo_pivot[df1.iloc[i1,1]][i2] = geo_pivot[df1.iloc[i1,0]][i2]
                else:
                    geo_pivot[df1.iloc[i1,1]][i2] = 9999
                    
            production = pd.merge(production,geo_pivot,how='left',on=df1.iloc[i1, 0])
            production = production.drop(df1.iloc[i1,0], axis = 1)
            production = production.rename(columns={df1.iloc[i1,1]: df1.iloc[i1,0]})
    
        #-------------------------------------------------------------------------------------------#
        # Step 4: Produce Alternate Lat-Long Data; Use it to Correct Existing Lat-Long Data         #
        #-------------------------------------------------------------------------------------------#
        
        # get alternate geocodes
        
        df2 = pd.DataFrame({'Shipper-Zip': ['Shipper_Zip','shipper_zip_lat','shipper_zip_lon'],
                           'Shipper-CityState': ['Shipper_City_State','shipper_citystate_lat','shipper_citystate_lon'],
                           'Consignee-Zip': ['Consignee_Zip','consignee_zip_lat','consignee_zip_lon'],
                           'Consignee-CityState': ['Consignee_City_State','consignee_citystate_lat','consignee_citystate_lon']})
      
        for i1 in range(0,len(df2.columns)):
            
            geo_pivot = pd.pivot_table(production, 
                                    index = [df2.iloc[0, i1]],
                                    values = ['Travel_Miles'],
                                    aggfunc=np.sum)
            geo_pivot = geo_pivot.reset_index()
            geo_pivot = geo_pivot.drop('Travel_Miles', axis = 1)
            
            if df2.iloc[0, i1][0] == 'S':
                country_code_pointer = 'shipper_country_code'
            else:
                country_code_pointer = 'consignee_country_code'
            
            prod_short = production.copy()
            prod_short = prod_short[[df2.iloc[0, i1],country_code_pointer]]
            geo_pivot = pd.merge(geo_pivot,prod_short,how='left',on=df2.iloc[0, i1])
            geo_pivot = geo_pivot.drop_duplicates(df2.iloc[0, i1])
            geo_pivot = geo_pivot.reset_index()
            
            geo_pivot[df2.iloc[1, i1]] = ''
            geo_pivot[df2.iloc[2, i1]] = ''
            
            for i2 in range(0,len(geo_pivot)):
                
                lat,lon = extract_lat_lng(geo_pivot.loc[i2,df2.iloc[0, i1]],
                                          geo_pivot.loc[i2,country_code_pointer])
                
                # dealing with lat
                geo_pivot.loc[i2,df2.iloc[1,i1]] = lat
                if type(geo_pivot.loc[i2,df2.iloc[1,i1]]) == float:
                    pass
                else:
                    geo_pivot.loc[i2,df2.iloc[1,i1]] = 9999
                
                # dealing with lon
                geo_pivot.loc[i2,df2.iloc[2,i1]] = lon
                if type(geo_pivot.loc[i2,df2.iloc[2,i1]]) == float:
                    pass
                else:
                    geo_pivot.loc[i2,df2.iloc[2,i1]] = 9999
    
            # merge result back into main production table
            production = production.drop(country_code_pointer,axis = 1)
            production = pd.merge(production,geo_pivot,how='left',on=df2.iloc[0, i1])
                    
        # use alternate geocodes to correct existing data
                          
        df3 = pd.DataFrame({'Zip': ['shipper_zip_lat','shipper_zip_lon','consignee_zip_lat','consignee_zip_lon'],
                           'CityState': ['shipper_citystate_lat','shipper_citystate_lon','consignee_citystate_lat','consignee_citystate_lon']})
        
        for i1 in range(0,len(df1)):
            for i2 in range(0,len(production)):
                if datacheck(production.loc[i2,df1.iloc[i1,0]],production.loc[i2,df3.iloc[i1,0]],production.loc[i2,df3.iloc[i1,1]]):
                    if (production.loc[i2,df3.iloc[i1,0]] < 200):
                        production.loc[i2,df1.iloc[i1,0]] = production.loc[i2,df3.iloc[i1,0]]
                    elif (production.loc[i2,df3.iloc[i1,1]] < 200):
                        production.loc[i2,df1.iloc[i1,0]] = production.loc[i2,df3.iloc[i1,1]]
                    else:
                        production.loc[i2,df1.iloc[i1,0]] = 'missing'
        
        # filter out rows where we ultimately have no lat-lon data    
        
        production['flag'] = ''
        
        for i1 in range(0,len(df1)):
            for i2 in range(0,len(production)):
                if production.loc[i2,df1.iloc[i1,0]] == 'missing':
                      production.loc[i2,'flag'] = 'delete'
    
        production = production[production['flag'] != 'delete']
        production = production.reset_index()
    
        # remove excess columns from the result    
        
        production = production[['Order_Move_Key', 'LegEndDate', 'Shipper_City_State',
           'Consignee_City_State', 'Shipper_Latitude', 'Shipper_Longitude',
           'Consignee_Latitude', 'Consignee_Longitude', 'CLASS - 3', 'CLASS - 4 ',
           'Shipper_Zip', 'Consignee_Zip', 'Travel_Miles']]
        
        # reformat lat-longs as strings and get ready for Batch-pro operation 
        
        for i1 in range(0,len(df1)):
            production[df1.iloc[i1,0]] = string_prep(production[df1.iloc[i1,0]])
        
        production['Shipper_Lat_Long'] = production['Shipper_Latitude'] + ',' + production['Shipper_Longitude']
        production = production.drop('Shipper_Latitude', axis = 1)
        production = production.drop('Shipper_Longitude', axis = 1)
    
        production['Consignee_Lat_Long'] = production['Consignee_Latitude'] + ',' + production['Consignee_Longitude']
        production = production.drop('Consignee_Latitude', axis = 1)
        production = production.drop('Consignee_Longitude', axis = 1)
        
        #----------------------------------------------------------------------------------------------------#
        # Step 5: Make Unique Lane Tabe; Import Branch Table; Run Contest; Assemble Result                   #
        #----------------------------------------------------------------------------------------------------#
        
        ### a) create unique_lane table
        
        production['unique_lane'] = production['Shipper_Lat_Long'] + '; ' + production['Consignee_Lat_Long'] \
            + '; '+ production['CLASS - 3']
        
        unique_lane = pd.pivot_table(production, 
                                     index = ['unique_lane','CLASS - 3','Shipper_Lat_Long','Consignee_Lat_Long'],
                                     values = None)
        
        unique_lane = unique_lane.drop('Travel_Miles', 1)
        
        unique_lane = unique_lane.reset_index()
        
        ### b) create terminal input table
        
        # import the terminal table
        os.chdir(dir_string + '/Data/Trimac Terminals')
        terminal_table = pd.read_csv('terminal_input.csv')
        os.chdir(dir_string)
    
        terminal_table = terminal_table[['Terminal','Business Lines','Terminal_Lat_Long']]
        terminal_table['Business Lines'] = terminal_table['Business Lines'].astype(str)
        
        terminal_table = terminal_table[terminal_table['Business Lines'] != '?']
        terminal_table = terminal_table.reset_index()
        
        # split the lat-long string
        new = terminal_table['Terminal_Lat_Long'].str.split(', ',expand=True)
        new.columns = ['Lat','Lon']
        terminal_table = pd.concat([terminal_table,new], axis=1)
        terminal_table = terminal_table.drop('Terminal_Lat_Long', axis=1)
        
        # clean up the separate lat and long strings
        terminal_table['Lat'] = terminal_table['Lat'].str.pad(width=11, side='right', fillchar='0')
        terminal_table['Lat'] = terminal_table['Lat'] + 'N'
        
        for i in range(0,len(terminal_table)):
           if terminal_table.loc[i,'Lon'][0] == '-':
               terminal_table.loc[i,'Lon'] = terminal_table.loc[i,'Lon'][1:]
        terminal_table['Lon'] = terminal_table['Lon'].str.pad(width=11, side='right', fillchar='0')
        terminal_table['Lon'] = terminal_table['Lon'] + 'W'
        
        # combine cleaned-up lat and long columns
        terminal_table['Terminal_Lat_Long'] = terminal_table['Lat'] + ',' + terminal_table['Lon']
        terminal_table = terminal_table.drop('Lat', axis=1)
        terminal_table = terminal_table.drop('Lon', axis=1)
        
        ### c) merge branch table with specific_lane_table; filter out branches not eligible for contest
        
        terminal_table['tmp'] = 1
        unique_lane['tmp'] = 1
        pcm_table = pd.merge(terminal_table, unique_lane, on=['tmp'])
        
        pcm_table = pcm_table.drop('tmp', axis=1)
        pcm_table = pcm_table.drop('index', axis=1)
        terminal_table = terminal_table.drop('tmp', axis=1)
        unique_lane = unique_lane.drop('tmp', axis=1)
        
        pcm_table = pcm_table.sort_values(['Shipper_Lat_Long',
                                           'Consignee_Lat_Long',
                                           'Terminal'], ascending=[True,True,True])
        
        # filter out combinations where the branch does not offer the service
        for i in range(0,len(pcm_table)):
            if pcm_table['CLASS - 3'][i] in pcm_table['Business Lines'][i]:
                pcm_table.at[i,'flag'] = "Keep"
            else:
                pcm_table.at[i,'flag'] = "Delete"
        
        pcm_table = pcm_table[pcm_table['flag'] == 'Keep']
        
        pcm_table = pcm_table.reset_index()
        pcm_table = pcm_table.drop('index', axis=1)
        
        ### d) prepare PC Miler Inputs from Merged Table
        
        terminal_orig_pairs = pcm_table[['Terminal_Lat_Long','Shipper_Lat_Long']]
        orig_des_pairs = pcm_table[['Shipper_Lat_Long','Consignee_Lat_Long']]
        des_terminal_pairs = pcm_table[['Consignee_Lat_Long','Terminal_Lat_Long']]
        
        os.chdir(dir_string + '/Data/PC Miler Data')
        terminal_orig_pairs.to_csv('terminal_orig.IN', index=False, sep=' ', header=None,
                                     quoting = csv.QUOTE_NONE, escapechar=" ")
        orig_des_pairs.to_csv('orig_des.IN', index=False, sep=' ', header=None,
                                     quoting = csv.QUOTE_NONE, escapechar=" ")
        des_terminal_pairs.to_csv('des_terminal.IN', index=False, sep=' ', header=None,
                                     quoting = csv.QUOTE_NONE, escapechar=" ")
        os.chdir(dir_string)
    
    return production, pcm_table