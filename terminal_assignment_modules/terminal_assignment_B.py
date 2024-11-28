# Import Needed Libraries and Functions

import numpy as np
import pandas as pd
import os
from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


def terminal_assign_B(production,pcm_table,term_assign_scope):

    if type(pcm_table) == str:
        
        os.chdir(dir_string + '/Data/Trimac Demand/Term_Assign_B_Output')
        production = pd.read_csv('term_assign_b_output.csv', header=0, index_col=None)
        os.chdir(dir_string)
        print('terminal_assign_B: No new data to process here.')
        
    else:
        
        # read in Batch-Pro Results and update column names
        
        colnames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']    
        
        os.chdir(dir_string + '/Data/PC Miler Data')
        
        terminal_orig_out = pd.read_csv('terminal_orig.OUT', header = None, names = colnames, index_col = None, sep = ' ', skipinitialspace = True)
        terminal_orig_out = terminal_orig_out.iloc[:,2:3]
        terminal_orig_out.columns = [['terminal_origin_miles']]
        
        orig_des_out = pd.read_csv('orig_des.OUT', header = None, names = colnames, index_col = None, sep = ' ', skipinitialspace = True)
        orig_des_out = orig_des_out.iloc[:,2:3]
        orig_des_out.columns = [['origin_dest_miles']]
        
        des_terminal_out = pd.read_csv('des_terminal.OUT', header = None, names = colnames, index_col = None, sep = ' ', skipinitialspace = True)
        des_terminal_out = des_terminal_out.iloc[:,2:3]
        des_terminal_out.columns = [['dest_terminal_miles']]
        
        os.chdir(dir_string)
    
        # combine the 3 separate Batch_pro results 
        totals = pd.concat([terminal_orig_out, orig_des_out,des_terminal_out], axis=1)
            
        # combine the Batch_pro results with pcm_table 
        pcm_table = pcm_table[['unique_lane','Terminal']]
        pcm_table = pd.concat([pcm_table,totals], axis=1)
        pcm_table.columns = ['unique_lane','Terminal','terminal_origin_miles','origin_dest_miles','dest_terminal_miles']
        
        # remove pc_miler errors 
        pcm_table = pcm_table[pcm_table['terminal_origin_miles'] != 'Error']
        pcm_table = pcm_table[pcm_table['origin_dest_miles'] != 'Error']
        pcm_table = pcm_table[pcm_table['dest_terminal_miles'] != 'Error']
        
        pcm_table['terminal_origin_miles'] = pcm_table['terminal_origin_miles'].astype(float)
        pcm_table['origin_dest_miles'] = pcm_table['origin_dest_miles'].astype(float)
        pcm_table['dest_terminal_miles'] = pcm_table['dest_terminal_miles'].astype(float)
            
        # calculate Out of Route Miles and other columns needed for contest
        pcm_table['OOR_Miles'] =  pcm_table['terminal_origin_miles'] + pcm_table['origin_dest_miles'] + \
            pcm_table['dest_terminal_miles'] - (2*pcm_table['origin_dest_miles'])
            
        pcm_table['OOR_Miles'] = pcm_table['OOR_Miles'] + pd.Series(np.random.normal(0,2,len(pcm_table)))
        
        pcm_table['OORMiles_percent'] = pcm_table['OOR_Miles'] / (pcm_table['terminal_origin_miles'] + pcm_table['origin_dest_miles'] + pcm_table['dest_terminal_miles'])
            
        pcm_table['terminal_origin_miles'] = pcm_table['terminal_origin_miles'] + pd.Series(np.random.normal(0,2,len(pcm_table)))
        
        # generate contest results and create contest look-up tables
        
        master_lookup1 = pd.pivot_table(pcm_table, 
                                    index = ['unique_lane'],
                                    values = ['OOR_Miles'],
                                    aggfunc=np.min)
        
        master_lookup1 = pd.merge(master_lookup1,pcm_table,how='left',on=['unique_lane','OOR_Miles'])
    
        master_lookup1 = master_lookup1[['unique_lane','Terminal','OORMiles_percent']]
        
        master_lookup1.columns = ['unique_lane','Terminal - LU1','OORMiles_%_best']
        
        master_lookup2 = pd.pivot_table(pcm_table, 
                                    index = ['unique_lane'],
                                    values = ['terminal_origin_miles'],
                                    aggfunc=np.min)
        
        master_lookup2 = pd.merge(master_lookup2,pcm_table,how='left',on=['unique_lane','terminal_origin_miles'])
    
        master_lookup2 = master_lookup2[['Terminal','terminal_origin_miles','origin_dest_miles','OORMiles_percent']]
    
        master_lookup2.columns = ['Terminal - LU2','terminal_origin_miles','origin_dest_miles','OORMiles_%_close']
        
        master_lookup3 = pd.concat([master_lookup1, master_lookup2], axis=1)
        
        master_lookup3['terminal'] = ""
        
        for i in range(0,len(master_lookup3)):
            master_lookup3.loc[i,'terminal'] = master_lookup3.loc[i,'Terminal - LU1']
            if (master_lookup3.loc[i,'terminal_origin_miles'] < 200 and master_lookup3.loc[i,'OORMiles_%_close'] < 0.25):
                master_lookup3.loc[i,'terminal'] = master_lookup3.loc[i,'Terminal - LU2']
            if (master_lookup3.loc[i,'terminal_origin_miles'] > 400 and master_lookup3.loc[i,'OORMiles_%_best'] > 0.75):
                master_lookup3.loc[i,'terminal'] = 'delete'
                
        master_lookup3 = master_lookup3[['unique_lane','terminal']]
        
        production = pd.merge(production,master_lookup3,how='left',on=['unique_lane'])
        
        production['terminal'] = production['terminal'].replace(np.nan, 'none', regex=True)
        
        production = production[production['terminal'] != 'none']
        production = production[production['terminal'] != 'delete']
        
        # adding previous data to the current product
        
        if term_assign_scope == 'top-up':
            os.chdir(dir_string + '/Data/Trimac Demand/Term_Assign_B_Output')
            production_hist = pd.read_csv('term_assign_b_output.csv', header=0, index_col=None)
            os.chdir(dir_string)
            production = pd.concat([production_hist, production], axis=0)
        
        # saving a copy of the result for future use with top-up runs
        
        os.chdir(dir_string + '/Data/Trimac Demand/Term_Assign_B_Output')
        production.to_csv('term_assign_b_output.csv',index = False)
        os.chdir(dir_string)

    return production