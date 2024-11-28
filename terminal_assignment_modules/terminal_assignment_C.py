# Import Needed Libraries and Functions

import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import *

def terminal_assign_C(prod):
    
    # process the data so that it can be merged with the main truckline data
        
    prod['LegEndDate'] = prod['LegEndDate'].astype('str')
    
    prod['LegEndDate'] = pd.to_datetime(prod['LegEndDate'])
    
    prod['week_starting'] = prod['LegEndDate'].where(prod['LegEndDate'] == ((prod['LegEndDate'] + Week(weekday=6)) - Week()), prod['LegEndDate'] - Week(weekday=6))
    
    prod['week_starting'] = prod['week_starting'].astype('str')
    
    prod['terminal_week_starting'] = prod['terminal'] + 'v' + prod['week_starting']
    
    prod = pd.pivot_table(prod, 
                                index = ['terminal_week_starting'],
                                values = ['Travel_Miles'],
                                aggfunc=np.sum)
    
    prod = prod.reset_index()
    
    prod[['terminal','week_starting']] = prod['terminal_week_starting'].str.split('v',expand=True)
    
    prod = prod[['terminal','week_starting','Travel_Miles']]
    
    prod['week_starting'] = pd.to_datetime(prod['week_starting'])
    
    prod = pd.pivot_table(prod, values='Travel_Miles', index=['week_starting'],
                            columns=['terminal'], fill_value=0).reset_index()
    
    prod = prod.rename(columns={'week_starting':'index'})
    
    prod['index'] = pd.to_datetime(prod['index'])

    return prod