'''
Notes:
There are 2 modules here.
The first module transforms data for the first time and does the work needed to figure out
what kinds of transformations are needed.
The second module transforms data, but does so based on the pattern from a previous transformation.

'''

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


def target_transform_fun1(series, power_trans, min_max):
    
    # step 1: set helper objects
    helper = []
    col_name = series.name
  
    # step 2: run power transform      
    if power_trans == 'yes':
        ind = series.index # saving index for later use
        pt = PowerTransformer(method='box-cox', standardize = False)
        series = pd.DataFrame(series)
        pt = pt.fit(series)
        series = pt.transform(series)
        series = pd.DataFrame(series, index = ind)
        series = pd.Series(series.iloc[:,0])
        series.name = col_name
        helper.append(pt)
    else:
        helper.append(None)
    
    # step 3: make stationary   
    series_copy = series.copy()
 
    a,b,c,d,e,f = adfuller(series)
    g,h,i,j = kpss(series, 'c')
    if b < 0.01 and h > 0.01:
        helper.append('Stationary at Level')
    else:
        # try first step differecing
        series = series.diff().dropna()
        a,b,c,d,e,f = adfuller(series)
        g,h,i,j = kpss(series, 'c')
        if b < 0.01 and h > 0.01:
            helper.append('1 Step Differencing')
        else: 
            # try second step differecing
            series = series.diff().dropna()
            a,b,c,d,e,f = adfuller(series)
            g,h,i,j = kpss(series,'c')
            if b < 0.01 and h > 0.01:
                helper.append('2 Step Differencing')
            else:
                # try log transform
                series = series_copy
                series = np.log(series)
                a,b,c,d,e,f = adfuller(series)
                g,h,i,j = kpss(series,'c')
                if b < 0.01 and h > 0.01:
                    helper.append('Log Transform')
                else:
                    # try log transform and first differencing
                    series = series.diff().dropna()
                    a,b,c,d,e,f = adfuller(series)
                    g,h,i,j = kpss(series,'c')
                    if b < 0.01 and h > 0.01:
                        helper.append('Log Transform & 1 Step Differencing')
                    else:
                        # try log transform and second diffeencing
                        series = series.diff().dropna()
                        a,b,c,d,e,f = adfuller(series)
                        g,h,i,j = kpss(series,'c')
                        if b < 0.01 and h > 0.01:
                            helper.append('Log Transform & 2 Step Differencing')
                        else:
                            # report failure to make stationary
                            helper.append('Failed Stationary')
        
    # step 4: run min-max scaling
    if min_max == 'yes':
        ind = series.index # updating index after differencing operation
        scaler = MinMaxScaler()
        series = pd.DataFrame(series)
        scaler = scaler.fit(series)
        series = scaler.transform(series)
        series = pd.DataFrame(series, index = ind)
        series = pd.Series(series.iloc[:,0])
        series.name = col_name
        helper.append(scaler)
    else:
        helper.append(None)
  
    return series, helper


def target_transform_fun2(series, helper):
       
    # step 1: set helper objects
    helper2 = []
    col_name = series.name
    if helper[0] is not None:
        power_trans = 'yes'
        pt = helper[0]
    else:
        power_trans = 'no'
    if helper[2] is not None:
        min_max = 'yes'
        scaler = helper[2]
    else:
        min_max = 'no'
    stationary = helper[1]
          
    # step 2: run power transform
    if power_trans == 'yes':
        ind = series.index # updating index after differencing operation
        series = pd.DataFrame(series)
        series = pt.transform(series)
        series = pd.DataFrame(series, index = ind)
        series = pd.Series(series.iloc[:,0])
        series.name = col_name
        helper2.append(pt)
    else:
        helper2.append(None)
        
    # step 3: make stationary
    series_copy = series.copy()
    
    if stationary == None or stationary == 'Failed Stationary':
        helper2.append(None)
        helper2.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])
    elif stationary == '1 Step Differencing':
        series = series.diff().dropna()
        helper2.append('1 Step Differencing')
        helper2.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])   
    elif stationary == '2 Step Differencing':
        series = series.diff().dropna()
        series = series.diff().dropna()
        helper2.append('2 Step Differencing')
        helper2.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])   
    elif stationary == 'Log Transform':
        series = np.log(series)
        helper2.append('Log Transform')
        helper2.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])   
    elif stationary == 'Log Transform & 1 Step Differencing':
        series = np.log(series)
        log_series = series.copy()
        series = series.diff().dropna()        
        helper2.append('Log Transform & 1 Step Differencing')
        helper2.append([float(log_series.iloc[-2]),float(log_series.iloc[-1])])
    elif stationary == 'Log Transform & 2 Step Differencing':
        series = np.log(series)
        log_series = series.copy()
        series = series.diff().dropna()
        series = series.diff().dropna()
        helper2.append('Log Transform & 2 Step Differencing')
        helper2.append([float(log_series.iloc[-2]),float(log_series.iloc[-1])])
       
    # step 4: run min-max scaling
    if min_max == 'yes':
        ind = series.index # updating index after differencing operation
        series = pd.DataFrame(series)
        series = scaler.transform(series)
        series = pd.DataFrame(series, index = ind)
        series = pd.Series(series.iloc[:,0])
        series.name = col_name
        helper2.append(scaler)
    else:
        helper2.append(None)
     
    return series, helper2
