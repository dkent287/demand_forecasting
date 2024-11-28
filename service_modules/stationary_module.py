import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def make_stationary_fun1(series):
    
    helper = []
    
    series_copy = series.copy()
    
    a,b,c,d,e,f = adfuller(series)
    g,h,i,j = kpss(series, 'c')
    if b < 0.01 and h > 0.01:
        helper.append(None)
        helper.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])   
    else:
        # try first differecing
        series = series.diff().dropna()
        a,b,c,d,e,f = adfuller(series)
        g,h,i,j = kpss(series, 'c')
        if b < 0.01 and h > 0.01:
            helper.append('1 Step Differencing')
            helper.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])
        else: 
            # try second differecing
            series = series.diff().dropna()
            a,b,c,d,e,f = adfuller(series.dropna())
            g,h,i,j = kpss(series.dropna(),'c')
            if b < 0.01 and h > 0.01:
                helper.append('2 Step Differencing')
                helper.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])
            else:
                # try log transform
                series = series_copy
                series = np.log(series)
                a,b,c,d,e,f = adfuller(series)
                g,h,i,j = kpss(series.dropna(),'c')
                if b < 0.01 and h > 0.01:
                    helper.append('Log Transform')
                    helper.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])
                else:
                    # try log transform and first differencing
                    log_series = series.copy()
                    series = series.diff().dropna()
                    a,b,c,d,e,f = adfuller(series.dropna())
                    g,h,i,j = kpss(series.dropna(),'c')
                    if b < 0.01 and h > 0.01:
                        helper.append('Log Transform & 1 Step Differencing')
                        helper.append([float(log_series.iloc[-2]),float(log_series.iloc[-1])])
                    else:
                        # try log transform and second diffeencing
                        series = series.diff().dropna()
                        a,b,c,d,e,f = adfuller(series.dropna())
                        g,h,i,j = kpss(series.dropna(),'c')
                        if b < 0.01 and h > 0.01:
                            helper.append('Log Transform & 2 Step Differencing')
                            helper.append([float(log_series.iloc[-2]),float(log_series.iloc[-1])])
                        else:
                            # report failure to make stationary
                            helper.append('Failed Stationary')
                            helper.append([float(series_copy.iloc[-2]),float(series_copy.iloc[-1])])
    
    return series, helper


def make_stationary_fun2(series, helper):
    
    # step 1: set helper objects
    helper2 = []
    series_copy = series.copy()
    stationary = helper[0]
          
    # step 2: make stationary
    
    series_copy = series.copy()
    
    if stationary == None or stationary == 'Failed Stationary':
        helper2.append('None')
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
        
    return series, helper2
