import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


def target_untransform_fun(series, helper):
      
    # step 1: reverse min-max scaling       
    if helper[2] is not None:
        scaler = helper[2]
        series = pd.DataFrame(series)
        series = scaler.inverse_transform(series)
        series = pd.DataFrame(series)
        series = pd.Series(series.iloc[:,0])
    
    # step 2: reverse stationarity transformation
         
    series_atlevel = series.copy() # setting initial value for the series this function will return
    
    # do nothing if we already have an at level forecast
    if helper[1] == (None or 'Failed Stationary'):
        pass
    # adjustment for forecast based on first differenced input
    elif helper[1] == '1 Step Differencing':
        series_atlevel.iloc[0] = helper[2][1] + series.iloc[0]
        for i1 in range(1,len(series)):
            series_atlevel.iloc[i1] = series_atlevel.iloc[i1-1] + series.iloc[i1]
    # adjustment for forecast based on second differenced input
    elif helper[1] == '2 Step Differencing':
        series_atlevel.iloc[0] = 2*helper[2][1] - helper[2][0] + series.iloc[0]
        series_atlevel.iloc[1] = 2*series_atlevel.iloc[0] - helper[2][1] + series.iloc[1]
        for i2 in range(2,len(series)):
            series_atlevel.iloc[i2] = 2*series_atlevel.iloc[i2-1] - series_atlevel.iloc[i2-2] + series.iloc[i2]
    # adjustment for forecast based on log transform
    elif helper[1] == 'Log Transform':
        series_atlevel = np.exp(series).dropna()
    # adjustment for forecast based on log transform and first differencing
    elif helper[1] == 'Log Transform & 1 Step Differencing':
        series_atlevel.iloc[0] = helper[2][1] + series.iloc[0]
        for i1 in range(1,len(series)):
            series_atlevel.iloc[i1] = series_atlevel.iloc[i1-1] + series.iloc[i1]
        series_atlevel = np.exp(series_atlevel).dropna()
    # adjustment for forecast based on log transform and second differencing
    elif helper[1] == 'Log Transform & 2 Step Differencing':
        series_atlevel.iloc[0] = 2*helper[2][1] - helper[2][0] + series.iloc[0]
        series_atlevel.iloc[1] = 2*series_atlevel.iloc[0] - helper[2][1] + series.iloc[1]
        for i2 in range(2,len(series)):
            series_atlevel.iloc[i2] = 2*series_atlevel.iloc[i2-1] - series_atlevel.iloc[i2-2] + series.iloc[i2]
        series_atlevel = np.exp(series_atlevel).dropna()
                        
    series = series_atlevel
     
    # step 3: reverse power transform
    
    if helper[0] is not None:
        pt = helper[0]
        series = pd.DataFrame(series)
        series = pt.inverse_transform(series)
        series = pd.DataFrame(series)
        series = pd.Series(series.iloc[:,0])
        
    return series


