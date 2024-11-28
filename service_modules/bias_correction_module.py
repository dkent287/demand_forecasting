'''
NOTES:
    
This module takes a series and a fitted model as inputs and produces a bias correction parameter.

The bias correction parameter should be added to the initial estimates produced by the model or 
subtracted from the initial residuals produced by the model.

The cutoff variable deals with residual values that are the same as the original series

'''

import numpy as np
import pandas as pd


def bias_correction_fun(trimac_series, model_fit, bias_return):
    
    ### step 1: extract residuals from model
    
    residuals = model_fit.resid
    
    ### step 2: correct the residuals series
    
    add_rows = len(trimac_series) - len(residuals)
    
    if add_rows > 0:
        for i1 in range(0,add_rows):
            residuals = np.insert(residuals, 0, 0, axis=0)
    
    i = 0
    cutoff = 0
    while i == 0:
        if (trimac_series[cutoff] == residuals[cutoff]) or (residuals[cutoff] == 0):
            cutoff = cutoff + 1
            if cutoff > 10:
                i = 1
        else:
            i = 1

    residuals_corrected = residuals[cutoff:]
    
    ### step 3: calculate bias correction factor as mean of the residuals
    
    residuals_mean = residuals_corrected.mean()
    
    residuals_corrected = residuals_corrected - residuals_mean
    
    ### step 4: prepare values to be returned
    
    if bias_return == 'corrected_residuals':
        bias_result = residuals_corrected
    elif bias_return == 'bias_adjustment':
        bias_result = residuals_mean
    
    return bias_result, cutoff
