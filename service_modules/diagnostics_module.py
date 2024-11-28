import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_white


def diagnostics_fun(trimac_series, residuals_bias_corrected, cutoff, model, diag_sig_level, **kwargs):
       
    ### step 1: verify residuals are normally distributed with the Shapiro-Wilk Test
    ### reference: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
    
    test_stat, p_value = shapiro(residuals_bias_corrected)
    
    if p_value > diag_sig_level:
        shapiro_wilk_test = 'pass'
    else:
        shapiro_wilk_test = 'fail'
    
    ### step 2: verify no residual autocorrelation with the Breusch_Godfrey Test
    ### https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html
    ### reference:https://www.statology.org/breusch-godfrey-test-python/
    
    # create equivilent OLS model
    X = add_constant(residuals_bias_corrected)
    X = X['const']
        
    OLS_model = OLS(residuals_bias_corrected, X)
    OLS_fit = OLS_model.fit()
    
    # run Breusch Godfrey Test
    a, p_value, c, d = acorr_breusch_godfrey(OLS_fit, nlags=13)
    
    if p_value > diag_sig_level:
        breusch_godfrey_test = 'pass'
    else:
        breusch_godfrey_test = 'fail'
    
    ### step 3: verify constant residual variance with the White Test
    ### statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html
    ### reference:https://towardsdatascience.com/heteroscedasticity-is-nothing-to-be-afraid-of-730dd3f7ca1f
    ### reference: https://www.youtube.com/watch?v=jy2OurbLFGg
    
    # create exogenous test input
    
    a = np.ones(shape=(len(trimac_series),1))
    model_variables = pd.DataFrame(a, index = trimac_series.index)
    
    if 'exog_1' in kwargs:
        model_variables = pd.concat([model_variables, kwargs['exog_1']], axis = 1)
    
    model_variables = model_variables.to_numpy()[cutoff + model.k_ar:] 
    
    trimac_series = trimac_series.to_numpy()[cutoff:]
    for i1 in range(0, model.k_ar):
        stub = trimac_series
        if (model.k_ar - i1 - 1) == 0:
            pass
        else:
            stub = trimac_series[(model.k_ar - i1 -1):]
        stub = stub[:-(i1 + 1)]
        model_variables = np.hstack([model_variables,stub[:,None]])
        
    # make sure that model_variables has at least 2 columns
    if np.size(model_variables,1) < 2:
        adder = np.ones(shape=(len(model_variables),1))
        model_variables = np.append(model_variables, adder, axis=1)

    # create the residual test input
    residuals_bias_corrected = residuals_bias_corrected.to_numpy()
    residuals_bias_corrected = residuals_bias_corrected[model.k_ar:]

    # run the test
    
    a, p_value, c, d = het_white(residuals_bias_corrected, model_variables)
    
    if p_value > diag_sig_level:
        white_test = 'pass'
    else:
        white_test = 'fail'
      
    ### step 5: assemble results
    
    if shapiro_wilk_test == 'pass' and breusch_godfrey_test == 'pass' and white_test == 'pass':
        diag_result = 'pass'
    else:
        diag_result = 'fail'
           
    return [shapiro_wilk_test, breusch_godfrey_test, white_test], diag_result