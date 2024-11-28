# Import Needed Libraries and Functions
import pandas as pd


def report_write_fun(models_master_best, models_master_bestuni, Branch, holiday_feature,
                     xmas_feature, power_trans, min_max):

    # address 'Best Model' column
    
    best_model = ''
    
    if models_master_best['Model Type'].nunique() == 1 and models_master_best['Uni or Multi'].nunique() == 1:
        best_model = models_master_best.iloc[0,3] + ' ' + models_master_best.iloc[0,4] 
    else:
        best_model = 'Ensemble'
    
    # address Best Model Details columns
    
    best_model_details = ''
    for i1 in range(0,len(models_master_best)):
        best_model_details = best_model_details + models_master_best.iloc[i1,0] + '' + \
            models_master_best.iloc[i1,3] + ' ' + str(models_master_best.iloc[i1,5]) + '; ' + \
                models_master_best.iloc[i1,6] + '\n'
            
    # generate MAPEs for the best model set
    
    mape_best_1_10 = round((models_master_best.iloc[0,7] + models_master_best.iloc[1,7]) / 2, 2)
    mape_best_11_20 = round((models_master_best.iloc[2,7] + models_master_best.iloc[3,7]) / 2, 2)
    mape_best_21_30 = round((models_master_best.iloc[4,7] + models_master_best.iloc[5,7]) / 2, 2)
    mape_best_31_39 = round((models_master_best.iloc[6,7] + models_master_best.iloc[7,7]) / 2, 2)

    # address 'Best Model' column
    
    next_best_model = models_master_bestuni.iloc[0,3]
    
    # address Next Best Model Details columns
    
    next_best_model_details = ''
    for i1 in range(0,len(models_master_bestuni)):
        next_best_model_details = next_best_model_details + models_master_bestuni.iloc[i1,0] + '' + \
            models_master_bestuni.iloc[i1,3] + ' ' + str(models_master_bestuni.iloc[i1,5]) + '; ' + \
                models_master_bestuni.iloc[i1,6] + '\n'
    
    # generate MAPEs for the best model set
    
    mape_next_best_1_10 = round((models_master_bestuni.iloc[0,7] + models_master_bestuni.iloc[1,7]) / 2, 2)
    mape_next_best_11_20 = round((models_master_bestuni.iloc[2,7] + models_master_bestuni.iloc[3,7]) / 2, 2)
    mape_next_best_21_30 = round((models_master_bestuni.iloc[4,7] + models_master_bestuni.iloc[5,7]) / 2, 2)
    mape_next_best_31_39 = round((models_master_bestuni.iloc[6,7] + models_master_bestuni.iloc[7,7]) / 2, 2)
    
    # Holiday Features
    
    holiday = ''
    if holiday_feature == 'yes' and xmas_feature == 'yes':
        holiday = 'all holidays' + '\n' + 'xmas-only holidays'
    elif holiday_feature == 'yes' and xmas_feature == 'no':
        holiday = 'all holidays'
    elif holiday_feature == 'no' and xmas_feature == 'yes':
        holiday = 'xmas-only holidays'
    else:
        holiday = 'None'
        
    # Target Variable Transformation
    
    target_transform = ''
    
    if power_trans == 'yes' and min_max == 'yes':
        target_transform = 'power_trans' + '\n' + '0-1 Min-Max' + '\n'
    elif power_trans == 'yes' and min_max == 'no':
        target_transform = 'power_trans' + '\n'
    elif power_trans == 'no' and min_max == 'yes':
        target_transform = '0-1 Min-Max' + '\n'
        
    if models_master_best.iloc[0,14][1] is not None and models_master_best.iloc[0,14][1] != 'Failed':
        target_transform = target_transform + str(models_master_best.iloc[0,14][1])
    else:
        target_transform = target_transform + 'no stationarity transform'
    
    # Assemble Final Report
            
    report_list = [Branch, best_model, best_model_details, mape_best_1_10, mape_best_11_20, mape_best_21_30,
                   mape_best_31_39, next_best_model, next_best_model_details, mape_next_best_1_10, mape_next_best_11_20,
                   mape_next_best_21_30, mape_next_best_31_39, holiday, target_transform]
                             
    return report_list