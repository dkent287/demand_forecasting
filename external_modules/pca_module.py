'''
This module can accept either one or 2 series inputs.  If there are 2 inputs, PCA is run on the second input 
with the PCA transformer fit on the first input.

'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca_fun(external_df_train, *args):
    
    if args:
        external_df_test = args[0]
        
    # split apart holidays from other exogenous variables
    if external_df_train.iloc[:,-1].name == 'all_holidays_dummy' or external_df_train.iloc[:,-1].name == 'xmas_only_dummy':
        if external_df_train.iloc[:,-2].name == 'all_holidays_dummy' or external_df_train.iloc[:,-2].name == 'xmas_only_dummy':
            holidays_train = external_df_train.iloc[:,-2:]
            external_df_train = external_df_train.iloc[:,:-2]
            feat_count = len(external_df_train.columns)
            if args:
                holidays_test = external_df_test.iloc[:,-2:]
                external_df_test = external_df_test.iloc[:,:-2]
        else:
            holidays_train = external_df_train.iloc[:,-1:]
            external_df_train = external_df_train.iloc[:,:-1]
            feat_count = len(external_df_train.columns)
            if args:
                holidays_test = external_df_test.iloc[:,-1:]
                external_df_test = external_df_test.iloc[:,:-1]
    else:
        holidays_train = pd.DataFrame()
        feat_count = len(external_df_train.columns)
        if args:
            holidays_test = pd.DataFrame()
    
    # run PCA
    if len(external_df_train.columns) > 4:

        ind_save_train = external_df_train.index
        if args:
            ind_save_test = external_df_test.index

        # standardize the data (i.e. mean 0 and st dev = 1)
        external_df_train_np = external_df_train.values
        if args:
            external_df_test_np = external_df_test.values
        
        scaler = StandardScaler()
        scaler.fit(external_df_train_np)
        external_df_train_sc = scaler.transform(external_df_train_np)
        if args:
            external_df_test_sc = scaler.transform(external_df_test_np)
    
        # run PCA; convert result back into dataframe
        pca = PCA(n_components=4)
        pca.fit(external_df_train_sc)
        
        external_df_train = pca.transform(external_df_train_sc)
        external_df_train = pd.DataFrame(data=external_df_train, index = ind_save_train,
                                         columns=['pca1','pca2','pca3','pca4'])
        
        if args:
            external_df_test = pca.transform(external_df_test_sc)
            external_df_test = pd.DataFrame(data=external_df_test, index = ind_save_test,
                                             columns=['pca1','pca2','pca3','pca4'])
    else:
        pca = None
           
    # put external series back together
    if len(holidays_train.columns) > 0 and len(external_df_train.columns) > 0:
        external_df_train = pd.concat([external_df_train, holidays_train], axis=1)
        if args:
            external_df_test = pd.concat([external_df_test, holidays_test], axis=1)
    elif len(holidays_train.columns) > 0 and len(external_df_train.columns) == 0:
        external_df_train = holidays_train
        if args:
            external_df_test = holidays_test
    else:
        pass # with have a non-holidays dataframe, nothing to do
         
    # produce feature description if appropriate
    if feat_count > 4:
        explained_variance_ratios = pca.explained_variance_ratio_
        explained_variance_total = round(np.sum(explained_variance_ratios),2)
        
        feature_description = '4 principal components with ' + str(explained_variance_total*100) + \
                        '% of explanatory power of ' + str(feat_count) + ' possible external series features'
    else:
        feature_description = None
     
    if args:
        return external_df_train, external_df_test, [pca, feature_description]
    else:
        return external_df_train, [pca, feature_description]
        
