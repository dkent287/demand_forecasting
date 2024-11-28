#-----------------------------------------------------------------------------#
# Step 1: Attend to Settings for this Run                                     #
#-----------------------------------------------------------------------------#
run_type = 'ccf_only' # options are te_dump_only, ccf_only or retrain
term_assign_scope = 'top-up' # options are top-up or full
te_library = 'existing' # options are existing or refresh; if refresh, make sure also to run te_dump to get updated te_library_desc
te_dump = 'skip' # options are run or skip; dumps current te database into Data\Trading Economics\External Data - A
ccf_data = 'folderA' # options are folderA or refresh; folderA uses te_dump result
gc_data = 'folderB' # options are folderA, folderB or refresh; folderA uses te_dump result; folderB uses ccf_fun result
ccf_sig_level = 0.05
gc_sig_level = 0.05
diag_sig_level = 0.05
test_length = 39
test_readings = 4
power_trans = 'no' # options are yes or no
min_max = 'no' # options are yes or no; this controls target scaling only; advise leave as "no"
holiday_feature = 'yes' # options are yes or no
xmas_feature = 'yes' # options are yes or no

'''
note re holiday_treatment and xmas_treatment: if either is 'yes', then we include it in the base model

'''
#-----------------------------------------------------------------------------#
# Step 2: Import Libraries                                                    #
#-----------------------------------------------------------------------------#
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------#
# Step 3: Import Internal Modules and Functions                               #
#-----------------------------------------------------------------------------#
from service_modules.directory_string_mod import directory_string
dir_string = directory_string()
os.chdir(dir_string)

# from miscellaneous_modules.dw_query_main import get_data

from service_modules.ancillary_functions import extract_lat_lng
from service_modules.ancillary_functions import model_set_selector_best, model_set_selector_best_uni
from service_modules.target_transform_module import target_transform_fun1
from service_modules.report_write_module import report_write_fun
from service_modules.report_export_module import report_export_fun
from service_modules.forecast_module import forecast_fun

from terminal_assignment_modules.terminal_assignment_A import terminal_assign_A
from terminal_assignment_modules.terminal_assignment_B import terminal_assign_B
from terminal_assignment_modules.terminal_assignment_C import terminal_assign_C

from external_modules.te_library import te_library_fun
from external_modules.te_dump_module import te_dump_fun
from external_modules.holidays_module import holidays_fun
from external_modules.ccf_module import ccf_fun
from external_modules.granger_module import gc_fun

from model_modules.sarimax import sarimax_fun
from model_modules.prophet import prophet_fun

# test Google Maps API functioning
extract_lat_lng('Toronto, ON','CA')

#-----------------------------------------------------------------------------#
# Step 4: Bring in Trimac Production Data                                     #
#-----------------------------------------------------------------------------#

os.chdir(dir_string + '/Data/Trimac Demand')

#data = get_data().set_index('CalendarWeek')    #load data from SQL Server
data2018 = pd.read_csv('data2018.csv')
data2019 = pd.read_csv('data2019.csv')
data2020 = pd.read_csv('data2020.csv')
data2021 = pd.read_csv('data2021.csv')

pdata2018 = pd.pivot_table(data2018, values='Travel_Miles', index=['CalendarWeek'],
                           columns=['Terminal'], fill_value=0)
pdata2019 = pd.pivot_table(data2019, values='Travel_Miles', index=['CalendarWeek'],
                           columns=['Terminal'], fill_value=0)
pdata2020 = pd.pivot_table(data2020, values='Travel_Miles', index=['CalendarWeek'],
                           columns=['Terminal'], fill_value=0)
pdata2021 = pd.pivot_table(data2021, values='Travel_Miles', index=['CalendarWeek'],
                           columns=['Terminal'], fill_value=0)

rng2018 = pd.date_range(start='2018-03-26', periods=len(pdata2018), freq='W') + pd.Timedelta('-7 day')
rng2019 = pd.date_range(start='2019-01-01', periods=len(pdata2019), freq='W') + pd.Timedelta('-7 day')
rng2020 = pd.date_range(start='2020-01-01', periods=len(pdata2020), freq='W') + pd.Timedelta('-7 day')
rng2021 = pd.date_range(start='2021-01-01', periods=len(pdata2021), freq='W') + pd.Timedelta('-7 day')

pdata2018 = pdata2018.set_index(rng2018).reset_index()
pdata2018 = pdata2018.drop(columns=['C03100','C03400']) 
pdata2019 = pdata2019.set_index(rng2019).reset_index()
pdata2020 = pdata2020.set_index(rng2020).reset_index()
pdata2021 = pdata2021.set_index(rng2021).reset_index()

#df = pd.concat([pdata2018, pdata2019, pdata2020, pdata2021]).groupby('index').sum()
#pdata = df.drop(columns = ['C00780'])

os.chdir(dir_string)

#-----------------------------------------------------------------------------#
# Step 5: Bring in Broker Data, Process and Add to Production Data            #
#-----------------------------------------------------------------------------#

os.chdir(dir_string + '/Data/Trimac Demand')
production_a = pd.read_csv('BrokerLoadCommodity.csv', header=0, index_col=None)
os.chdir(dir_string)

production_b,pcm_table = terminal_assign_A(production_a,term_assign_scope)

if production_b == 'null':
    pass
else:
    pauser = input("Pausing to let you run PC Miler.  Enter 'gonow' to continue . . .")
    if pauser == "go":
        pass

production_with_terminals = terminal_assign_B(production_b, pcm_table,term_assign_scope)

del production_a, production_b

brokerload_data = terminal_assign_C(production_with_terminals.copy())

pdata = pd.concat([pdata2018, pdata2019, pdata2020, pdata2021, brokerload_data]).groupby('index').sum()
pdata = pdata.drop(columns = ['C00780'])
pdata.index.freq = 'W'

# slimming down the list for test purposes
# pdata = pdata[['C00960','C00170','A03070','C00710','A03450','C00030']]
pdata = pdata[['C00170']]

#-----------------------------------------------------------------------------#
# Step 6: Bring In or Refresh Trading Economics Library                       #
#-----------------------------------------------------------------------------#

if te_library == 'existing':
    os.chdir(dir_string + '/Data/Trading Economics')
    te_library_desc = pd.read_csv('te_library_desc_working.csv')
    os.chdir(dir_string)
else:
    te_library_desc = te_library_fun()
    
#-----------------------------------------------------------------------------#
# Step 7: TE Dump                                                             #
#-----------------------------------------------------------------------------#

if te_dump == 'run':
    te_library_desc = te_dump_fun(te_library_desc)

#-----------------------------------------------------------------------------#
# Step 8: Create Master Training Set                                          #
#-----------------------------------------------------------------------------#

test_start = len(pdata) - test_length
pdata_train = pdata.iloc[:test_start,:]

#-----------------------------------------------------------------------------#
# Step 9: Transform Training Data                                             #
#-----------------------------------------------------------------------------#

pdata_train_transform = pd.DataFrame(index = pdata_train.index)

helper_master = pd.DataFrame(index = pdata_train.columns, columns = ['helpers'])

for i1 in range(0,len(pdata_train.columns)):
    series_transform, helper = target_transform_fun1(pdata_train.iloc[:,i1], power_trans, min_max)
    pdata_train_transform = pd.concat([pdata_train_transform, series_transform], axis = 1)
    helper_master.iloc[i1,0] = helper

pdata_train_transform = pdata_train_transform.dropna()

#-----------------------------------------------------------------------------#
# Step 10: Produce CCF Matrix and CCF Summary                                 #
#-----------------------------------------------------------------------------#

if run_type == 'ccf_only':
    CCF_Matrix, CCF_Summary, failed_series = ccf_fun(pdata_train_transform, te_library_desc,
                                                     ccf_data, ccf_sig_level)

#-----------------------------------------------------------------------------#
# Step 11: Produce GC Matrix                                                  #
#-----------------------------------------------------------------------------#

if run_type == 'retrain':
    GC_Matrix = gc_fun(pdata_train_transform, pdata, te_library_desc, gc_data, gc_sig_level)

# os.chdir(dir_string + '/Data/CCF_GC_Data')
# GC_Matrix = pd.read_pickle('GC_Matrix.pkl')
# os.chdir(dir_string)

#-----------------------------------------------------------------------------#
# Step 12: Generate Models and Forecasts for All Forecast Items               #
#-----------------------------------------------------------------------------#

if run_type == 'retrain':
        
    starttime = pd.Timestamp.now()
    
    report_master = []
    
    forecast_master = pd.DataFrame()
   
    for i1 in range(0,len(pdata.columns)):
     
        # creating an empty df to be used for model selection
        models_master = pd.DataFrame(columns=['Label','First Step','Last Step','Model Type',
                                              'Uni or Multi','Model_Parameters','Feature_Description',
                                              'Mape_Error','RMSE_Error','Diagnostic_Details',
                                              'Diagnostic_Overall','Bias_Correction',
                                              'External-DataFrame','Model - Final','Helper'])
        
        # assemble miles, branch name and transformation inputs
        Miles = pdata.iloc[:,i1]
        Branch = Miles.name
        helper = helper_master.iloc[i1,0]
        
        # assemble holidays features        
        holidays_main_extended = None
        holidays_main_type = "None"
        
        if holiday_feature == 'yes':
            all_holidays_extended = holidays_fun(pdata.iloc[:,i1],'all_holidays',test_length)
            holidays_main_extended = all_holidays_extended
            holidays_main_type = 'all holidays feature'
        
        if xmas_feature == 'yes':
            xmas_only_extended = holidays_fun(pdata.iloc[:,i1],'xmas_only',test_length)
            if holiday_feature == 'yes':
                holidays_main_extended = pd.concat([holidays_main_extended, xmas_only_extended], axis = 1)
                holidays_main_type = 'all holidays feature; xmas holidays only feature'
            else:
                holidays_main_extended = xmas_only_extended
                holidays_main_type = 'xmas holidays only feature'
          
        # run basic sarimax model (with or without holidays as selected)        
        model_summary = sarimax_fun(Miles, Branch, helper, test_length, test_readings,
                                    diag_sig_level, holidays_main_extended)
        models_master = pd.concat([models_master, model_summary], axis=0)
        
        # # run sarimax model with macro / markets data (with or without holidays)        
        # model_summary = sarimax_fun(Miles, Branch, helper, test_length, test_readings, diag_sig_level,
        #                             holidays_main_extended, GC_Matrix = GC_Matrix)
        # models_master = pd.concat([models_master, model_summary], axis=0)
        
        # run basic prophet model (with or without holidays)
        model_summary = prophet_fun(Miles, Branch, helper, test_length, test_readings, holidays_main_extended)
        models_master = pd.concat([models_master, model_summary], axis=0)

        # # run prophet model with macro / markets data (with or without holidays as selected)
        # model_summary = prophet_fun(Miles, Branch, helper, test_length, test_readings,
        #                             holidays_main_extended, GC_Matrix = GC_Matrix)
        # models_master = pd.concat([models_master, model_summary], axis=0)

        # select a best model set
        models_master_best = model_set_selector_best(models_master)
        
        # select a best univariable model
        models_master_bestuni = model_set_selector_best_uni(models_master)
  
        # update report_master
        report_list = report_write_fun(models_master_best, models_master_bestuni, Branch,
                                       holiday_feature, xmas_feature, power_trans, min_max)
        report_master.append(report_list)
    
        # update forecast_master
        forecast_results = forecast_fun(models_master_best, Branch, Miles)
        forecast_master = pd.concat([forecast_master, forecast_results], axis=0)
    
#-----------------------------------------------------------------------------#
# Step 13: Export report_master and forecast_master; Report Elapsed Time      #
#-----------------------------------------------------------------------------#

if run_type == 'retrain':

    # export report_master    
    report_export_fun(report_master)
    
    # export forcast master    
    today = str(pd.Timestamp.now().round('min'))
    today = today.replace(':','-',2)
    today = today[:-3]
    today = today[:10] + " T " + today[11:]
    os.chdir(dir_string + '/Data/Main Output')
    forecast_master.to_csv('Forecasts-' + today + '.csv')
    os.chdir(dir_string)
    
    # assess elapsed time
    endtime = pd.Timestamp.now()
    totaltime = endtime - starttime

