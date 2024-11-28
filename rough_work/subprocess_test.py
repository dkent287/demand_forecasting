# Defining the R script and loading the instance in Python
import pandas as pd
import numpy as np
import os

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP

# build the R Function to be Called from Python
os.chdir('C:/Users/dkent/OneDrive - Trimac Management Services/resource_planning/r_processes')
with open('subprocess.r', 'r') as f:
    string = f.read()
forecast_func_in_python = STAP(string, "test_fun")
os.chdir('C:/Users/dkent/OneDrive - Trimac Management Services/resource_planning')

# Producing Test data
lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks']
lst2 = [11, 22, 33, 44, 55, 66, 77]
df = pd.DataFrame(list(zip(lst, lst2)),columns =['Name', 'Values'])

a = 1
b = 2
c = 32
alist = [a,b,c]

# Calling R function
e,f = forecast_func_in_python.test_fun(df,alist)

# unpacking the result 
f = list(np.array(f).flatten())


