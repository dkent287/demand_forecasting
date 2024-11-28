'''
Notes:
It is difficult / not convenient to save Prophet models as objects in dataframes for later use.  This module 
converts these models into json files which CAN be saved as dataframe objects for later use.  This module has
functions to convert both ways.

'''

import os
import json
from prophet.serialize import model_to_json, model_from_json
from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


def json_create(model):

    os.chdir(dir_string + '/Data/Temp_Object_Holder/JSON_Slush_Folder')

    with open('serialized_model.json', 'w') as fout:
        json.dump(model_to_json(model), fout)  # save model as json
    
    with open('serialized_model.json', 'r') as myfile:
        model_json = myfile.read() # bring back in as unparsed json
    
    # clear folder
    mypath = dir_string + '/Data/Temp_Object_Holder/JSON_Slush_Folder'
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    f = open('gitplaceholderA.txt','w')
    f.close()
    del f
    
    os.chdir(dir_string)  
    
    return model_json

def json_unwind(model_json):

    os.chdir(dir_string + '/Data/Temp_Object_Holder/JSON_Slush_Folder')

    text_file = open("text_file.json", "w")
    text_file.write(model_json)
    text_file.close() # save file to folder

    with open('text_file.json', 'r') as fin:
        model = model_from_json(json.load(fin))  # import and parse
        
    # clear folder
    mypath = dir_string + '/Data/Temp_Object_Holder/JSON_Slush_Folder'
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    f = open('gitplaceholderA.txt','w')
    f.close()
    del f
    
    os.chdir(dir_string)
    
    return model