import pandas as pd
import os
import xlsxwriter as xw

from service_modules.directory_string_mod import directory_string
dir_string = directory_string()


def report_export_fun(report_master):
    
    # set today's date
    today = str(pd.Timestamp.now().round('min'))
    today = today.replace(':','-',2)
    today = today[:-3]
    today = today[:10] + " T " + today[11:]
 
    # switch directories to where the file shoul dbe saved
    os.chdir(dir_string + '/Data/Main Output')
    
    # converting from list object to dataframe      
    report_master = pd.DataFrame(report_master,
                                  columns=['Branch','Best Model','Best Model Details',
                                          'Best Model MAPE:' + '\n' + '1- 10 weeks',
                                          'Best Model MAPE:' + '\n' + '11- 20 weeks',
                                          'Best Model MAPE:' + '\n' + '21- 30 weeks',
                                          'Best Model MAPE:' + '\n' + '31- 39 weeks',
                                          'Next Best Single Variable Model',
                                          'Next Best Single Variable Model Details',
                                          'Next Best Single Variable Model MAPE:' + '\n' + '1- 10 weeks',
                                          'Next Best Single Variable Model MAPE:' + '\n' + '11- 20 weeks',
                                          'Next Best Single Variable Model MAPE:' + '\n' + '21- 30 weeks',
                                          'Next Best Single Variable Model MAPE:' + '\n' + '31- 39 weeks',
                                          'Holiday Features',
                                          'Target Variable Transformation'])
         
    # creating a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter('Report-' + today + '.xlsx', engine='xlsxwriter')
    
    # Convert the dataframe to an XlsxWriter Excel object
    report_master.to_excel(writer, sheet_name = 'Report-' + today, index = False)
        
    # Get the xlsxwriter workbook and worksheet objects
    workbook  = writer.book
    worksheet = writer.sheets['Report-' + today]
    
    # creating some formats
    
    body_format1 = workbook.add_format({'text_wrap': True,
                                   'valign': 'top',
                                   'align': 'left'})
    
    header_format1 = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'border': 0,
        'align': 'center'})
    
    header_format2 = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'border': 0,
        'align': 'left'})
     
    # format body
    worksheet.set_column('A:A', 9, body_format1)
    worksheet.set_column('B:B', 13, body_format1)
    worksheet.set_column('C:C', 54, body_format1)
    worksheet.set_column('D:G', 10, body_format1)
    worksheet.set_column('H:H', 13, body_format1)
    worksheet.set_column('I:I', 48, body_format1)
    worksheet.set_column('J:M', 10, body_format1)
    worksheet.set_column('N:N', 16, body_format1)
    worksheet.set_column('O:O', 16, body_format1)

    # format header
    for col_num, value in enumerate(report_master.columns.values):
        worksheet.write(0, col_num, value, header_format1)
    
    worksheet.write(0, 0, 'Branch', header_format2)
    
    # export the Excel fule and close the pandas Excel writer
    
    writer.save()
    writer.close()
    
    os.chdir(dir_string)