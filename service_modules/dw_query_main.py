#-----------------------------------------------------------------------------#
# [1] Function loading data from SQL Server to Python 
#-----------------------------------------------------------------------------#

#pip install pyodbc
import pyodbc
import pandas as pd

def get_data():

    # Transform and load data from SQL Server to Python script 
    conn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};"
                          "Server=SQLTMWDWPRD2.trimac.com\TMWDWPRD;"
                          "Database=DATASTORE;"
                          "Trusted_Connection=yes;")
    cursor = conn.cursor()
    
    # Insert data  from SQL table into Python Pandas data frame
    sql_query = pd.read_sql_query('''
    DECLARE @datecol datetime = GETDATE();           
    DECLARE @WeekNum INT
          , @YearNum char(4);  
    SELECT @WeekNum = DATEPART(WK, @datecol)
         , @YearNum = CAST(DATEPART(YY, @datecol) AS CHAR(4));
    
    with LegDate as 
    (
    SELECT D.CALENDAR_DATE AS 'Date'
	   ,right('00' + CONVERT(VARCHAR(2),d.calendar_month_number),2) as CalendarMonth
	   ,d.calendar_year as 'CalendarYear'
       ,convert(varchar(4), d.calendar_year) + '-' + Right('00'+convert(varchar(2),d.week_number),2) as 'CalendarWeek'
    FROM dim_DATE_VW (NOLOCK) D
    ), 
    LegBranch as 
    (
    SELECT S.Terminal
	  ,S.TerminalName + ' ('+ S.Terminal + ')'  as 'AllocatedBranch'
      ,S.LineOfBusiness
      ,S.RegionName
      ,b.Terminal as 'Dept'
    FROM SSAS_Common_Terminal_vw (NOLOCK) B
     JOIN SSAS_Common_Terminal_vw (NOLOCK) S ON S.Terminal  = B.SweepTerminal
     )
    SELECT LegDATE.CalendarWeek, SUM(TOD.Travel_Miles) as Travel_Miles, LegBranch.Terminal, LegBranch.RegionName
    FROM [dbo].[SSAS_TripOrder_TripOrderDetail_vw] (NOLOCK) TOD
    LEFT JOIN LegDate ON LegDate.Date = TOD.LegEndDate
    LEFT JOIN LegBranch ON LegBranch.Dept = TOD.AllocatedTerminalCode
    WHERE LegOutStatus != 'Planned' 
    AND   Is_SInvoice = 'No' 
    AND   LegBranch.RegionName LIKE 'TL %' 
    AND   LegBranch.RegionName NOT LIKE '% NRT' 
    AND   LegDate.CalendarYear >= '2019'
    AND   LegDate.Date < DATEADD(wk, DATEDIFF(wk, 6, '1/1/' + @YearNum) + (@WeekNum-1), 6)
    GROUP BY LegDATE.CalendarWeek, LegBranch.Terminal, LegBranch.RegionName
    ORDER BY LegBranch.Terminal ASC, CalendarWeek ASC
    ''',conn)
    return sql_query

