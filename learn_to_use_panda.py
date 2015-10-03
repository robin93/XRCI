#http://blog.kaggle.com/2013/01/17/getting-started-with-pandas-predicting-sat-scores-for-new-york-city-schools/

import pandas as pd

dsProgReports = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_Progress_Reports_-_All_Schools_-_2009-10.csv')
dsDistrict = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_District_Breakdowns.csv')
dsClassSize = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/2009-10_Class_Size_-_School-level_Detail.csv')
dsAttendEnroll = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_Attendance_and_Enrollment_Statistics_by_District__2010-11_.csv')
dsSATs = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/SAT__College_Board__2010_School_Level_Results.csv')

dsSATs.info()

