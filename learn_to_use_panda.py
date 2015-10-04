# #http://blog.kaggle.com/2013/01/17/getting-started-with-pandas-predicting-sat-scores-for-new-york-city-schools/

# import pandas as pd

# dsProgReports = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_Progress_Reports_-_All_Schools_-_2009-10.csv')
# dsDistrict = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_District_Breakdowns.csv')
# dsClassSize = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/2009-10_Class_Size_-_School-level_Detail.csv')
# dsAttendEnroll = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/School_Attendance_and_Enrollment_Statistics_by_District__2010-11_.csv')
# dsSATs = pd.read_csv('/home/robin/Desktop/CDS/XRCI/NYC_Schools/SAT__College_Board__2010_School_Level_Results.csv')

# dsSATs.info()

# # dsSATs join dsClassSize on dsSATs['DBN'] = dsClassSize['SCHOOL CODE']
# # join dsProgReports on dsSATs['DBN'] = dsProgReports['DBN']
# # join dsDistrict on dsProgReports['DISTRICT'] = dsDistrict['JURISDICTION NAME']
# # join dsAttendEnroll on dsProgReports['DISTRICT'] = dsAttendEnroll['DISTRICT']

# #strip the first 2 letters of the DBN in the data
# dsProgReports.DBN = dsProgReports.DBN.map(lambda x: x[2:])
# dsSATs.DBN = dsSATs.DBN.map(lambda x: x[2:])

# #viewing an instance of data
# print pd.DataFrame(data=[dsProgReports['DBN'].take(range(20)), dsSATs['DBN'].take(range(20)), dsClassSize['SCHOOL CODE'].take(range(20))])

# # import re
# # dsDistrict['JURISDICTION NAME'] = dsDistrict['JURISDICTION NAME'].map(lambda x: re.match( r'([A-Za-z]*\s)([0-9]*)',x).group(2)).astype(int) 
# # dsAttendEnroll.District = dsAttendEnroll.District.map(lambda x: x[-2:]).astype(int)


# print pd.DataFrame(data=[dsProgReports['DISTRICT'][:3], dsDistrict['JURISDICTION NAME'][:3], dsAttendEnroll['District'][:3]])



#http://www.analyticsvidhya.com/blog/2014/08/baby-steps-python-performing-exploratory-analysis-python/
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot #has to be imported separately

df = pd.read_csv('/home/robin/Desktop/CDS/XRCI/titanic/train.csv')

df.info()  #summary of the contents of the data
#print df
print df.head(10)
print df.describe()
print "median of age",df['Age'].median()
print 'Unique values of sex',df['Sex'].unique()

# #ploting the histogram of the Age to check the distribution
# fig = plt.pyplot.figure()
# ax = fig.add_subplot(111)
# ax.hist(df['Age'],bins=10,range = (df['Age'].min(),df['Age'].max()))
# plt.pyplot.title('Age Distribution')
# plt.pyplot.xlabel('Age')
# plt.pyplot.ylabel('Count of passengers')
# plt.pyplot.show()

# #ploting the histogram of the fare to check the distribution
# fig = plt.pyplot.figure()
# ax = fig.add_subplot(111)
# ax.hist(df['Fare'],bins=10,range = (df['Fare'].min(),df['Fare'].max()))
# plt.pyplot.title('Fare Distribution')
# plt.pyplot.xlabel('Fare')
# plt.pyplot.ylabel('Count of passengers')
# plt.pyplot.show()

# #check box plot for fare column
# df.boxplot(column='Fare')
# plt.pyplot.show()
# df.boxplot(column='Fare', by = 'Pclass')
# plt.pyplot.show()


#understanding the distribution of the survival ratio on the categorical variables
temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.pyplot.figure(figsize =(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of passengers')
ax1.set_title('Passengers by Pclass')
temp1.plot(kind='bar')
#plt.pyplot.show()

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")
#plt.pyplot.show()

temp3 = pd.crosstab([df.Pclass, df.Sex], df.Survived.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.pyplot.show()





