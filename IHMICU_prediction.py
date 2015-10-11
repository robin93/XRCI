"""
read training files
read validation files 

treatment on the training dataset
    Join tables
    check foreign key constraints
    check for missing values
    
treatment on the test dataset
    join tables
    check foreign key constraints
    check for missing values

for all rows in training set:
    if timestamp in ICU:
        if first timestamp in ICU:
            Modified features based on the current timestamp and previous non-icu timestamp
        if non-first timestamp in ICU:
            Modified features based on the current timestamp values
            Get the data for previous non-ICU timestamps of the patient
            Modified features based on the non-current timestamp values
            
for all rows in validation set:(Non-ICU timestamps should not be present in the output)
    if timestamp in ICU:
        if first timestamp in ICU:
            Modified features based on the current timestamp and previous non-icu timestamp
        if non-first timestamp in ICU:
            Modified features based on the current timestamp values
            Get the data for previous non-ICU timestamps of the patient
            Modified features based on the non-current timestamp values
            

Train random forest classifier

run the classifier on the validation data


Sensitivity,specificity and accuracy calculation:
    Difference of predicted and actual values:
       0 and predicted = 1->True positive++
       0 and predicted = 0 ->True negative++
       1 ->False positive++
      -1->False negative++
    Sensitivity = True positive/(True positive + False negative)
    Specificity = True negative/(True negative + False positive)
    accuracy = (True positive + True negative) /(True negative + False negative + True positive + True negative)
    median time =median[for all patients in test]    
             
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv as csv
import os

cwd = os.getcwd()

#Reading the train files
id_age_train = pd.read_csv(os.path.join(cwd,'id_age_train.csv'))
id_label_train = pd.read_csv(os.path.join(cwd,'id_label_train.csv'))
id_time_labs_train = pd.read_csv(os.path.join(cwd,'id_time_labs_train.csv'))
id_time_vitals_train = pd.read_csv(os.path.join(cwd,'id_time_vitals_train.csv'))


#Merging the train files
id_time_train = pd.merge(id_time_labs_train,id_time_vitals_train,on = ['ID','Time'],how = 'outer')
id_age_time_train = pd.merge(id_age_train,id_time_train,on =['ID'],how = 'outer')
train_df = pd.merge(id_age_time_train,id_label_train,on = ['ID'],how = 'outer')

#print train_df.head(10)

index = []

columns = ['first_modified_feature']

modified_feature_train_matrix = pd.DataFrame(columns = columns)

#Iterate over rows and print rows
modified_matrix_index = 0
for index, row in train_df.iterrows():
    if row['ICU'] == 1:
    	modified_matrix_index = modified_matrix_index + 1
    	if len((train_df[(train_df.ID == row['ID']) & (train_df.Time == row['Time'])]).index) < 0:
    		modified_feature_train_matrix.set_value(modified_matrix_index,'first_modified_feature',0)
    	else:
    		modified_feature_train_matrix.set_value(modified_matrix_index,'first_modified_feature',1)

print modified_feature_train_matrix

# #Reading the validation files
# id_age_val = pd.read_csv(os.path.join(cwd,'id_age_val.csv'))
# id_label_val = pd.read_csv(os.path.join(cwd,'.id_label_val.csv'))
# id_time_labs_val = pd.read_csv(os.path.join(cwd,'id_time_labs_val.csv'))
# id_time_vitals_val = pd.read_csv(os.path.join(cwd,'id_time_vitals_val.csv'))

# #Merging the validation files
# id_time_val = pd.merge(id_time_labs_val,id_time_vitals_val,on = ['ID','Time'],how = 'outer')
# id_age_time_val = pd.merge(id_age_val,id_time_val,on =['ID'],how = 'outer')
# val_df = pd.merge(id_age_time_val,id_label_val,on = ['ID'],how = 'outer')

    
# #Reading the test data files
# id_age_test = pd.read_csv(os.path.join(cwd,'id_age_test.csv'))
# id_time_labs_test = pd.read_csv(os.path.join(cwd,'id_time_labs_test.csv'))
# id_time_vitals_test = pd.read_csv(os.path.join(cwd,'id_time_vitals_test.csv'))


#print id_age_train.head(10)
#print id_label_test.head(10)
#print id_time_labs_train.head(10)

#joining tables

#checking foreign key constraint 

#checking for missing values


#code block to create modified features for training data
#http://www.datacarpentry.org/python-ecology/02-index-slice-subset
#Iterate over rows and print rows
# for index, row in val_df.iterrows():
#     for column in val_df.columns:
#         print row[column],
#     print

#code block to create modified features for the test data



# #remove name of the columns from the modified train before feeding it to the classifier
# modified_train_df = modified_train_df.drop(['column_name1','Column_name2'],axis=1)

# #remove name of the columns from the modified validation before feeding it to the classifier
# modified_val_df = modified_val_df.drop(['column_name1','Column_name2'],axis=1)

# #the data now is ready to be fed to the classifier
# #converting data back to the numpy array
# train_data = modified_train_df.values
# test_data = modified_val_df.values

# print "Training....."
# forest = RandomForestClassifier(n_estimators = 10)
# forest = forest.fit( train_data[0::,1::], train_data[0::,0])

# print "predicting..."
# output = forest.predict(test_data).astype(int)

# predictions_file = open("output_prediction.csv","wb")
# open_file_object = csv.writer(prediction_file)
# open_file_object.writerow(["Patient_id","Time_Stamp","Output_Prediction"])
# open_file_object.writerows(zip(Patient_id,Time_Stamp,Output_Prediction))
# predictions_file.close()
# print "Done"

# #calculating scores for the output
# #print "calculating scores"

# #Output_file = pd.read_csv(os.path.join(cwd,'output_prediction.csv'))
# #check_file = pd.merge(Output_file,id,on = ['ID','Time'],how = 'outer')











 