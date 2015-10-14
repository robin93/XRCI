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

#print id_age_train.head(10)

#Merging the train files
id_time_train = pd.merge(id_time_labs_train,id_time_vitals_train,on = ['ID','TIME'],how = 'outer')
id_age_time_train = pd.merge(id_age_train,id_time_train,on =['ID'],how = 'outer')
train_df = pd.merge(id_age_time_train,id_label_train,on = ['ID'],how = 'outer')

print train_df.head(10)

columns = ['Age','Pulse','Mean_BP','Respiratory_rate','Temperature','AaDO2','LABEL']

modified_feature_train_matrix = pd.DataFrame(columns = columns)

def score_pulse(pulse):
	if pulse<=39:
		return 8
	elif pulse> 40 and pulse<50:
		return 5
	elif pulse>=50 and pulse<100:
		return 0
	elif pulse>=100 and pulse<110:
		return 1
	elif pulse>=110 and pulse<120:
		return 5
	elif pulse>=120 and pulse<140:
		return 7
	elif pulse>=140 and pulse <154:
		return 13
	else:
		return 17

def score_mean_blood_pressure(systolic_bp,diastolic_bp):
	if not pd.isnull(systolic_bp):
		if not pd.isnull(diastolic_bp):
			mean_bp = (systolic_bp + diastolic_bp)/2
			if mean_bp <= 39:
				return 23
			elif mean_bp>39 and mean_bp<60:
				return 15
			elif mean_bp>=60 and mean_bp<70:
				return 7
			elif mean_bp>=70 and mean_bp<80:
				return 6
			elif mean_bp>=80 and mean_bp<100:
				return 0
			elif mean_bp>=100 and mean_bp<120:
				return 4
			elif mean_bp>=120 and mean_bp<130:
				return 7
			elif mean_bp>=130 and mean_bp<140:
				return 9
			elif mean_bp>=140:
				return 10

def score_respiratory_rate(respiratory_rate):
	if respiratory_rate<=5:
		return 17
	elif respiratory_rate> 5 and respiratory_rate<=11:
		return 8
	elif respiratory_rate>11 and respiratory_rate<=13:
		return 7
	elif respiratory_rate>13 and respiratory_rate<=24:
		return 0
	elif respiratory_rate>24 and respiratory_rate<=34:
		return 6
	elif respiratory_rate>34 and respiratory_rate<=39:
		return 9
	elif respiratory_rate>39 and respiratory_rate<=49:
		return 11
	elif respiratory_rate>49:
		return 18

def score_temperature(temperature):
	temp_celcius = float((temperature-32)*5/9)
	if temp_celcius <=32.9:
		return 20
	elif temp_celcius>=33 and temp_celcius <=33.4:
	    return 16
	elif temp_celcius>33.4 and temp_celcius<=33.9:
	    return 13
	elif temp_celcius>33.9 and temp_celcius<=34.9:
	    return 8
	elif temp_celcius>34.9 and temp_celcius<=35.9:
	    return 2
	elif temp_celcius>35.9 and temp_celcius<=36.9:
	    return 0
	else:
	    return 4

def score_AaDO2_partial_pressure(fio2,paCO2,paO2):
	AaDO2 = ((7.13*float(fio2)) - ((float(paCO2))/0.8) - float(paO2))
	if AaDO2 <100:
		return 0
	elif AaDO2>=100 and AaDO2<250:
		return 7
	elif AaDO2>=250 and AaDO2<350:
		return 9
	elif AaDO2 >=350 and AaDO2<500:
		return 11
	else:
		return 14


modified_matrix_index = 0


Patients_list = pd.unique(train_df.ID)
for patient in Patients_list:
 	if not pd.isnull(patient):
 		print 'PatientID',patient

 		#Age info
 		Age_subdata = id_age_train[(id_age_train.ID == patient)]
 		Patient_age = int(Age_subdata.AGE)
 		#print "age is - ",Patient_age

 		if int((id_label_train[(id_label_train.ID == patient)]).LABEL) == 1:  #selecting patients who died
 			modified_matrix_index = modified_matrix_index + 1
 			print modified_matrix_index
 			print "This patient died"
 			print "Now fetching the subset the data of the patient from the main data"
			data_sub = train_df[(train_df.ID == patient)]
			print 'number of measurement of this patient is',len(data_sub)

			
 			#EXTRACTING MODIFIED FEATURES FROM NON-ICU DATA of the current patient
 			
 			non_icu_data = data_sub[(data_sub.ICU == 0)]   #extract data of the patient when he was NOT in ICU
 			print 'Pre ICU measurements - ',len(data_sub[(data_sub.ICU == 0)])

 			#Pulse modified feature for Non-ICU of current patient
 			pulse_data_outside_icu = data_sub.V3
 			score_list = []
 			for measurement in pulse_data_outside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_pulse(int(measurement)))
 			if not score_list:
 				pulse_non_icu_score = 0
 			else:
 				pulse_non_icu_score = min(score_list)

 			#Mean BP modified feature for Non-ICU of current patient
 			pressure_data = data_sub[['V1','V2']]
 			score_list = []
 			for index,row in pressure_data.iterrows():
 				systolic_bp = float(row['V1'])
 				diastolic_bp = float(row['V2'])
 				if not pd.isnull(systolic_bp):
 					if not pd.isnull(diastolic_bp):
 						mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
 						score_list.append(mean_bp)
 			if not score_list:
 				mean_bp_non_icu_score = 0
 			else:
 				mean_bp_non_icu_score = min(score_list)



 			#Temperature modified feature for Non-ICU of current patient
			temperature_data_outside_icu = data_sub.V6
			score_list_temperature = []
			for measurement in temperature_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_temperature.append(score_temperature(float(measurement)))
			if not score_list_temperature:
				temperature_non_icu_score = 0
			else:
				temperature_non_icu_score = min(score_list_temperature)
 			
 			#Respiratory Rate modified feature for Non-ICU of current patient
 			respiratory_rate_data_outside_icu = data_sub.V4
 			score_list = []
 			for measurement in respiratory_rate_data_outside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_respiratory_rate(int(measurement)))
 			if not score_list:
 				respiratory_rate_non_icu_score = 0
 			else:
 				respiratory_rate_non_icu_score = min(score_list)

 			#Partial Pressure of Oxygen modified feature for Non-ICU of current patient
 			#AaDO2 modified feature for Non-ICU of current patient
 			AaDO2_data = data_sub[['L20','L2','L3']]
 			score_list = []
 			for index,row in AaDO2_data.iterrows():
 				fio2 = float(row['L20'])
 				paCO2 = float(row['L2'])
 				paO2 = float(row['L3'])
 				if not pd.isnull(fio2):
 					if not pd.isnull(paCO2):
 						if not pd.isnull(paO2):
 							AaDO2_score = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
 							score_list.append(AaDO2_score)
 			if not score_list:
 				AaDO2_non_icu_score = 0
 			else:
 				AaDO2_non_icu_score = min(score_list)

 			#Hematocrit modified feature for Non-ICU of current patient
 			#WBC count modified feature for Non-ICU of current patient
 			#Serum Creatinine modified feature for Non-ICU of current patient
 			#Urine Output modified feature for Non-ICU of current patient
 			#Serun BUN modified feature for Non-ICU of current patient
 			#Serum Na modified feature for Non-ICU of current patient
 			#Serum Albumin modified feature for Non-ICU of current patient
 			#Serum Bilirubin modified feature for Non-ICU of current patient
 			#Serum Glucose modified feature for Non-ICU of current patient
 			


 			# #Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'LABEL',0)



 			#EXTRACTING MODIFIED FEATURES FROM ICU DATA OF THE CURRENT PATIENT
 			modified_matrix_index = modified_matrix_index +1
 			print modified_matrix_index
 			icu_data = data_sub[(data_sub.ICU == 1)]       #extract data of the patient when he was in IC
 			print 'ICU measurements - ',len(data_sub[(data_sub.ICU == 1)])

 			#Pulse modified feature for ICU of current patient
 			pulse_data_inside_icu = data_sub.V3
 			score_list = []
 			for measurement in pulse_data_inside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_pulse(int(measurement)))
 			if not score_list:
 				pulse_icu_score = pulse_non_icu_score
 			else:
 				pulse_icu_score = max(score_list)

 			#Mean BP modified feature for ICU of current patient
 			pressure_data = data_sub[['V1','V2']]
 			score_list = []
 			for index,row in pressure_data.iterrows():
 				systolic_bp = float(row['V1'])
 				diastolic_bp = float(row['V2'])
 				if not pd.isnull(systolic_bp):
 					if not pd.isnull(diastolic_bp):
 						mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
 						score_list.append(mean_bp)
 			if not score_list:
 				mean_bp_icu_score = mean_bp_non_icu_score
 			else:
 				mean_bp_icu_score = max(score_list)


 			#Temperature modified feature for ICU of current patient
			temperature_data_inside_icu = data_sub.V6
			score_list_temperature = []
			for measurement in temperature_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_temperature.append(score_temperature(float(measurement)))
			if not score_list_temperature:
				temperature_icu_score = temperature_non_icu_score
			else:
				temperature_icu_score = min(score_list_temperature)
 			#Respiratory Rate modified feature for ICU of current patient
 			respiratory_rate_data_inside_icu = data_sub.V4
 			score_list = []
 			for measurement in respiratory_rate_data_inside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_pulse(int(measurement)))
 			if not score_list:
 				respiratory_rate_icu_score = respiratory_rate_non_icu_score
 			else:
 				respiratory_rate_icu_score = max(score_list)

 			#Partial Pressure of Oxygen modified feature for ICU of current patient
 			#AaDO2 modified feature for ICU of current patient
 			AaDO2_data = data_sub[['L20','L2','L3']]
 			score_list = []
 			for index,row in AaDO2_data.iterrows():
 				fio2= float(row['L20'])
 				paCO2 = float(row['L2'])
 				paO2 = float(row['L3'])
 				if not pd.isnull(fio2):
 					if not pd.isnull(paCO2):
 						if not pd.isnull(paO2):
 							AaDO2_score = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
 							score_list.append(AaDO2_score)
 			if not score_list:
 				AaDO2_icu_score = AaDO2_non_icu_score
 			else:
 				AaDO2_icu_score = max(score_list)

 			#Hematocrit modified feature for ICU of current patient
 			#WBC count modified feature for ICU of current patient
 			#Serum Creatinine modified feature for ICU of current patient
 			#Urine Output modified feature for ICU of current patient
 			#Serun BUN modified feature for ICU of current patient
 			#Serum Na modified feature for ICU of current patient
 			#Serum Albumin modified feature for ICU of current patient
 			#Serum Bilirubin modified feature for ICU of current patient
 			#Serum Glucose modified feature for ICU of current patient

 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'LABEL',1)

 		else:   #selecting patients who survived
 			print "This patient did not die"
 			print "Now fetching the subset the data of the patient from the main data"
 			data_sub = train_df[(train_df.ID == patient)]
 			print 'number of measurement of this patient is',len(data_sub)
 			print 'Pre ICU measurements - ',len(data_sub[(data_sub.ICU == 0)])
 			
 			#EXTRACTING MODIFIED FEATURES FROM NON-ICU DATA of the current patient
 			modified_matrix_index = modified_matrix_index + 1
 			non_icu_data = data_sub[(data_sub.ICU == 0)]   #extract data of the patient when he was NOT in ICU
 			print 'Pre ICU measurements - ',len(data_sub[(data_sub.ICU == 0)])

 			#Pulse modified feature for Non-ICU of current patient
 			pulse_data_outside_icu = data_sub.V3
 			score_list = []
 			for measurement in pulse_data_outside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_pulse(int(measurement)))
 			if not score_list:
 				pulse_non_icu_score = 0
 			else:
 				pulse_non_icu_score = min(score_list)

 			#Mean BP modified feature for Non-ICU of current patient
 			pressure_data = data_sub[['V1','V2']]
 			score_list = []
 			for index,row in pressure_data.iterrows():
 				systolic_bp = float(row['V1'])
 				diastolic_bp = float(row['V2'])
 				if not pd.isnull(systolic_bp):
 					if not pd.isnull(diastolic_bp):
 						mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
 						score_list.append(mean_bp)
 			if not score_list:
 				mean_bp_non_icu_score = 0
 			else:
 				mean_bp_non_icu_score = min(score_list)

 			#Temperature modified feature for Non-ICU of current patient
			temperature_data_outside_icu = data_sub.V6
			score_list_temperature = []
			for measurement in temperature_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_temperature.append(score_temperature(float(measurement)))
			if not score_list_temperature:
				temperature_non_icu_score = 0
			else:
				temperature_non_icu_score = min(score_list_temperature)
 			#Respiratory Rate modified feature for Non-ICU of current patient
 			respiratory_rate_data_outside_icu = data_sub.V4
 			score_list = []
 			for measurement in respiratory_rate_data_outside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_respiratory_rate(int(measurement)))
 			if not score_list:
 				respiratory_rate_non_icu_score = 0
 			else:
 				respiratory_rate_non_icu_score = min(score_list)

 			#Partial Pressure of Oxygen modified feature for Non-ICU of current patient
 			#AaDO2 modified feature for Non-ICU of current patient
 			AaDO2_data = data_sub[['L20','L2','L3']]
 			score_list = []
 			for index,row in AaDO2_data.iterrows():
 				fio2= float(row['L20'])
 				paCO2 = float(row['L2'])
 				paO2 = float(row['L3'])
 				if not pd.isnull(fio2):
 					if not pd.isnull(paCO2):
 						if not pd.isnull(paO2):
 							AaDO2_score = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
 							score_list.append(AaDO2_score)
 			if not score_list:
 				AaDO2_icu_score = AaDO2_non_icu_score
 			else:
 				AaDO2_icu_score = min(score_list)

 			#Hematocrit modified feature for Non-ICU of current patient
 			#WBC count modified feature for Non-ICU of current patient
 			#Serum Creatinine modified feature for Non-ICU of current patient
 			#Urine Output modified feature for Non-ICU of current patient
 			#Serun BUN modified feature for Non-ICU of current patient
 			#Serum Na modified feature for Non-ICU of current patient
 			#Serum Albumin modified feature for Non-ICU of current patient
 			#Serum Bilirubin modified feature for Non-ICU of current patient
 			#Serum Glucose modified feature for Non-ICU of current patient

 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'LABEL',0)

 			#EXTRACTING MODIFIED FEATURES FROM ICU DATA OF THE CURRENT PATIENT
 			modified_matrix_index = modified_matrix_index + 1
 			icu_data = data_sub[(data_sub.ICU == 1)]       #extract data of the patient when he was in IC
 			print 'ICU measurements - ',len(data_sub[(data_sub.ICU == 1)])

 			#Pulse modified feature for ICU of current patient
 			pulse_data_inside_icu = data_sub.V3
 			score_list = []
 			for measurement in pulse_data_inside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_pulse(int(measurement)))
 			if not score_list:
 				pulse_icu_score = pulse_non_icu_score
 			else:
 				pulse_icu_score = max(score_list)

 			#Mean BP modified feature for ICU of current patient
 			pressure_data = data_sub[['V1','V2']]
 			score_list = []
 			for index,row in pressure_data.iterrows():
 				systolic_bp = float(row['V1'])
 				diastolic_bp = float(row['V2'])
 				if not pd.isnull(systolic_bp):
 					if not pd.isnull(diastolic_bp):
 						mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
 						score_list.append(mean_bp)
 			if not score_list:
 				mean_bp_icu_score = mean_bp_non_icu_score
 			else:
 				mean_bp_icu_score = max(score_list)

 			#Temperature modified feature for ICU of current patient
			temperature_data_inside_icu = data_sub.V6
			score_list_temperature = []
			for measurement in temperature_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_temperature.append(score_temperature(float(measurement)))
			if not score_list_temperature:
				temperature_icu_score = temperature_non_icu_score
			else:
				temperature_icu_score = min(score_list_temperature)
 			#Respiratory Rate modified feature for ICU of current patient
 			respiratory_rate_data_inside_icu = data_sub.V4
 			score_list = []
 			for measurement in respiratory_rate_data_inside_icu:
 				if not pd.isnull(measurement):
 					score_list.append(score_respiratory_rate(int(measurement)))
 			if not score_list:
 				respiratory_rate_icu_score = respiratory_rate_non_icu_score
 			else:
 				respiratory_rate_icu_score = max(score_list)


 			#Partial Pressure of Oxygen modified feature for ICU of current patient
 			#AaDO2 modified feature for ICU of current patient
 			AaDO2_data = data_sub[['L20','L2','L3']]
 			score_list = []
 			for index,row in AaDO2_data.iterrows():
 				fio2= float(row['L20'])
 				paCO2 = float(row['L2'])
 				paO2 = float(row['L3'])
 				if not pd.isnull(fio2):
 					if not pd.isnull(paCO2):
 						if not pd.isnull(paO2):
 							AaDO2_score = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
 							score_list.append(AaDO2_score)
 			if not score_list:
 				AaDO2_icu_score = AaDO2_non_icu_score
 			else:
 				AaDO2_icu_score = max(score_list)

 			#Hematocrit modified feature for ICU of current patient
 			#WBC count modified feature for ICU of current patient
 			#Serum Creatinine modified feature for ICU of current patient
 			#Urine Output modified feature for ICU of current patient
 			#Serun BUN modified feature for ICU of current patient
 			#Serum Na modified feature for ICU of current patient
 			#Serum Albumin modified feature for ICU of current patient
 			#Serum Bilirubin modified feature for ICU of current patient
 			#Serum Glucose modified feature for ICU of current patient

 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'LABEL',0)



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