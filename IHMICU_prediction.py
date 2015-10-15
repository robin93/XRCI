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

columns = ['Age','Pulse','Mean_BP','Respiratory_rate','Temperature','PaO2','AaDO2','Hematocrit','Urine_output','Bilirubin','WBC','BUN','Sodium','Serum_glucose','Serum_creatinine','Albumin_output','LABEL']

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

def score_PaO2(PaO2):
	if PaO2<=49:
		return 15
	elif PaO2>49 and PaO2<=69:
		return 5
	elif PaO2>69 and PaO2<=79:
		return 2
	elif PaO2>79:
		return 0

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

def score_Hematocrit(Hematocrit):
	if Hematocrit <=40.9:
		return 3
	elif Hematocrit >40.9 and Hematocrit <=49:
		return 0
	elif Hematocrit >49:
		return 3

def score_Urine_output(Urine_output):
	if Urine_output <=399:
		return 15
	elif Urine_output > 399 and Urine_output <=599:
		return 8
	elif Urine_output >599 and Urine_output <= 899:
		return 7
	elif Urine_output >899 and Urine_output <= 1499:
		return 5
	elif Urine_output >1499 and Urine_output <=1999:
		return 4
	elif Urine_output >1999 and Urine_output <4000:
		return 0
	else:
		return 1

def score_bilirubin(bilirubin):
	if bilirubin<=1.9:
		return 0
	elif bilirubin>1.9 and bilirubin<=2.9:
		return 5
	elif bilirubin>2.9 and bilirubin<=4.9:
		return 6
	elif bilirubin>4.9 and bilirubin<=7.9:
		return 8
	else:
		return 16

def score_WBC(WBC):
	if WBC<1:
		return 19
	elif WBC>=1 and WBC<3:
		return 5
	elif WBC>=3 and WBC<20:
		return 0
	elif WBC>=20 and WBC<25:
		return 1
	else:
	 return 5

def score_BUN(BUN):
	if BUN<17:
		return 0
	elif BUN>=17 and BUN<20:
		return 2
	elif BUN>=20 and BUN<40:
		return 7
	elif BUN>=40 and BUN<80:
		return 11
	else:
	 return 12

def score_Sodium(Sodium):
	if Sodium<120:
		return 3
	elif Sodium>=120 and Sodium<135:
		return 2
	elif Sodium>=135 and Sodium<155:
		return 0
	else:
	 return 4

def score_serum_glucose(glucose):
	if glucose<=39:
		return 8
	elif glucose>39 and glucose<=59:
		return 9
	elif glucose>59 and glucose<=199:
		return 0
	elif glucose>199 and glucose<=349:
		return 3
	else:
		return 5

def score_serum_creatinine(serum_creatinine):
	if serum_creatinine<=0.4:
		return 3
	elif serum_creatinine>0.4 and serum_creatinine<=1.4:
		return 0
	elif serum_creatinine>1.4 and serum_creatinine<=1.94:
		return 4
	else:
		return 7

def score_Albumin_output(Albumin):
	if Albumin<2:
		return 11
	elif Albumin>=2 and Albumin<2.5:
		return 6
	elif Albumin>=2.5 and Albumin<4.5:
		return 0
	else:
	 return 4  

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
 			PaO2_data_outside_icu = data_sub.L3
			score_list = []
			for measurement in PaO2_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_PaO2(int(measurement)))
			if not score_list:
				PaO2_non_icu_score = 0
			else:
				PaO2_non_icu_score = min(score_list)


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
 			Hematocrit_data_outside_icu = data_sub.L10
			score_list_Hematocrit = []
			for measurement in Hematocrit_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Hematocrit.append(score_Hematocrit(float(measurement)))
			if not score_list_Hematocrit:
				Hematocrit_non_icu_score = 0
			else:
				Hematocrit_non_icu_score = min(score_list_Hematocrit)

 			#WBC count modified feature for Non-ICU of current patient
 			WBC_data_outside_icu = data_sub.L9
			score_list = []
			for measurement in WBC_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_WBC(int(measurement)))
			if not score_list:
				WBC_non_icu_score = 0
			else:
				WBC_non_icu_score = min(score_list)

 			#Serum Creatinine modified feature for Non-ICU of current patient
 			Serum_creatinine_data_outside_icu = data_sub.L8
			score_list = []
			for measurement in Serum_creatinine_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_creatinine(int(measurement)))
			if not score_list:
				Serum_creatinine_non_icu_score = 0
			else:
				Serum_creatinine_non_icu_score = min(score_list)

 			#Urine Output modified feature for Non-ICU of current patient
 			Urine_output_data_outside_icu = data_sub.L13
			score_list_Urine_output = []
			for measurement in Urine_output_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Urine_output.append(score_Urine_output(float(measurement)))
			if not score_list_Urine_output:
				Urine_output_non_icu_score = 0
			else:
				Urine_output_non_icu_score = min(score_list_Urine_output)
 			#Serun BUN modified feature for Non-ICU of current patient
 			BUN_data_outside_icu = data_sub.L7
			score_list = []
			for measurement in BUN_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_BUN(int(measurement)))
			if not score_list:
				BUN_non_icu_score = 0
			else:
				BUN_non_icu_score = min(score_list)

 			#Serum Na modified feature for Non-ICU of current patient
 			Sodium_data_outside_icu = data_sub.L4
			score_list = []
			for measurement in Sodium_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_Sodium(int(measurement)))
			if not score_list:
				Sodium_non_icu_score = 0
			else:
				Sodium_non_icu_score = min(score_list)
 			
#Serum Albumin modified feature for Non-ICU of current patient
			Albumin_output_data_outside_icu = data_sub.L21
			score_list_Albumin_output = []
			for measurement in Albumin_output_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Albumin_output.append(score_Albumin_output(float(measurement)))
			if not score_list_Albumin_output:
				Albumin_output_non_icu_score = 0
			else:
				Albumin_output_non_icu_score = min(score_list_Albumin_output)

 			#Serum Bilirubin modified feature for Non-ICU of current patient
 			Bilirubin_data_outside_icu = data_sub.L12
			score_list = []
			for measurement in Bilirubin_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_bilirubin(int(measurement)))
			if not score_list:
				Bilirubin_non_icu_score = 0
			else:
				Bilirubin_non_icu_score = min(score_list)


 			#Serum Glucose modified feature for Non-ICU of current patient
 			Serum_glucose_data_outside_icu = data_sub.L19
			score_list = []
			for measurement in Serum_glucose_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_glucose(int(measurement)))
			if not score_list:
				Serum_glucose_non_icu_score = 0
			else:
				Serum_glucose_non_icu_score = min(score_list)


 			# #Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'PaO2',PaO2_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Hematocrit',Hematocrit_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Urine_output',Urine_output_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Bilirubin',Bilirubin_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'WBC',WBC_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'BUN',BUN_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Sodium',Sodium_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_glucose',Serum_glucose_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_creatinine',Serum_creatinine_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Albumin_output',Albumin_output_non_icu_score)
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
 			PaO2_data_inside_icu = data_sub.L3
			score_list = []
			for measurement in PaO2_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_PaO2(int(measurement)))
			if not score_list:
				PaO2_icu_score = PaO2_non_icu_score
			else:
				PaO2_icu_score = max(score_list)

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
 			Hematocrit_data_inside_icu = data_sub.L10
			score_list_Hematocrit = []
			for measurement in Hematocrit_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Hematocrit.append(score_Hematocrit(float(measurement)))
			if not score_list_Hematocrit:
				Hematocrit_icu_score = Hematocrit_non_icu_score
			else:
				Hematocrit_icu_score = max(score_list_Hematocrit)

 			#WBC count modified feature for ICU of current patient
 			WBC_data_inside_icu = data_sub.L9
			score_list = []
			for measurement in WBC_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_WBC(int(measurement)))
			if not score_list:
				WBC_icu_score = WBC_non_icu_score
			else:
				WBC_icu_score = max(score_list)

 			#Serum Creatinine modified feature for ICU of current patient
 			Serum_creatinine_data_icu = data_sub.L8
			score_list = []
			for measurement in Serum_creatinine_data_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_creatinine(int(measurement)))
			if not score_list:
				Serum_creatinine_icu_score = Serum_creatinine_non_icu_score
			else:
				Serum_creatinine_icu_score = max(score_list)

 			#Urine Output modified feature for ICU of current patient
 			Urine_output_data_inside_icu = data_sub.L13
			score_list_Urine_output = []
			for measurement in Urine_output_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Urine_output.append(score_Urine_output(float(measurement)))
			if not score_list_Urine_output:
				Urine_output_icu_score = Urine_output_non_icu_score
			else:
				Urine_output_icu_score = max(score_list_Urine_output)

 			#Serun BUN modified feature for ICU of current patient
 			BUN_data_inside_icu = data_sub.L7
			score_list = []
			for measurement in BUN_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_BUN(int(measurement)))
			if not score_list:
				BUN_icu_score = BUN_non_icu_score
			else:
				BUN_icu_score = max(score_list)

 			#Serum Na modified feature for ICU of current patient
 			Sodium_data_inside_icu = data_sub.L4
			score_list = []
			for measurement in Sodium_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_Sodium(int(measurement)))
			if not score_list:
				Sodium_icu_score = Sodium_non_icu_score
			else:
				Sodium_icu_score = max(score_list)

 			#Serum Albumin modified feature for ICU of current patient
			Albumin_output_data_inside_icu = data_sub.L21
			score_list_Albumin_output = []
			for measurement in Albumin_output_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Albumin_output.append(score_Albumin_output(float(measurement)))
			if not score_list_Albumin_output:
				Albumin_output_icu_score = Albumin_output_non_icu_score
			else:
				Albumin_output_icu_score = min(score_list_Albumin_output)

 			#Serum Bilirubin modified feature for ICU of current patient
 			Bilirubin_data_icu = data_sub.L12
			score_list = []
			for measurement in Bilirubin_data_icu:
				if not pd.isnull(measurement):
					score_list.append(score_bilirubin(int(measurement)))
			if not score_list:
				Bilirubin_icu_score = Bilirubin_non_icu_score
			else:
				Bilirubin_icu_score = max(score_list)

 			#Serum Glucose modified feature for ICU of current patient
 			Serum_glucose_data_icu = data_sub.L19
			score_list = []
			for measurement in Serum_glucose_data_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_glucose(int(measurement)))
			if not score_list:
				Serum_glucose_icu_score = Serum_glucose_non_icu_score
			else:
				Serum_glucose_icu_score = max(score_list)

 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'PaO2',PaO2_icu_score)		
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Hematocrit',Hematocrit_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Urine_output',Urine_output_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Bilirubin',Bilirubin_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'WBC',WBC_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'BUN',BUN_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Sodium',Sodium_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_glucose',Serum_glucose_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_creatinine',Serum_creatinine_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Albumin_output',Albumin_output_icu_score)
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
 			PaO2_data_outside_icu = data_sub.L3
			score_list = []
			for measurement in PaO2_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_PaO2(int(measurement)))
			if not score_list:
				PaO2_non_icu_score = 0
			else:
				PaO2_non_icu_score = min(score_list)

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
 			Hematocrit_data_outside_icu = data_sub.L10
			score_list_Hematocrit= []
			for measurement in Hematocrit_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Hematocrit.append(score_Hematocrit(float(measurement)))
			if not score_list_Hematocrit:
				Hematocrit_non_icu_score = 0
			else:
				Hematocrit_non_icu_score = min(score_list_Hematocrit)

 			#WBC count modified feature for Non-ICU of current patient
 			BC_data_outside_icu = data_sub.L9
			score_list = []
			for measurement in WBC_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_WBC(int(measurement)))
			if not score_list:
				WBC_non_icu_score = 0
			else:
				WBC_non_icu_score = min(score_list)

 			#Serum Creatinine modified feature for Non-ICU of current patient
 			Serum_creatinine_data_outside_icu = data_sub.L8
			score_list = []
			for measurement in Serum_creatinine_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_creatinine(int(measurement)))
			if not score_list:
				Serum_creatinine_non_icu_score = 0
			else:
				Serum_creatinine_non_icu_score = min(score_list)

 			#Urine Output modified feature for Non-ICU of current patient
 			Urine_output_data_outside_icu = data_sub.L13
			score_list_Urine_output = []
			for measurement in Urine_output_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Urine_output.append(score_Urine_output(float(measurement)))
			if not score_list_Urine_output:
				Urine_output_non_icu_score = 0
			else:
				Urine_output_non_icu_score = min(score_list_Urine_output)

 			#Serun BUN modified feature for Non-ICU of current patient
 			BUN_data_outside_icu = data_sub.L7
			score_list = []
			for measurement in BUN_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_BUN(int(measurement)))
			if not score_list:
				BUN_non_icu_score = 0
			else:
				BUN_non_icu_score = min(score_list)

 			#Serum Na modified feature for Non-ICU of current patient
 			Sodium_data_outside_icu = data_sub.L4
			score_list = []
			for measurement in Sodium_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_Sodium(int(measurement)))
			if not score_list:
				Sodium_non_icu_score = 0
			else:
				Sodium_non_icu_score = min(score_list)

 			#Serum Albumin modified feature for Non-ICU of current patient
			Albumin_output_data_outside_icu = data_sub.L21
			score_list_Albumin_output = []
			for measurement in Albumin_output_data_outside_icu:
				if not pd.isnull(measurement):
					score_list_Albumin_output.append(score_Albumin_output(float(measurement)))
			if not score_list_Albumin_output:
				Albumin_output_non_icu_score = 0
			else:
				Albumin_output_non_icu_score = min(score_list_Albumin_output)

 			#Serum Bilirubin modified feature for Non-ICU of current patient
 			Bilirubin_data_outside_icu = data_sub.L12
			score_list = []
			for measurement in Bilirubin_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_bilirubin(int(measurement)))
			if not score_list:
				Bilirubin_non_icu_score = 0
			else:
				Bilirubin_non_icu_score = min(score_list)
 			#Serum Glucose modified feature for Non-ICU of current patient
 			Serum_glucose_data_outside_icu = data_sub.L19
			score_list = []
			for measurement in Serum_glucose_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_glucose(int(measurement)))
			if not score_list:
				Serum_glucose_non_icu_score = 0
			else:
				Serum_glucose_non_icu_score = min(score_list)


 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_non_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'PaO2',PaO2_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Hematocrit',Hematocrit_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Urine_output',Urine_output_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Bilirubin',Bilirubin_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'WBC',WBC_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'BUN',BUN_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Sodium',Sodium_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_glucose',Serum_glucose_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_creatinine',Serum_creatinine_non_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Albumin',Albumin_output_non_icu_score)
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
				temperature_icu_score = max(score_list_temperature)
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
 			PaO2_data_inside_icu = data_sub.L3
			score_list = []
			for measurement in PaO2_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_PaO2(int(measurement)))
			if not score_list:
				PaO2_icu_score = PaO2_non_icu_score
			else:
				PaO2_icu_score = max(score_list)

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
 			Hematocrit_data_inside_icu = data_sub.L10
			score_list_Hematocrit = []
			for measurement in Hematocrit_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Hematocrit.append(score_temperature(float(measurement)))
			if not score_list_Hematocrit:
				Hematocrit_icu_score = Hematocrit_non_icu_score
			else:
				Hematocrit_icu_score = max(score_list_Hematocrit)

 			#WBC count modified feature for ICU of current patient
 			WBC_data_inside_icu = data_sub.L9
			score_list = []
			for measurement in WBC_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_WBC(int(measurement)))
			if not score_list:
				WBC_icu_score = WBC_non_icu_score
			else:
				WBC_icu_score = max(score_list)

 			#Serum Creatinine modified feature for ICU of current patient
 			Serum_creatinine_data_icu = data_sub.L8
			score_list = []
			for measurement in Serum_creatinine_data_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_creatinine(int(measurement)))
			if not score_list:
				Serum_creatinine_icu_score = Serum_creatinine_non_icu_score
			else:
				Serum_creatinine_icu_score = max(score_list)

 			#Urine Output modified feature for ICU of current patient
 			Urine_output_data_inside_icu = data_sub.L13
			score_list_Urine_output = []
			for measurement in Urine_output_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Urine_output.append(score_Urine_output(float(measurement)))
			if not score_list_Urine_output:
				Urine_output_icu_score = Urine_output_non_icu_score
			else:
				Urine_output_icu_score = max(score_list_Urine_output)
 			#Serun BUN modified feature for ICU of current patient
 			BUN_data_inside_icu = data_sub.L7
			score_list = []
			for measurement in BUN_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_BUN(int(measurement)))
			if not score_list:
				BUN_icu_score = BUN_non_icu_score
			else:
				BUN_icu_score = max(score_list)
 			#Serum Na modified feature for ICU of current patient
 			Sodium_data_inside_icu = data_sub.L4
			score_list = []
			for measurement in Sodium_data_inside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_Sodium(int(measurement)))
			if not score_list:
				Sodium_icu_score = Sodium_non_icu_score
			else:
				Sodium_icu_score = max(score_list)

 			#Serum Albumin modified feature for ICU of current patient
			Albumin_output_data_inside_icu = data_sub.L21
			score_list_Albumin_output = []
			for measurement in Albumin_output_data_inside_icu:
				if not pd.isnull(measurement):
					score_list_Albumin_output.append(score_Albumin_output(float(measurement)))
			if not score_list_Albumin_output:
				Albumin_output_icu_score = Albumin_output_non_icu_score
			else:
				Albumin_output_icu_score = min(score_list_Albumin_output)

 			#Serum Bilirubin modified feature for ICU of current patient
 			Bilirubin_data_outside_icu = data_sub.L12
			score_list = []
			for measurement in Bilirubin_data_outside_icu:
				if not pd.isnull(measurement):
					score_list.append(score_bilirubin(int(measurement)))
			if not score_list:
				Bilirubin_icu_score = Bilirubin_non_icu_score
			else:
				Bilirubin_icu_score = max(score_list)
 			
 			#Serum Glucose modified feature for ICU of current patient
 			Serum_glucose_data_icu = data_sub.L19
			score_list = []
			for measurement in Serum_glucose_data_icu:
				if not pd.isnull(measurement):
					score_list.append(score_serum_glucose(int(measurement)))
			if not score_list:
				Serum_glucose_icu_score = Serum_glucose_non_icu_score
			else:
				Serum_glucose_icu_score = max(score_list)

 			#Appending the modified non-ICU features for this patient
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Age',Patient_age)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Pulse',pulse_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Mean_BP',mean_bp_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'Respiratory_rate',respiratory_rate_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Temperature',temperature_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'PaO2',PaO2_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'AaDO2',AaDO2_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Hematocrit',Hematocrit_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Urine_output',Urine_output_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Bilirubin',Bilirubin_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'WBC',WBC_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'BUN',BUN_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Sodium',Sodium_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_glucose',Serum_glucose_icu_score)
			modified_feature_train_matrix.set_value(modified_matrix_index,'Serum_creatinine',Serum_creatinine_icu_score)
  			modified_feature_train_matrix.set_value(modified_matrix_index,'Albumin_output',Albumin_output_icu_score)
 			modified_feature_train_matrix.set_value(modified_matrix_index,'LABEL',0)



print modified_feature_train_matrix


print "Modified features creation done on the training data"

print "Reading the validation files"
#Reading the validation files
id_age_val = pd.read_csv(os.path.join(cwd,'id_age_val.csv'))
id_time_labs_val = pd.read_csv(os.path.join(cwd,'id_time_labs_val.csv'))
id_time_vitals_val = pd.read_csv(os.path.join(cwd,'id_time_vitals_val.csv'))

print "Merging the validation files"
#Merging the validation files
id_time_val = pd.merge(id_time_labs_val,id_time_vitals_val,on = ['ID','TIME'],how = 'outer')
val_df = pd.merge(id_age_val,id_time_val,on =['ID'],how = 'outer')


print val_df.head(10)

columns = ['ID','TIME','Age','Pulse','Mean_BP','Temperature','AaDO2']

modified_feature_val_matrix = pd.DataFrame(columns = columns)

modified_val_matrix_index = 0


for index,row in val_df.iterrows():
	if float(row['ICU']) == 1:
		modified_val_matrix_index = modified_val_matrix_index + 1

		#extracting and filling the patient id and timestamp information
		patient_id = int(row['ID'])
		timestamp = int(row['TIME'])
		age = int(row['Age'])
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'ID',patient_id)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'TIME',timestamp)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Age',age)


		#Pulse modified feature
		pulse_current_value = float(row['V3'])   #extract current value
		if pd.isnull(pulse_current_value):       #check if current value is null or not
			if timestamp>0:                      #if it is null then timestamp
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				pulse_data_history = data_sub.V3
				if timestamp < (3600*24):
					score_list = []
					for measurement in pulse_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_pulse(int(measurement)))
					if not score_list:
						pulse_score_value = 0
					else:
						pulse_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					pulse_data_history = data_sub.V3
					score_list = []
					for measurement in pulse_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_pulse(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						pulse_data_history = data_sub.V3
						score_list = []
						for measurement in pulse_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_pulse(int(measurement)))
						if not score_list:
							pulse_score_value = 0
						else:
							pulse_score_value = max(score_list)
					else:
						pulse_score_value = max(score_list)
			else:
				pulse_score_value = 0
		else:
			pulse_score_value = score_pulse(pulse_current_value)


		#Mean BP modified feature
		systolic_current_value = float(row['V1'])
		diastolic_current_value = float(row['V2'])
		if ((pd.isnull(systolic_current_value)) or (pd.isnull(diastolic_current_value))):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				bp_data_history = data_sub[['V1','V2']]
				if timestamp < (3600*24):
					score_list = []
					for index,row_1 in bp_data_history.iterrows():
						systolic_bp = float(row_1['V1'])
						diastolic_bp = float(row_1['V2'])
						if (pd.isnull(systolic_bp)) or (pd.isnull(diastolic_bp)):
							score_list.append(0)
						else:
							mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
							score_list.append(mean_bp)
						if not score_list:
							mean_bp_score = 0
						else:
							mean_bp_score = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					bp_data_history = data_sub[['V1','V2']]
					score_list = []
					for index,row_2 in bp_data_history.iterrows():
						systolic_bp = float(row_2['V1'])
						diastolic_bp = float(row_2['V2'])
						if (pd.isnull(systolic_bp)) or (pd.isnull(diastolic_bp)):
							score_list.append(0)
						else:
							mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
							score_list.append(mean_bp)
						if not score_list:
							data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
							bp_data_history = data_sub[['V1','V2']]
							score_list = []
							for index,row in bp_data_history.iterrows():
								systolic_bp = float(row_2['V1'])
								diastolic_bp = float(row_2['V2'])
								if (pd.isnull(systolic_bp)) or (pd.isnull(diastolic_bp)):
									score_list.append(0)
								else:
									mean_bp = score_mean_blood_pressure(systolic_bp,diastolic_bp)
									score_list.append(mean_bp)
							if not score_list:
								mean_bp_score = 0
							else:
								mean_bp_score = max(score_list)
						else:
							mean_bp_score = max(score_list)
			else:
				mean_bp_score = 0
		else:
			mean_bp_score = score_mean_blood_pressure(systolic_current_value,diastolic_current_value)

		#Temperature modified feature
		temperature_current_value = float(row['V6'])
		if pd.isnull(temperature_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				temperature_data_history = data_sub.V6
				if timestamp < (3600*24):
					score_list = []
					for measurement in temperature_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_temperature(float(measurement)))
					if not score_list:
						temperature_score_value = 0
					else:
						temperature_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					temperature_data_history = data_sub.V6
					score_list = []
					for measurement in temperature_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_temperature(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						temperature_data_history = data_sub.V6
						score_list = []
						for measurement in temperature_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_temperature(float(measurement)))
						if not score_list:
							temperature_score_value = 0
						else:
							temperature_score_value = max(score_list)
					else:
						temperature_score_value = max(score_list)
			else:
				temperature_score_value = 0
		else:
			temperature_score_value = score_temperature(temperature_current_value)

		#Respiratory Rate modified feature
		respiratory_rate_current_value = float(row['V4'])
		if pd.isnull(respiratory_rate_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				respiratory_rate_data_history = data_sub.V4
				if timestamp < (3600*24):
					score_list = []
					for measurement in respiratory_rate_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_respiratory_rate(int(measurement)))
					if not score_list:
						respiratory_rate_score_value = 0
					else:
						respiratory_rate_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					respiratory_rate_data_history = data_sub.V4
					score_list = []
					for measurement in respiratory_rate_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_respiratory_rate(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						respiratory_rate_data_history = data_sub.V4
						score_list = []
						for measurement in respiratory_rate_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_respiratory_rate(int(measurement)))
						if not score_list:
							respiratory_rate_score_value = 0
						else:
							respiratory_rate_score_value = max(score_list)
					else:
						respiratory_rate_score_value = max(score_list)
			else:
				respiratory_rate_score_value = 0
		else:
			respiratory_rate_score_value = score_respiratory_rate(respiratory_rate_current_value)

		#Partial Pressure of Oxygen modified feature
		PaO2_current_value = float(row['L3'])
		if pd.isnull(PaO2_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				PaO2_data_history = data_sub.L3
				if timestamp < (3600*24):
					score_list = []
					for measurement in PaO2_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_PaO2(float(measurement)))
					if not score_list:
						PaO2_score_value = 0
					else:
						PaO2_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					PaO2_data_history = data_sub.L3
					score_list = []
					for measurement in PaO2_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_PaO2(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						PaO2_data_history = data_sub.L3
						score_list = []
						for measurement in PaO2_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_PaO2(float(measurement)))
						if not score_list:
							PaO2_score_value = 0
						else:
							PaO2_score_value = max(score_list)
					else:
						PaO2_score_value = max(score_list)
			else:
				PaO2_score_value = 0
		else:
			PaO2_score_value = score_PaO2(PaO2_current_value)

		#AaDO2 modified feature
		fio2_current_value = float(row['L20'])
 		paCO2_current_value = float(row['L2'])
 		paO2_current_value = float(row['L3'])
		if ((pd.isnull(fio2_current_value)) or (pd.isnull(paCO2_current_value)) or (pd.isnull(paO2_current_value))):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				AaDO2_data_history = data_sub[['L20','L2','L3']]
				if timestamp < (3600*24):
					score_list = []
					for index,row_3 in AaDO2_data_history.iterrows():
						fio2_current_value = float(row_3['L20'])
 						paCO2_current_value = float(row_3['L2'])
 						paO2_current_value = float(row_3['L3'])
						if ((pd.isnull(fio2_current_value)) or (pd.isnull(paCO2_current_value)) or (pd.isnull(paO2_current_value))):
							score_list.append(0)
						else:
							AaDO2_score_value = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
							score_list.append(AaDO2_score_value)
						if not score_list:
							AaDO2_score = 0
						else:
							AaDO2_score = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					AaDO2_data_history = data_sub[['L20','L2','L3']]
					score_list = []
					for index,row_4 in AaDO2_data_history.iterrows():
						fio2_current_value = float(row_4['L20'])
 						paCO2_current_value = float(row_4['L2'])
 						paO2_current_value = float(row_4['L3'])
						if ((pd.isnull(fio2_current_value)) or (pd.isnull(paCO2_current_value)) or (pd.isnull(paO2_current_value))):
							score_list.append(0)
						else:
							AaDO2_score_value = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
							score_list.append(AaDO2_score_value)
						if not score_list:
							data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
							AaDO2_data_history = data_sub[['L20','L2','L3']]
							score_list = []
							for index,row_5 in AaDO2_data_history.iterrows():
								fio2_current_value = float(row_5['L20'])
 								paCO2_current_value = float(row_5['L2'])
 								paO2_current_value = float(row_5['L3'])
								if ((pd.isnull(fio2_current_value)) or (pd.isnull(paCO2_current_value)) or (pd.isnull(paO2_current_value))):
									score_list.append(0)
								else:
									AaDO2_score_value = score_AaDO2_partial_pressure(fio2,paCO2,paO2)
									score_list.append(AaDO2_score_value)
							if not score_list:
								AaDO2_score = 0
							else:
								AaDO2_score = max(score_list)
						else:
							AaDO2_score = max(score_list)
			else:
				AaDO2_score = 0
		else:
			AaDO2_score = score_AaDO2_partial_pressure(fio2_current_value,paCO2_current_value,paO2_current_value)
		
		#Hematocrit modified feature
		Hematocrit_current_value = float(row['L10'])
		if pd.isnull(Hematocrit_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Hematocrit_data_history = data_sub.L10
				if timestamp < (3600*24):
					score_list = []
					for measurement in Hematocrit_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Hematocrit(int(measurement)))
					if not score_list:
						Hematocrit_score_value = 0
					else:
						Hematocrit_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Hematocrit_data_history = data_sub.L10
					score_list = []
					for measurement in Hematocrit_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Hematocrit(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Hematocrit_data_history = data_sub.L10
						score_list = []
						for measurement in Hematocrit_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_Hematocrit(int(measurement)))
						if not score_list:
							Hematocrit_score_value = 0
						else:
							Hematocrit_score_value = max(score_list)
					else:
						Hematocrit_score_value = max(score_list)
			else:
				Hematocrit_score_value = 0
		else:
			Hematocrit_score_value = score_Hematocrit(Hematocrit_current_value)

		#WBC count modified feature
		WBC_current_value = float(row['L9'])
		if pd.isnull(WBC_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				WBC_data_history = data_sub.L9
				if timestamp < (3600*24):
					score_list = []
					for measurement in WBC_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_WBC(int(measurement)))
					if not score_list:
						WBC_score_value = 0
					else:
						WBC_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					WBC_data_history = data_sub.L9
					score_list = []
					for measurement in WBC_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_WBC(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						WBC_data_history = data_sub.L9
						score_list = []
						for measurement in WBC_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_WBC(int(measurement)))
						if not score_list:
							WBC_score_value = 0
						else:
							WBC_score_value = max(score_list)
					else:
						WBC_score_value = max(score_list)
			else:
				WBC_score_value = 0
		else:
			WBC_score_value = score_WBC(WBC_current_value)

		#Serum Creatinine modified
		serum_creatinine_current_value = float(row['L8'])
		if pd.isnull(serum_creatinine_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				serum_creatinine_data_history = data_sub.L8
				if timestamp < (3600*24):
					score_list = []
					for measurement in serum_creatinine_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_serum_creatinine(int(measurement)))
					if not score_list:
						serum_creatinine_score_value = 0
					else:
						serum_creatinine_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					serum_creatinine_data_history = data_sub.L8
					score_list = []
					for measurement in serum_creatinine_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_serum_creatinine(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						serum_creatinine_data_history = data_sub.L8
						score_list = []
						for measurement in serum_creatinine_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_serum_creatinine(int(measurement)))
						if not score_list:
							serum_creatinine_score_value = 0
						else:
							serum_creatinine_score_value = max(score_list)
					else:
						serum_creatinine_score_value = max(score_list)
			else:
				serum_creatinine_score_value = 0
		else:
			serum_creatinine_score_value = score_serum_creatinine(serum_creatinine_current_value)

		#Urine Output modified feature
		Urine_output_current_value = float(row['L13'])
		if pd.isnull(Urine_output_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Urine_output_data_history = data_sub.L13
				if timestamp < (3600*24):
					score_list = []
					for measurement in Urine_output_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Urine_output(float(measurement)))
					if not score_list:
						Urine_output_score_value = 0
					else:
						Urine_output_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Urine_output_data_history = data_sub.L13
					score_list = []
					for measurement in Urine_output_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Urine_output(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Urine_output_data_history = data_sub.L13
						score_list = []
						for measurement in Urine_output_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_Urine_output(float(measurement)))
						if not score_list:
							Urine_output_score_value = 0
						else:
							Urine_output_score_value = max(score_list)
					else:
						Urine_output_score_value = max(score_list)
			else:
				Urine_output_score_value = 0
		else:
			Urine_output_score_value = score_Urine_output(Urine_output_current_value)

		#Serun BUN modified feature
		BUN_current_value = float(row['L7'])
		if pd.isnull(BUN_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				BUN_data_history = data_sub.L7
				if timestamp < (3600*24):
					score_list = []
					for measurement in BUN_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_BUN(float(measurement)))
					if not score_list:
						BUN_score_value = 0
					else:
						BUN_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					BUN_data_history = data_sub.L7
					score_list = []
					for measurement in BUN_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_BUN(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						BUN_data_history = data_sub.L7
						score_list = []
						for measurement in BUN_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_BUN(float(measurement)))
						if not score_list:
							BUN_score_value = 0
						else:
							BUN_score_value = max(score_list)
					else:
						BUN_score_value = max(score_list)
			else:
				BUN_score_value = 0
		else:
			BUN_score_value = score_BUN(BUN_current_value)
		#Serum Na modified feature
		Sodium_current_value = float(row['L4'])
		if pd.isnull(Sodium_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Sodium_data_history = data_sub.L4
				if timestamp < (3600*24):
					score_list = []
					for measurement in Sodium_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Sodium(float(measurement)))
					if not score_list:
						Sodium_score_value = 0
					else:
						Sodium_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Sodium_data_history = data_sub.L4
					score_list = []
					for measurement in Sodium_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Sodium(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Sodium_data_history = data_sub.L4
						score_list = []
						for measurement in Sodium_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_Sodium(float(measurement)))
						if not score_list:
							Sodium_score_value = 0
						else:
							Sodium_score_value = max(score_list)
					else:
						Sodium_score_value = max(score_list)
			else:
				Sodium_score_value = 0
		else:
			Sodium_score_value = score_Sodium(Sodium_current_value)

		#Serum Albumin modified feature
		Albumin_current_value = float(row['L21'])
		if pd.isnull(Albumin_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Albumin_data_history = data_sub.L21
				if timestamp < (3600*24):
					score_list = []
					for measurement in Albumin_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Albumin_output(float(measurement)))
					if not score_list:
						Albumin_score_value = 0
					else:
						Albumin_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Albumin_data_history = data_sub.L21
					score_list = []
					for measurement in Albumin_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_Albumin_output(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Albumin_data_history = data_sub.L21
						score_list = []
						for measurement in Albumin_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_Albumin_output(float(measurement)))
						if not score_list:
							Albumin_score_value = 0
						else:
							Albumin_score_value = max(score_list)
					else:
						Albumin_score_value = max(score_list)
			else:
				Albumin_score_value = 0
		else:
			Albumin_score_value = score_Albumin_output(Albumin_current_value)

		#Serum Bilirubin modified feature
		Bilirubin_current_value = float(row['L12'])
		if pd.isnull(Bilirubin_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Bilirubin_data_history = data_sub.L12
				if timestamp < (3600*24):
					score_list = []
					for measurement in Bilirubin_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_bilirubin(float(measurement)))
					if not score_list:
						Bilirubin_score_value = 0
					else:
						Bilirubin_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Bilirubin_data_history = data_sub.L12
					score_list = []
					for measurement in Bilirubin_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_bilirubin(float(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Bilirubin_data_history = data_sub.L12
						score_list = []
						for measurement in Bilirubin_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_bilirubin(float(measurement)))
						if not score_list:
							Bilirubin_score_value = 0
						else:
							Bilirubin_score_value = max(score_list)
					else:
						Bilirubin_score_value = max(score_list)
			else:
				Bilirubin_score_value = 0
		else:
			Bilirubin_score_value = score_bilirubin(Bilirubin_current_value)
		
		#Serum Glucose modified feature
		Serum_glucose_current_value = float(row['L19'])
		if pd.isnull(Serum_glucose_current_value):
			if timestamp>0:
				data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
				Serum_glucose_data_history = data_sub.L19
				if timestamp < (3600*24):
					score_list = []
					for measurement in Serum_glucose_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_serum_glucose(int(measurement)))
					if not score_list:
						Serum_glucose_score_value = 0
					else:
						Serum_glucose_score_value = max(score_list)
				else:
					timestamp_less_24 = timestamp - (3600*24)
					data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp) & (val_df.TIME > timestamp_less_24)]
					Serum_glucose_data_history = data_sub.L19
					score_list = []
					for measurement in Serum_glucose_data_history:
						if not pd.isnull(measurement):
							score_list.append(score_serum_glucose(int(measurement)))
					if not score_list:
						data_sub = val_df[(val_df.ID == patient_id) & (val_df.TIME < timestamp)]
						Serum_glucose_data_history = data_sub.L19
						score_list = []
						for measurement in Serum_glucose_data_history:
							if not pd.isnull(measurement):
								score_list.append(score_serum_glucose(int(measurement)))
						if not score_list:
							Serum_glucose_score_value = 0
						else:
							Serum_glucose_score_value = max(score_list)
					else:
						Serum_glucose_score_value = max(score_list)
			else:
				Serum_glucose_score_value = 0
		else:
			Serum_glucose_score_value = score_serum_glucose(Serum_glucose_current_value)



		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Pulse',pulse_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Mean_BP',mean_bp_score)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'AaDO2',AaDO2_score)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Temperature',temperature_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Respiratory_rate',respiratory_rate_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'PaO2',PaO2_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Hematocrit',Hematocrit_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Urine_output',Urine_output_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Bilirubin',Bilirubin_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'WBC',WBC_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'BUN',BUN_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Sodium',Sodium_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Serum_glucose',Serum_glucose_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Serum_creatinine',serum_creatinine_score_value)
		modified_feature_val_matrix.set_value(modified_val_matrix_index,'Albumin',Albumin_score_value)

print modified_feature_val_matrix


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


