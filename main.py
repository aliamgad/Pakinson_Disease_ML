import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import ast
data = pd.read_csv('parkinsons_disease_data_reg.csv')


#region Preprocessing

''' no duplicates in the dataset'''
# data= data.drop_duplicates()
# print(data.duplicated().sum())

''' No Outliers in the dataset '''
# # Select only numerical columns
# numerical_data = data.select_dtypes(include=[np.number])

# # Function to detect outliers using IQR
# def detect_outliers_iqr(data):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
#     return outliers.sum()

# # Apply the function to each numerical column
# outlier_counts = numerical_data.apply(detect_outliers_iqr)

# outlier_counts.sort_values(ascending=False)

# for feature, count in outlier_counts.sort_values(ascending=False).items():
#     print(f"{feature}: {count} outliers")

'''Filling missing values'''
# data['EducationLevel'].fillna('No Education', inplace=True)
data.fillna({'EducationLevel': 'No Education'}, inplace=True)
# print(data['EducationLevel'])
# newdata = data.isna().sum()
# print(newdata)


'''Splitting the last two column into new features'''

data['MedicalHistory'] = data['MedicalHistory'].apply(ast.literal_eval)
data['Symptoms'] = data['Symptoms'].apply(ast.literal_eval)

medical_history_data = data['MedicalHistory'].apply(pd.Series)
symptoms_data = data['Symptoms'].apply(pd.Series)

data = pd.concat([data.drop(columns=['MedicalHistory', 'Symptoms']),
                  medical_history_data.add_prefix('MedHist_'), 
                         symptoms_data.add_prefix('Symptom_')], axis=1)

#endregion

#region Normalization and Feature Encoding

'''Normalization'''
Numerical_cols = data.select_dtypes(exclude='object').columns #select  only the colums with type =='object'
Numerical_cols = Numerical_cols.drop(['PatientID'])

# data = preprocessing.normalize(data[Numerical_cols], axis=0)
scaler = preprocessing.MinMaxScaler()
data[Numerical_cols] = scaler.fit_transform(data[Numerical_cols])
# print(data[Numerical_cols].head())

'''Encoding categorical features'''
categorical_cols = data.select_dtypes(include='object').columns #select  only the colums with type =='object'
categorical_cols = categorical_cols.drop(["WeeklyPhysicalActivity (hr)"])
# label_encoders = {}
for col in categorical_cols:
    le = preprocessing.LabelEncoder()
    # print(data[col].astype(str).unique())
    data[col] = le.fit_transform(data[col].astype(str))
    # label_encoders[col] = le #save for later (transform encoded values back to object)
    # print("After")
    # print(data[col].astype(str).unique())

#endregion

target_feature = data['UPDRS']
