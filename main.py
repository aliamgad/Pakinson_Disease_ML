import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
import ast
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

data = pd.read_csv('parkinsons_disease_data_reg.csv')


#region Preprocessing

''' no duplicates in the dataset'''
# data= data.drop_duplicates()
# print(data.duplicated().sum())

''' No Outliers in the dataset '''
# numerical_data = data.select_dtypes(exclude='object')

# def detect_outliers_iqr(data):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
#     return outliers.sum()

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
'''Tansform WeeklyPhysicalActivity (hr) into Minutes'''
def hour_to_minutes(time):

    split = str(time).split(':')
    hour = int(split[0])
    minute= int(split[1])
    
    return hour*60 + minute

data["WeeklyPhysicalActivity (hr)"] = data["WeeklyPhysicalActivity (hr)"].apply(hour_to_minutes)



'''Normalization'''
Numerical_cols = data.select_dtypes(exclude='object').columns
Numerical_cols = Numerical_cols.drop(['PatientID'])


scaler = preprocessing.MinMaxScaler()
data[Numerical_cols] = scaler.fit_transform(data[Numerical_cols])

'''Encoding categorical features'''
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    le = preprocessing.LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

#endregion

#region FeatureSelection

#Numerical features
'''Pearson Correlation'''
Corr_Numerical_data = data[Numerical_cols]
corr = Corr_Numerical_data.corr()
top_feature_OF_Numericals = corr.index[abs(corr['UPDRS'])>0.03] # 0.02 can be used for correlation threshold
#Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = Corr_Numerical_data[top_feature_OF_Numericals].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
top_feature_OF_Numericals = top_feature_OF_Numericals.drop("UPDRS")


# # Plot the frequency of the correlation coefficients
# abs_corr = abs(corr['UPDRS'].round(2))
# abs_corr = abs_corr[abs_corr < 1]
# freq = collections.Counter(abs_corr)
# sorted_dict = dict(sorted(freq.items()))
# plt.plot(list(sorted_dict.keys()), list(sorted_dict.values()), marker='o')
# plt.xlabel('Correlation Coefficient (Rounded to 2)')
# plt.ylabel('Frequency')
# plt.title('Frequency of Correlation Coefficients')
# plt.grid()
# plt.show()


#Categorical features

significant_features = []

for col in categorical_cols:
    if len(data[col].unique()) >= 3:
        '''Anova Correlation'''
        groups = [data[data[col] == val]['UPDRS'] for val in data[col].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        if p_value < 0.05:
            significant_features.append(col)
    else:
        '''t-test Correlation'''
        group1 = data[data[col] == 0]['UPDRS']
        group2 = data[data[col] == 1]['UPDRS']
        t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
        if p_value < 0.05:
            significant_features.append(col)

'''Random Forest'''

X = data.drop(columns=['UPDRS', 'PatientID'])
Y = data['UPDRS']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

top_7_features = X.columns[indices[:7]]

'''Backward Elimination'''
model = LinearRegression()
num_features_to_select = 7 
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)
X_rfe = rfe.fit_transform(X, Y)

selected_features = X.columns[rfe.support_]

#endregion

#Trainig and Testing Models

top_feature=top_7_features.union(top_feature_OF_Numericals).union(selected_features).union(significant_features)

X = X[top_feature]
Y = data['UPDRS']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def model_trial(X_train, X_test, y_train, y_test, model, degree=2):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    model.fit(X_train_poly, y_train)

    y_train_predicted = model.predict(X_train_poly)
    prediction = model.predict(poly_features.fit_transform(X_test))

    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    print('Train subset (MSE) for degree {}: '.format(degree), round(train_err, 4))
    print('Test subset (MSE) for degree {}: '.format(degree), round(test_err, 4))
    
    

print("Polynomial Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.LinearRegression())
print("Ridge Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.Ridge())
print("Lasso Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.Lasso())

#endregion
