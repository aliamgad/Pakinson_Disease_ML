import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import ast
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

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
'''Tansform WeeklyPhysicalActivity (hr) into Minutes'''
def hour_to_minutes(time):

    split = str(time).split(':')
    hour = int(split[0])
    minute= int(split[1])
    
    return hour*60 + minute

data["WeeklyPhysicalActivity (hr)"] = data["WeeklyPhysicalActivity (hr)"].apply(hour_to_minutes)



'''Normalization'''
Numerical_cols = data.select_dtypes(exclude='object').columns #select  only the colums with type =='object'
Numerical_cols = Numerical_cols.drop(['PatientID'])

# data = preprocessing.normalize(data[Numerical_cols], axis=0)
scaler = preprocessing.MinMaxScaler()
data[Numerical_cols] = scaler.fit_transform(data[Numerical_cols])
# print(data[Numerical_cols].head())

'''Encoding categorical features'''
categorical_cols = data.select_dtypes(include='object').columns #select  only the colums with type =='object'
# label_encoders = {}

for col in categorical_cols:
    le = preprocessing.LabelEncoder()
    # print(data[col].astype(str).unique())
    data[col] = le.fit_transform(data[col].astype(str))
    # label_encoders[col] = le #save for later (transform encoded values back to object)
    # print("After")
    # print(data[col].astype(str).unique())
#endregion

#region FeatureSelection

# # Pearson Correlation
# Corr_Numerical_data = data[Numerical_cols]
# corr = Corr_Numerical_data.corr()
# print("Correlation matrix:")
# print(abs(corr['UPDRS']).sort_values(ascending=False))
# # print(corr['UPDRS'])

# # plt.subplots(figsize=(12, 8))
# # sns.heatmap(corr, annot=True)
# # plt.show()

# Anova Correlation
three_value_cols = ['Ethnicity','EducationLevel']

for col in three_value_cols:
    f_stat, p_value = stats.f_oneway(data[data[col] == 0]['UPDRS'], data[data[col] == 1]['UPDRS'])
    # print(f"ANOVA for {col}: F-statistic = {f_stat}, p-value = {p_value}")
    if p_value < 0.05:
        print(f"Feature {col} is significant (p < 0.05)")
    # else:
    #     print(f"Feature {col} is not significant (p >= 0.05)")


# t-test correlation
binary_categorical_cols = [col for col in categorical_cols if data[col].nunique() == 2]

for col in binary_categorical_cols:
    group1 = data[data[col] == 0]['UPDRS']
    group2 = data[data[col] == 1]['UPDRS']
    t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    # print(f"t-test for {col}: t-statistic = {t_stat}, p-value = {p_value}")
    if p_value < 0.05:
        print(f"Feature {col} is significant (p < 0.05)")
    # else:
    #     print(f"Feature {col} is not significant (p >= 0.05)")

##Use Random Forest to select important features

X = data.drop(columns=['UPDRS'])
y = data['UPDRS']

#Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=31
)

#Set up Random Forest and hyperparameter grid
random_forest_model = RandomForestRegressor(random_state=17)

param_dictionary = {
    'n_estimators': [100, 200, 500, 1000, 1500],
    'max_depth': [None, 5, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': [None, 'log2', 'sqrt', 0.5, 0.75]
}

#Randomized search over the grid with 5‑fold CV
rand_search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=param_dictionary,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=31,
    n_jobs=-1
)

rand_search.fit(X_train, y_train)

#print("Best hyperparameters:", rand_search.best_params_)
best_random_forest_model = rand_search.best_estimator_

#Extract feature importances from the best model
importances = best_random_forest_model.feature_importances_
indices     = np.argsort(importances)[::-1]           # sort indices by importance

#Compute cumulative sum and select until threshold
cumulative_importance   = np.cumsum(importances[indices])
threshold    = 0.85
#Keep all features up to the first index where cumulative >= threshold
cutoff_idx   = np.where(cumulative_importance >= threshold)[0][0]  
selected_idx = indices[: cutoff_idx + 1]             

#Map back to feature names
selected_features = X.columns[selected_idx].tolist()
#print(f"Features selected (cumulative ≥ {threshold*100:.0f}%):")
#for feat in selected_features:
#    print(f"  • {feat}")

X_train_selected = X_train.iloc[:, selected_idx]
X_test_selected  = X_test.iloc[:, selected_idx]
#endregion


#Retrain final model on selected features 
final_random_forest_model = RandomForestRegressor(**rand_search.best_params_, random_state=31, n_jobs=-1)
final_random_forest_model.fit(X_train_selected, y_train)

# Evaluate the final model on X_test_sel, y_test as usual
mse = metrics.mean_squared_error(y_test, final_random_forest_model.predict(X_test_selected))
print(f"Final Test MSE: {mse:.4f}")
