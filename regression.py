import pickle
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
from sklearn.pipeline import Pipeline

data = pd.read_csv('parkinsons_disease_data_reg.csv')

#region Preprocessing

''' no duplicates in the dataset'''
print("Duplicates in the dataset")
print(data.duplicated().sum())

''' No Outliers in the dataset '''
numerical_data = data.select_dtypes(exclude='object')

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers.sum()

outlier_counts = numerical_data.apply(detect_outliers_iqr)

print("Outliers in the dataset")
for feature, count in outlier_counts.sort_values(ascending=False).items():
    print(f"{feature}: {count} outliers")

'''Filling missing values'''
data.fillna({'EducationLevel': 'No Education'}, inplace=True)

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
'''Transform WeeklyPhysicalActivity (hr) into Minutes'''
def hour_to_minutes(time):
    split = str(time).split(':')
    hour = int(split[0])
    minute = int(split[1])
    return hour*60 + minute

data["WeeklyPhysicalActivity (hr)"] = data["WeeklyPhysicalActivity (hr)"].apply(hour_to_minutes)

'''Normalization'''
Numerical_cols = data.select_dtypes(exclude='object').columns
Numerical_cols = Numerical_cols.drop(['PatientID'])

scaler = preprocessing.MinMaxScaler()
data[Numerical_cols] = scaler.fit_transform(data[Numerical_cols])

'''Encoding categorical features'''
categorical_cols = data.select_dtypes(include='object').columns
categorical_cols = categorical_cols.drop(['DoctorInCharge'])
label_encoders = {}
for col in categorical_cols:
    le = preprocessing.LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

#endregion

#region FeatureSelection

#Numerical features
'''Pearson Correlation'''
Corr_Numerical_data = data[Numerical_cols]
corr = Corr_Numerical_data.corr()
top_feature_OF_Numericals = corr.index[abs(corr['UPDRS'])>0.03]
plt.subplots(figsize=(12, 8))
top_corr = Corr_Numerical_data[top_feature_OF_Numericals].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature_OF_Numericals = top_feature_OF_Numericals.drop("UPDRS")
print(top_feature_OF_Numericals)

abs_corr = abs(corr['UPDRS'].round(2))
abs_corr = abs_corr[abs_corr < 1]
freq = collections.Counter(abs_corr)
sorted_dict = dict(sorted(freq.items()))
plt.plot(list(sorted_dict.keys()), list(sorted_dict.values()), marker='o')
plt.xlabel('Correlation Coefficient (Rounded to 2)')
plt.ylabel('Frequency')
plt.title('Frequency of Correlation Coefficients')
plt.grid()
plt.show()

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
        if len(data[col].unique()) < 2:
            continue
        group1 = data[data[col] == 0]['UPDRS']
        group2 = data[data[col] == 1]['UPDRS']
        t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
        if p_value < 0.05:
            significant_features.append(col)

'''Random Forest'''
X = data.drop(columns=['UPDRS', 'PatientID', 'DoctorInCharge'])
Y = data['UPDRS']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

estimator = [10,20,30,40,50,60,70,80,90,100]
list_estimator = []
for i in estimator:
    rf = RandomForestRegressor(n_estimators=i, random_state=42)
    rf.fit(X_train, Y_train)
    prediction = rf.predict(X_test)
    train_err = metrics.mean_squared_error(Y_train, rf.predict(X_train))
    test_err = metrics.mean_squared_error(Y_test, prediction)
    print('Train error (MSE) for n_estimators {}: '.format(i), round(train_err, 4))
    print('Test error (MSE) for n_estimators {}: '.format(i), round(test_err, 4))
    list_estimator.append(abs(test_err-train_err))

best_estimator = estimator[list_estimator.index(min(list_estimator))]
print("Best Estimator: ", best_estimator)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
rf = RandomForestRegressor(n_estimators=best_estimator, random_state=42)
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

# region Training and Testing Models
top_feature = top_7_features.union(top_feature_OF_Numericals).union(selected_features).union(significant_features)
X = X[top_feature]
Y = data['UPDRS']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
diffrence_error = []
model_performance = []

def plot_prediction(y_actual, y_predicted, title, color, label):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_predicted, color=color, label=label)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             color='green', linestyle='--', label='Prediction Line')
    plt.xlabel('Actual UPDRS')
    plt.ylabel('Predicted UPDRS')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test, model, degree=3, alpha=None):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    model.fit(X_train_poly, y_train)
    y_train_predicted = model.predict(X_train_poly)
    y_test_predicted = model.predict(poly_features.fit_transform(X_test))
    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, y_test_predicted)
    print(f'Train error (MSE) : {round(train_err, 4)}')
    print(f'Test error (MSE) : {round(test_err, 4)}')
    model_performance.append({
        'model_name': model.__class__.__name__,
        'degree': degree,
        'alpha': alpha,
        'train_mse': train_err,
        'test_mse': test_err,
        'error_diff': test_err - train_err,
        'R2_train': metrics.r2_score(y_train, y_train_predicted),
        'R2_test': metrics.r2_score(y_test, y_test_predicted)
    })
    return y_train_predicted, y_test_predicted

def model_trial(X_train, X_test, y_train, y_test, model, degree=3, alpha=None):
    y_train_predicted, y_test_predicted = train_and_evaluate(X_train, X_test, y_train, y_test, model, degree, alpha)
    print("Train R^2 score: ", round(metrics.r2_score(y_train, y_train_predicted), 4))
    print("Test R^2 score: ", round(metrics.r2_score(y_test, y_test_predicted), 4))
    plot_prediction(y_train, y_train_predicted, f'Train Data: Prediction vs Actual in Model {model.__class__.__name__} (Degree {degree} & Alpha {alpha})', 'blue', 'Train Data')
    plot_prediction(y_test, y_test_predicted, f'Test Data: Prediction vs Actual in Model {model.__class__.__name__} (Degree {degree} & Alpha {alpha})', 'red', 'Test Data')

model_performance = []
degree = [1, 2, 3, 4, 5]
for i in degree:
    print('---------------------------------------------------------')
    print(f"Degree: {i}")
    print("Polynomial Regression")
    train_and_evaluate(X_train, X_test, Y_train, Y_test, linear_model.LinearRegression(), i)
    print("Ridge Regression")
    train_and_evaluate(X_train, X_test, Y_train, Y_test, linear_model.Ridge(), i)
    print("Lasso Regression")
    train_and_evaluate(X_train, X_test, Y_train, Y_test, linear_model.Lasso(), i)
    print('---------------------------------------------------------')

best_degree_model = min(model_performance, key=lambda x:x['test_mse'])
best_degree = best_degree_model['degree']
print(f"Best Degree: {best_degree}")

model_performance = []
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in alpha:
    print('*******************************************************************')
    print(f"Alpha: {i}")
    print("Ridge Regression")
    train_and_evaluate(X_train, X_test, Y_train, Y_test, linear_model.Ridge(alpha=i), best_degree, i)
    print("Lasso Regression")
    train_and_evaluate(X_train, X_test, Y_train, Y_test, linear_model.Lasso(alpha=i), best_degree, i)
    print('*******************************************************************')

best_alpha_model = min(model_performance, key=lambda x: x['test_mse'])
best_alpha = best_alpha_model['alpha']
print(f"Best Alpha (based on smallest train-test MSE difference): {best_alpha}")

model_performance = []
print("Polynomial Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.LinearRegression(), best_degree)
print("Ridge Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.Ridge(alpha=best_alpha), best_degree, best_alpha)
print("Lasso Regression")
model_trial(X_train, X_test, Y_train, Y_test, linear_model.Lasso(alpha=best_alpha), best_degree, best_alpha)

best_model = min(model_performance, key=lambda x: x['test_mse'])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Best Model with least test MSE:")
print(f"Model: {best_model['model_name']}")
print(f"Degree: {best_model['degree']}")
print(f"Alpha: {best_model['alpha']}")
print(f"Train MSE: {round(best_model['train_mse'], 4)}")
print(f"Test MSE: {round(best_model['test_mse'], 4)}")
print(f"Train R^2: {round(best_model['R2_train'], 4)}")
print(f"Test R^2: {round(best_model['R2_test'], 4)}")

# endregion

# region Saving Models and Preprocessing Artifacts
def save_artifact(obj, filename):
    """Save an object to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {filename}")


model_configs = [
    ('polynomial_regression', Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree)),
        ('model', linear_model.LinearRegression())
    ])),
    ('ridge_regression', Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree)),
        ('model', linear_model.Ridge(alpha=best_alpha))
    ])),
    ('lasso_regression', Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree)),
        ('model', linear_model.Lasso(alpha=best_alpha))
    ]))
]


for name, pipeline in model_configs:
    pipeline.fit(X_train, Y_train)
    save_artifact(pipeline, f'{name}_model.pkl')
    
preprocessing_artifacts = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'selected_features' : list(top_feature),
    'numerical_cols': list(Numerical_cols),
    'categorical_cols': list(categorical_cols)
}
save_artifact(preprocessing_artifacts, 'preprocessing_artifacts.pkl')

print("All artifacts saved successfully:")
for name, _ in model_configs:
    print(f"- {name}_model.pkl")
print("- preprocessing_artifacts.pkl (includes scaler, label encoders, and feature lists)")
# endregion