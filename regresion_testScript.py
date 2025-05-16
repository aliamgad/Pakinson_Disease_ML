import pickle
import pandas as pd
import numpy as np
import ast
from sklearn import metrics


test_data = pd.read_csv('parkinsons_disease_data_reg_test.csv')  # Replace with your test file path


with open('preprocessing_artifacts.pkl', 'rb') as f:
    preprocessing_artifacts = pickle.load(f)

scaler = preprocessing_artifacts['scaler']
label_encoders = preprocessing_artifacts['label_encoders']
selected_features = preprocessing_artifacts['selected_features']
categorical_cols = preprocessing_artifacts['categorical_cols']
numerical_cols = preprocessing_artifacts['numerical_cols']


def hour_to_minutes(time):
    if pd.isna(time):
        return np.nan
    split = str(time).split(':')
    hour = int(split[0])
    minute = int(split[1])
    return hour * 60 + minute


test_data['WeeklyPhysicalActivity (hr)'] = test_data['WeeklyPhysicalActivity (hr)'].apply(hour_to_minutes)

# Fill null values

for col in numerical_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna(test_data[col].median())

# Categorical columns: fill with mode
for col in categorical_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])


test_data['EducationLevel'] = test_data['EducationLevel'].fillna('No Education')
test_data['DoctorInCharge'] = test_data['DoctorInCharge'].fillna('Unknown')
test_data['WeeklyPhysicalActivity (hr)'] = test_data['WeeklyPhysicalActivity (hr)'].fillna(test_data['WeeklyPhysicalActivity (hr)'].mode()[0])

# MedicalHistory and Symptoms: fill with default dictionary
default_dict = {'FamilyHistoryParkinsons': 'No', 'TraumaticBrainInjury': 'No', 'Hypertension': 'No', 
                'Diabetes': 'No', 'Depression': 'No', 'Stroke': 'No'}
test_data['MedicalHistory'] = test_data['MedicalHistory'].fillna(str(default_dict))
default_symptoms = {'Tremor': 'No', 'Rigidity': 'No', 'Bradykinesia': 'No', 'PosturalInstability': 'No', 
                   'SpeechProblems': 'No', 'SleepDisorders': 'No', 'Constipation': 'No'}
test_data['Symptoms'] = test_data['Symptoms'].fillna(str(default_symptoms))


test_data['MedicalHistory'] = test_data['MedicalHistory'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
test_data['Symptoms'] = test_data['Symptoms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
medical_history_data = test_data['MedicalHistory'].apply(pd.Series)
symptoms_data = test_data['Symptoms'].apply(pd.Series)
test_data = pd.concat([test_data.drop(columns=['MedicalHistory', 'Symptoms']),
                       medical_history_data.add_prefix('MedHist_'),
                       symptoms_data.add_prefix('Symptom_')], axis=1)



test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])


for col in categorical_cols:
    if col in test_data.columns and col in label_encoders:
        # Handle unseen categories by assgning to the first class (mode from training)
        # test_data[col] = test_data[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0]) #if the category was not in the training set
        test_data[col] = label_encoders[col].transform(test_data[col].astype(str))


X_test = test_data[selected_features]
Y_test = test_data['UPDRS'] if 'UPDRS' in test_data.columns else None


model_files = [
    'polynomial_regression_model.pkl',
    'ridge_regression_model.pkl',
    'lasso_regression_model.pkl'
]

for model_file in model_files:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    

    y_pred = model.predict(X_test)
    

    model_name = model_file.replace('_model.pkl', '')
    print(f"\nPredictions for {model_name}:")
    print(y_pred[:10]) 
    

    if Y_test is not None:
        mse = metrics.mean_squared_error(Y_test, y_pred)
        r2 = metrics.r2_score(Y_test, y_pred)
        print(f"Metrics for {model_name}:")
        print(f"Test MSE: {round(mse, 4)}")
        print(f"Test R^2: {round(r2, 4)}")


predictions_df = pd.DataFrame({
    'PatientID': test_data['PatientID'],
    'Polynomial_Predictions': pickle.load(open('polynomial_regression_model.pkl', 'rb')).predict(X_test),
    'Ridge_Predictions': pickle.load(open('ridge_regression_model.pkl', 'rb')).predict(X_test),
    'Lasso_Predictions': pickle.load(open('lasso_regression_model.pkl', 'rb')).predict(X_test)
})
if Y_test is not None:
    predictions_df['Actual_UPDRS'] = Y_test

predictions_df.to_csv('model_predictions.csv', index=False)
print("\nPredictions saved to 'model_predictions.csv'")