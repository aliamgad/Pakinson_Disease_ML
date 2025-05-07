import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import time
import matplotlib.pyplot as plt
import os

# Load pre-trained models
models = {}
for name in ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'Voting_Classifier', 'Stacking_Classifier']:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

# Load pre-trained preprocessing objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    imputer_data = pickle.load(f)

imputers = imputer_data['imputers']
numerical_cols = imputer_data['numerical_cols']

# Load selected features
if os.path.exists('selected_features.txt'):
    with open('selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()
    print("Loaded selected features from text file:", selected_features)
else:
    # Fallback to pickle file for backward compatibility
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    print("Loaded selected features from pickle file:", selected_features)


def convert_to_minutes(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    try:
        hours, minutes = map(int, time_str.split(':'))
        return (hours * 60) + minutes
    except (ValueError, AttributeError):
        return np.nan

def feature_engineering(df):
    df = df.copy()

    def parse_column(col, keywords):
        if isinstance(col, str):
            return pd.Series([1 if kw in col.lower() else 0 for kw in keywords], index=keywords)
        elif isinstance(col, list):
            return pd.Series([1 if kw in [x.lower() for x in col] else 0 for kw in keywords], index=keywords)
        else:
            return pd.Series([0] * len(keywords), index=keywords)

    # Motor symptoms
    motor_symptoms = ['tremor', 'rigidity', 'bradykinesia', 'posturalinstability']
    if 'Symptoms' in df.columns:
        symptom_features = df['Symptoms'].apply(lambda x: parse_column(x, motor_symptoms))
        df[motor_symptoms] = symptom_features
    else:
        for sym in motor_symptoms:
            df[sym] = 0
    df['MotorSymptomCount'] = df[motor_symptoms].fillna(0).sum(axis=1)

    # Interactions
    df['MoCA_MotorSymptomInteraction'] = df['MoCA'].fillna(0) * df['MotorSymptomCount']
    df['Age_UPDRS_Interaction'] = df['Age'].fillna(0) * df['UPDRS'].fillna(0)
    df['Age_MoCA_Interaction'] = df['Age'].fillna(0) * df['MoCA'].fillna(0)
    df['Age_FunctionalAssessment_Interaction'] = df['Age'].fillna(0) * df['FunctionalAssessment'].fillna(0)

    # Medical history
    medical_history_keywords = ['familyhistoryparkinsons', 'traumaticbraininjury', 'hypertension',
                               'diabetes', 'depression', 'stroke']
    if 'MedicalHistory' in df.columns:
        medical_features = df['MedicalHistory'].apply(lambda x: parse_column(x, medical_history_keywords))
        df[medical_history_keywords] = medical_features
    else:
        for kw in medical_history_keywords:
            df[kw] = 0
    df['FamilyHistory_MotorSymptom_Interaction'] = df['familyhistoryparkinsons'].fillna(0) * df['MotorSymptomCount']
    df['TBI_UPDRS_Interaction'] = df['traumaticbraininjury'].fillna(0) * df['UPDRS'].fillna(0)

    # Symptom combinations
    if 'Symptoms' in df.columns:
        df['Tremor_and_Rigidity'] = df['tremor'].fillna(0) * df['rigidity'].fillna(0)
        df['Bradykinesia_and_PosturalInstability'] = df['bradykinesia'].fillna(0) * df['posturalinstability'].fillna(0)
        additional_symptoms = ['sleepdisorders', 'constipation']
        symptom_features = df['Symptoms'].apply(lambda x: parse_column(x, additional_symptoms))
        df[additional_symptoms] = symptom_features
    else:
        df['Tremor_and_Rigidity'] = 0
        df['Bradykinesia_and_PosturalInstability'] = 0
        df['sleepdisorders'] = 0
        df['constipation'] = 0
    df['SleepDisorders_and_Constipation'] = df['sleepdisorders'].fillna(0) * df['constipation'].fillna(0)

    # Severity indicators
    df['UPDRS_per_Age'] = df['UPDRS'].fillna(0) / (df['Age'].replace(0, np.nan).fillna(1e-6))
    df['MoCA_per_Age'] = df['MoCA'].fillna(0) / (df['Age'].replace(0, np.nan).fillna(1e-6))

    # Cholesterol ratios
    df['LDL_to_HDL'] = df['CholesterolLDL'].fillna(0) / (df['CholesterolHDL'].replace(0, np.nan).fillna(1e-6))
    df['Total_to_HDL'] = df['CholesterolTotal'].fillna(0) / (df['CholesterolHDL'].replace(0, np.nan).fillna(1e-6))

    # Pulse pressure
    df['PulsePressure'] = df['SystolicBP'].fillna(0) - df['DiastolicBP'].fillna(0)

    # Comorbidity score
    hist_cols = [col for col in medical_history_keywords if col in df.columns]
    df['ComorbidityScore'] = df[hist_cols].fillna(0).sum(axis=1)

    # Cognitive-motor interaction
    df['CogMotorInteraction'] = df['MoCA'].fillna(0) * df['FunctionalAssessment'].fillna(0)

    # Normalize health behavior features
    health_cols = ['DietQuality', 'SleepQuality', 'WeeklyPhysicalActivity (min)']
    for c in health_cols:
        if c in df.columns:
            max_val = df[c].max()
            df[c] = df[c] / (max_val if max_val != 0 else 1e-6)
        else:
            df[c] = 0
    df['HealthBehaviorScore'] = df[health_cols].sum(axis=1)

    return df

def predict_new_data(test_file):
    test_data = pd.read_csv(test_file)
    X_new = test_data.drop(columns=['Diagnosis', 'DoctorInCharge'], errors='ignore')
    y_new = test_data['Diagnosis']

    # Convert WeeklyPhysicalActivity (hr) to minutes
    if 'WeeklyPhysicalActivity (hr)' in X_new.columns:
        X_new['WeeklyPhysicalActivity (min)'] = X_new['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)
        X_new = X_new.drop(columns=['WeeklyPhysicalActivity (hr)'], errors='ignore')

    # Ensure numerical columns are numeric
    numerical_cols_expected = ['Age', 'BMI', 'AlcoholConsumption', 'DietQuality', 'SleepQuality',
                              'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                              'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
                              'FunctionalAssessment', 'WeeklyPhysicalActivity (min)']
    for col in numerical_cols_expected:
        if col in X_new.columns:
            X_new[col] = pd.to_numeric(X_new[col], errors='coerce')

    # Impute numerical columns
    X_new_imputed = X_new.copy()
    for col in numerical_cols:  # Global numerical_cols from imputer_data
        if col in X_new.columns and col in imputers:
            X_new_imputed[[col]] = imputers[col].transform(X_new[[col]])
        elif col in numerical_cols and col not in X_new.columns:
            X_new_imputed[col] = imputers[col].statistics_[0]  # Use imputed value from training

    # Apply feature engineering
    X_new = feature_engineering(X_new_imputed)

    # Handle nested data (MedicalHistory, Symptoms)
    def parse_nested(data, column):
        if column in data.columns:
            data[column] = data[column].apply(lambda x: eval(x) if pd.notnull(x) else [])
            return pd.get_dummies(data[column].explode()).groupby(level=0).sum()
        return pd.DataFrame()

    # Ensure dummy variables match training data
    med_hist_dummies = parse_nested(X_new, 'MedicalHistory').add_prefix('MedHist_')
    symptom_dummies = parse_nested(X_new, 'Symptoms').add_prefix('Symptom_')

    # Add missing dummy columns from training
    expected_dummy_cols = [col for col in selected_features if col.startswith(('MedHist_', 'Symptom_'))]
    for col in expected_dummy_cols:
        if col not in med_hist_dummies.columns and col.startswith('MedHist_'):
            med_hist_dummies[col] = 0
        if col not in symptom_dummies.columns and col.startswith('Symptom_'):
            symptom_dummies[col] = 0

    X_new = pd.concat([
        X_new.drop(['MedicalHistory', 'Symptoms'], axis=1, errors='ignore'),
        med_hist_dummies,
        symptom_dummies
    ], axis=1)

    # Ensure all expected categorical columns are present
    expected_categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
    for col in expected_categorical_cols:
        if col not in X_new.columns:
            X_new[col] = 'missing'

    # Encode categorical columns
    categorical_cols = X_new.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        with open('ordinal_encoder.pkl', 'rb') as f:
            ordinal_encoder = pickle.load(f)
        X_new[categorical_cols] = X_new[categorical_cols].fillna('missing')
        X_new[categorical_cols] = ordinal_encoder.transform(X_new[categorical_cols])

    # Scale numerical columns
    numerical_cols_to_scale = X_new.select_dtypes(include=['int64', 'float64']).columns
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_
        for feature in expected_features:
            if feature not in X_new.columns:
                X_new[feature] = 0  # Add missing features
        numerical_cols_to_scale = [f for f in expected_features if f in X_new.columns]
    if len(numerical_cols_to_scale) > 0:
        X_new[numerical_cols_to_scale] = scaler.transform(X_new[numerical_cols_to_scale].astype('float64'))

    # Align with selected features
    for feature in selected_features:
        if feature not in X_new.columns:
            X_new[feature] = 0  # Add missing selected features
    X_new = X_new[selected_features]  # Keep only selected features in the correct order

    # Make predictions
    predictions = {}
    test_times = {}
    accuracies = {}
    conf_matrices = {}
    class_reports = {}
    for name, model in models.items():
        start_time = time.time()
        pred = model.predict(X_new)
        test_times[name] = time.time() - start_time
        predictions[name] = pred
        accuracies[name] = accuracy_score(y_new, pred)
        conf_matrices[name] = confusion_matrix(y_new, pred)
        class_reports[name] = classification_report(y_new, pred, output_dict=True)

    return predictions, test_times, accuracies, conf_matrices, class_reports


def evaluate_test_data(test_file):
    predictions, test_times, accuracies, conf_matrices, class_reports = predict_new_data(test_file)

    print("\nDetailed Test Results Report")
    print("-" * 50)
    for name in models.keys():
        print(f"\nModel: {name}")
        print(f"Test Accuracy: {accuracies[name]:.4f}")
        print(f"Prediction Time: {test_times[name]:.4f} seconds")
        print("\nConfusion Matrix:")
        print(conf_matrices[name])
        print("\nClassification Report:")
        for label, metrics in class_reports[name].items():
            if isinstance(metrics, dict):
                print(f"Class {label}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        print("-" * 50)

    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    best_time = test_times[best_model_name]
    best_class_report = class_reports[best_model_name]

    print("\nBest Model Summary")
    print("=" * 50)
    print(f"Best Model: {best_model_name}")
    print(f"Reason: This model achieved the highest test accuracy of {best_accuracy:.4f}.")
    print(f"Prediction Time: {best_time:.4f} seconds")
    print("\nPerformance Breakdown:")
    print(f"Test Accuracy: {best_accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrices[best_model_name])
    print("\nClassification Report:")
    for label, metrics in best_class_report.items():
        if isinstance(metrics, dict):
            print(f"Class {label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    print("=" * 50)

    # Generate test time bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(test_times.keys(), test_times.values(), color='salmon')
    plt.title('Total Test Time of Models')
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_time.png')

    return accuracies, test_times


# Example usage with detailed report
if __name__ == "__main__":
    predictions, test_times, accuracies, conf_matrices, class_reports = predict_new_data('test.csv')

    print("\nDetailed Test Results Report")
    print("-" * 50)
    for name in models.keys():
        print(f"\nModel: {name}")
        print(f"Test Accuracy: {accuracies[name]:.4f}")
        print(f"Prediction Time: {test_times[name]:.4f} seconds")
        print("\nConfusion Matrix:")
        print(conf_matrices[name])
        print("\nClassification Report:")
        for label, metrics in class_reports[name].items():
            if isinstance(metrics, dict):
                print(f"Class {label}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        print("-" * 50)