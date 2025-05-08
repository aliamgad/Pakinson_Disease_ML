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

with open('ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Load selected features
# if os.path.exists('selected_features.txt'):
#     with open('selected_features.txt', 'r') as f:
#         selected_features = f.read().splitlines()
#     print("Loaded selected features from text file:", selected_features)
# else:
    # Fallback to pickle file for backward compatibility
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
print("Loaded selected features from pickle file:", selected_features)


def predict_new_data(test_file):
    test_data = pd.read_csv(test_file)

    def parse_nested(data, column):

        data[column] = data[column].apply(eval)  # Convert string to list
        return pd.get_dummies(data[column].explode()).groupby(level=0).sum()

    test_data = pd.concat([test_data.drop(['MedicalHistory', 'Symptoms'], axis=1),
                   parse_nested(test_data, 'MedicalHistory').add_prefix('MedHist_'),
                   parse_nested(test_data, 'Symptoms').add_prefix('Symptom_')], axis=1)

    x = test_data.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])
    y = test_data['Diagnosis']

    # Handle nested data (MedicalHistory, Symptoms)

    iter = ['MedHist_Depression', 'MedHist_Diabetes',
            'MedHist_FamilyHistoryParkinsons', 'MedHist_Hypertension',
            'MedHist_Stroke', 'MedHist_TraumaticBrainInjury',
            'Symptom_Bradykinesia', 'Symptom_Constipation',
            'Symptom_PosturalInstability', 'Symptom_Rigidity',
            'Symptom_SleepDisorders', 'Symptom_SpeechProblems', 'Symptom_Tremor']

    for col in iter:
        x[col] = x[col].astype('object')  # Ensure float type for chi2

    def hour_to_minutes(time):
        hours, minutes = map(int, time.split(':'))
        return hours + minutes / 60.0

    x["WeeklyPhysicalActivity (hr)"] = x["WeeklyPhysicalActivity (hr)"].apply(hour_to_minutes)

    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = x.select_dtypes(include=['object']).columns

    # 1- Imputing
    x[categorical_cols] = x[categorical_cols].fillna('missing')
    x[numerical_cols] = imputer.transform(x[numerical_cols])

    # 2- Encoding
    x[categorical_cols] = ordinal_encoder.transform(x[categorical_cols])

    # 3- Scaling
    x[numerical_cols] = scaler.transform(x[numerical_cols])

    # 4- Feature Selection
    x = x[selected_features]

    # Make predictions and calculate detailed test metrics
    predictions = {}
    test_times = {}
    accuracies = {}
    conf_matrices = {}
    class_reports = {}
    for name, model in models.items():
        start_time = time.time()
        pred = model.predict(x)
        test_times[name] = time.time() - start_time
        predictions[name] = pred
        accuracies[name] = accuracy_score(y, pred)
        conf_matrices[name] = confusion_matrix(y, pred)
        class_reports[name] = classification_report(y, pred, output_dict=True)

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