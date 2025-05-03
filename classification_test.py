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
    imputer = pickle.load(f)

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


def predict_new_data(test_file):
    test_data = pd.read_csv(test_file)
    X_new = test_data.drop(columns=['Diagnosis', 'DoctorInCharge'])
    y_new = test_data['Diagnosis']

    # Handle nested data (MedicalHistory, Symptoms)
    def parse_nested(data, column):
        data[column] = data[column].apply(eval)  # Convert string to list
        return pd.get_dummies(data[column].explode()).groupby(level=0).sum()

    X_new = pd.concat([X_new.drop(['MedicalHistory', 'Symptoms'], axis=1),
                       parse_nested(X_new, 'MedicalHistory').add_prefix('MedHist_'),
                       parse_nested(X_new, 'Symptoms').add_prefix('Symptom_')], axis=1)

    # Encode categorical variables using saved LabelEncoders
    categorical_cols = X_new.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        with open(f'label_encoder_{col}.pkl', 'rb') as f:
            le = pickle.load(f)
        # Handle unseen labels by mapping to a default (e.g., 'missing')
        X_new[col] = X_new[col].fillna('missing')
        X_new[col] = X_new[col].map(lambda x: x if x in le.classes_ else 'missing')
        X_new[col] = le.transform(X_new[col])

    # Select only the features used during training
    X_new = X_new[selected_features]

    # Scale numerical features using saved scaler
    numerical_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
    X_new[numerical_cols] = scaler.transform(X_new[numerical_cols])

    # Impute missing values using saved imputer
    X_new = imputer.transform(X_new)

    # Make predictions and calculate detailed test metrics
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