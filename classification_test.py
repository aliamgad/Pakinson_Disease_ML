import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Load pre-trained models
models = {}
for name in ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'Voting', 'Stacking']:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)


# Load and preprocess new test data
def predict_new_data(test_file):
    test_data = pd.read_csv(test_file)
    X_new = test_data.drop(columns=['Diagnosis', 'DoctorInCharge'])

    # Handle nested data (MedicalHistory, Symptoms)
    def parse_nested(data, column):
        data[column] = data[column].apply(eval)  # Convert string to list
        return pd.get_dummies(data[column].explode()).groupby(level=0).sum()

    X_new = pd.concat([X_new.drop(['MedicalHistory', 'Symptoms'], axis=1),
                       parse_nested(X_new, 'MedicalHistory').add_prefix('MedHist_'),
                       parse_nested(X_new, 'Symptoms').add_prefix('Symptom_')], axis=1)

    # Encode categorical variables (assuming same LabelEncoder from training)
    categorical_cols = X_new.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X_new[col] = le.fit_transform(X_new[col].fillna('missing'))  # Handle missing with 'missing' label

    # Scale numerical features (assuming same scaler from training)
    numerical_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_new[numerical_cols] = scaler.fit_transform(X_new[numerical_cols])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_new = imputer.fit_transform(X_new)

    # Make predictions
    predictions = {name: model.predict(X_new) for name, model in models.items()}
    return predictions


# Example usage
if __name__ == "__main__":
    predictions = predict_new_data('new_test_data.csv')
    for name, pred in predictions.items():
        print(f"{name} predictions: {pred}")
