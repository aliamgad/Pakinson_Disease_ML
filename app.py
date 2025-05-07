from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import sklearn
import warnings

# Check scikit-learn version
if int(sklearn.__version__.split('.')[0]) < 1:
    warnings.warn(
        f"scikit-learn version {sklearn.__version__} detected. SimpleImputer requires scikit-learn >= 1.0. Upgrading is recommended.")

app = Flask(__name__)

# Load pre-trained models and preprocessing objects
try:
    models = {}
    for name in ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'Voting_Classifier', 'Stacking_Classifier']:
        with open(f'{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('imputer.pkl', 'rb') as f:
        imputer_data = pickle.load(f)
        imputers = imputer_data['imputers']
        numerical_cols_imputed = imputer_data['numerical_cols']

    with open('ordinal_encoder.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)

    # Load selected features
    if os.path.exists('selected_features.pkl'):
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
    else:
        raise FileNotFoundError("selected_features.pkl not found")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required file: {str(e)}")
except Exception as e:
    raise Exception(f"Error loading models or preprocessing objects: {str(e)}")

# Best model (based on test accuracy from your test script)
best_model_name = 'RandomForestClassifier'  # Update based on test results if needed
best_model = models[best_model_name]

# Define feature engineering function (same as in classification_train.py and classification_test.py)
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

# Convert WeeklyPhysicalActivity (hr) to minutes
def convert_to_minutes(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    try:
        hours, minutes = map(int, time_str.split(':'))
        return (hours * 60) + minutes
    except (ValueError, AttributeError):
        return np.nan

# Define raw input columns (before feature engineering)
raw_numerical_cols = ['Age', 'BMI', 'AlcoholConsumption', 'DietQuality', 'SleepQuality',
                      'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                      'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
                      'FunctionalAssessment', 'WeeklyPhysicalActivity (hr)']
raw_categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking']
raw_nested_cols = ['MedicalHistory', 'Symptoms']

# Define feature types for selected features
def load_feature_types():
    numerical_cols = [col for col in selected_features if col in numerical_cols_imputed or col in [
        'MotorSymptomCount', 'MoCA_MotorSymptomInteraction', 'Age_UPDRS_Interaction',
        'Age_MoCA_Interaction', 'Age_FunctionalAssessment_Interaction',
        'FamilyHistory_MotorSymptom_Interaction', 'TBI_UPDRS_Interaction',
        'Tremor_and_Rigidity', 'Bradykinesia_and_PosturalInstability',
        'SleepDisorders_and_Constipation', 'UPDRS_per_Age', 'MoCA_per_Age',
        'LDL_to_HDL', 'Total_to_HDL', 'PulsePressure', 'ComorbidityScore',
        'CogMotorInteraction', 'HealthBehaviorScore'
    ]]
    categorical_cols = [col for col in selected_features if col in raw_categorical_cols]
    binary_cols = [col for col in selected_features if col.startswith('MedHist_') or col.startswith('Symptom_') or col in [
        'tremor', 'rigidity', 'bradykinesia', 'posturalinstability', 'familyhistoryparkinsons',
        'traumaticbraininjury', 'hypertension', 'diabetes', 'depression', 'stroke',
        'sleepdisorders', 'constipation'
    ]]
    return numerical_cols, categorical_cols, binary_cols

numerical_cols, categorical_cols, binary_cols = load_feature_types()

@app.route('/')
def index():
    # Pass raw input columns to the template for user input
    return render_template('index.html',
                         raw_numerical_cols=raw_numerical_cols,
                         raw_categorical_cols=raw_categorical_cols,
                         raw_nested_cols=raw_nested_cols,
                         numerical_cols=numerical_cols,
                         categorical_cols=categorical_cols,
                         binary_cols=binary_cols)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        print(input_data)

        # Create a DataFrame with raw input columns
        raw_cols = raw_numerical_cols + raw_categorical_cols + raw_nested_cols
        data = {col: [np.nan] for col in raw_cols}  # Initialize with NaN
        df = pd.DataFrame(data)

        # Fill in the input data
        for feature, value in input_data.items():
            if feature in raw_cols:
                if feature in raw_numerical_cols:
                    try:
                        df[feature] = float(value)
                    except ValueError:
                        df[feature] = np.nan
                elif feature in raw_categorical_cols:
                    df[feature] = value if value else 'missing'
                elif feature in raw_nested_cols:
                    # Expect Symptoms and MedicalHistory as comma-separated strings
                    df[feature] = [value.split(',') if value else []]

        # Convert WeeklyPhysicalActivity (hr) to minutes
        if 'WeeklyPhysicalActivity (hr)' in df.columns:
            df['WeeklyPhysicalActivity (min)'] = df['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)
            df = df.drop(columns=['WeeklyPhysicalActivity (hr)'])

        # Ensure numerical columns are numeric
        for col in raw_numerical_cols:
            if col in df.columns and col != 'WeeklyPhysicalActivity (hr)':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Impute numerical columns
        for col in numerical_cols_imputed:
            if col in df.columns and col in imputers:
                df[[col]] = imputers[col].transform(df[[col]])
            elif col in imputers:
                df[col] = imputers[col].statistics_[0]  # Use training imputed value

        # Apply feature engineering
        df = feature_engineering(df)

        # Handle nested data (MedicalHistory, Symptoms)
        def parse_nested(data, column):
            if column in data.columns:
                return pd.get_dummies(data[column].explode()).groupby(level=0).sum()
            return pd.DataFrame()

        med_hist_dummies = parse_nested(df, 'MedicalHistory').add_prefix('MedHist_')
        symptom_dummies = parse_nested(df, 'Symptoms').add_prefix('Symptom_')

        # Add missing dummy columns
        expected_dummy_cols = [col for col in selected_features if col.startswith(('MedHist_', 'Symptom_'))]
        for col in expected_dummy_cols:
            if col not in med_hist_dummies.columns and col.startswith('MedHist_'):
                med_hist_dummies[col] = 0
            if col not in symptom_dummies.columns and col.startswith('Symptom_'):
                symptom_dummies[col] = 0

        df = pd.concat([
            df.drop(['MedicalHistory', 'Symptoms'], axis=1, errors='ignore'),
            med_hist_dummies,
            symptom_dummies
        ], axis=1)

        # Encode categorical columns
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].fillna('missing')
            df[categorical_cols] = ordinal_encoder.transform(df[categorical_cols])

        # Scale numerical columns
        if numerical_cols:
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
                for feature in expected_features:
                    if feature not in df.columns:
                        df[feature] = 0
                numerical_cols_to_scale = [f for f in expected_features if f in df.columns]
            else:
                numerical_cols_to_scale = numerical_cols
            df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale].astype('float64'))

        # Align with selected features
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        df = df[selected_features]  # Keep only selected features in correct order

        # Make prediction
        prediction = best_model.predict(df)[0]
        prediction_prob = best_model.predict_proba(df)[0] if hasattr(best_model, 'predict_proba') else None

        # Format response
        result = {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'probability': [float(prob) for prob in prediction_prob] if prediction_prob is not None else None
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)