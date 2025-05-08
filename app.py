from flask import Flask, render_template, request, jsonify, json
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
        imputer = pickle.load(f)

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
best_model_name = 'Stacking_Classifier'  # Update based on test results if needed
best_model = models[best_model_name]


# Define categorical and numerical columns from selected features
def load_feature_types():
    try:
        train = pd.read_csv('train.csv')

        def parse_nested(data, column):

            data[column] = data[column].apply(eval)  # Convert string to list
            return pd.get_dummies(data[column].explode()).groupby(level=0).sum()

        train = pd.concat([train.drop(['MedicalHistory', 'Symptoms'], axis=1),
                           parse_nested(train, 'MedicalHistory').add_prefix('MedHist_'),
                           parse_nested(train, 'Symptoms').add_prefix('Symptom_')], axis=1)

        y = train['Diagnosis']
        x = train.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])

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
        return numerical_cols, categorical_cols
    except Exception as e:
        raise Exception(f"Error processing feature types: {str(e)}")


numerical_cols0, categorical_cols0 = load_feature_types()


@app.route('/')
def index():
    return render_template('index.html', features=selected_features, numerical_cols=numerical_cols0,
                           categorical_cols=categorical_cols0)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        medical_history = json.loads(input_data.pop('MedicalHistory', '{}'))
        symptoms = json.loads(input_data.pop('Symptoms', '{}'))

        input_data = pd.DataFrame([input_data])

        med_hist_df = pd.DataFrame([{
            f'MedHist_{key}': 'Yes' if value == 'Yes' else 'No'
            for key, value in medical_history.items()
        }])

        symptoms_df = pd.DataFrame([{
            f'Symptom_{key}': 'Yes' if value == 'Yes' else 'No'
            for key, value in symptoms.items()
        }])

        input_data = pd.concat([input_data, med_hist_df, symptoms_df], axis=1)

        expected_columns = list(numerical_cols0) + list(categorical_cols0)

        # print("Expected columns:", expected_columns)

        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 'No' if col.startswith('MedHist_') or col.startswith('Symptom_') else np.nan

        # print(input_data)

        x = input_data
        # print(x)

        # Handle nested data (MedicalHistory, Symptoms)

        iter = ['MedHist_Depression', 'MedHist_Diabetes',
                'MedHist_FamilyHistoryParkinsons', 'MedHist_Hypertension',
                'MedHist_Stroke', 'MedHist_TraumaticBrainInjury',
                'Symptom_Bradykinesia', 'Symptom_Constipation',
                'Symptom_PosturalInstability', 'Symptom_Rigidity',
                'Symptom_SleepDisorders', 'Symptom_SpeechProblems', 'Symptom_Tremor']

        for col in iter:
            x[col] = x[col].astype('object')

        iter = ['Age', 'BMI', 'AlcoholConsumption', 'DietQuality', 'SleepQuality',
                'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
                'FunctionalAssessment']

        for col in iter:
            x[col] = pd.to_numeric(x[col], errors='coerce')

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
        # print(selected_features)
        x = x[selected_features]
        # Make prediction
        prediction = best_model.predict(x)[0]
        prediction_prob = best_model.predict_proba(x)[0] if hasattr(best_model, 'predict_proba') else None

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
