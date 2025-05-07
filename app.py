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
        # Load train data to infer feature types
        train_data = pd.read_csv('train.csv')

        # Parse nested data as in training
        def parse_nested(data, column):
            data[column] = data[column].apply(lambda x: eval(x) if pd.notnull(x) else [])
            return pd.get_dummies(data[column].explode()).groupby(level=0).sum()

        train_data = pd.concat([
            train_data.drop(['MedicalHistory', 'Symptoms'], axis=1),
            parse_nested(train_data, 'MedicalHistory').add_prefix('MedHist_'),
            parse_nested(train_data, 'Symptoms').add_prefix('Symptom_')
        ], axis=1)

        X = train_data.drop(columns=['Diagnosis', 'DoctorInCharge'])
        numerical_cols = X[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X[selected_features].select_dtypes(include=['object']).columns.tolist()
        binary_cols = [col for col in selected_features if col.startswith('MedHist_') or col.startswith('Symptom_')]
        return numerical_cols, categorical_cols, binary_cols
    except Exception as e:
        raise Exception(f"Error processing feature types: {str(e)}")


numerical_cols, categorical_cols, binary_cols = load_feature_types()


@app.route('/')
def index():
    return render_template('index.html', features=selected_features, numerical_cols=numerical_cols,
                           categorical_cols=categorical_cols, binary_cols=binary_cols)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()

        print(input_data)
        print(selected_features)

        # Create a DataFrame with selected features
        data = {feature: [0] for feature in selected_features}  # Initialize with zeros
        df = pd.DataFrame(data)

        # Fill in the input data
        for feature, value in input_data.items():
            if feature in selected_features:
                if feature in numerical_cols:
                    df[feature] = float(value)
                elif feature in categorical_cols:
                    df[feature] = value
                elif feature in binary_cols:
                    df[feature] = 1 if value.lower() == 'on' else 0

        # Preprocess the input data
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].fillna('missing')
            df[categorical_cols] = ordinal_encoder.transform(df[categorical_cols])

        if numerical_cols:
            df[numerical_cols] = imputer.transform(df[numerical_cols])
            df[numerical_cols] = scaler.transform(df[numerical_cols])

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