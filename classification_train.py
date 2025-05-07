import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
import pickle
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Load dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Convert WeeklyPhysicalActivity (hr) from hours:minutes to total minutes
def convert_to_minutes(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    try:
        hours, minutes = map(int, time_str.split(':'))
        return (hours * 60) + minutes
    except (ValueError, AttributeError):
        return np.nan


train_data['WeeklyPhysicalActivity (min)'] = train_data['WeeklyPhysicalActivity (hr)'].apply(convert_to_minutes)
train_data = train_data.drop(columns=['WeeklyPhysicalActivity (hr)'])

# Ensure other numerical columns are numeric
numerical_cols_expected = ['Age', 'BMI', 'AlcoholConsumption', 'DietQuality', 'SleepQuality',
                           'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                           'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS', 'MoCA',
                           'FunctionalAssessment', 'WeeklyPhysicalActivity (min)']
for col in numerical_cols_expected:
    if col in train_data.columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

# Debug: Print data types and sample values
print("Data types before imputation:")
print(train_data[numerical_cols_expected].dtypes)
print("Sample values:")
print(train_data[numerical_cols_expected].head())

# Imputation (before feature engineering)
SKEWNESS_THRESHOLD = 0.5
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
imputers = {}
imputed_values = {}
train_data_imputed = train_data.copy()

# Impute numerical columns
for col in numerical_cols:
    skewness = train_data[col].skew(skipna=True)
    strategy = 'median' if abs(skewness) > SKEWNESS_THRESHOLD else 'mean'
    imputer = SimpleImputer(strategy=strategy)
    train_data_imputed[[col]] = imputer.fit_transform(train_data[[col]])
    imputers[col] = imputer
    imputed_values[col] = imputer.statistics_[0]
    print(f"Column '{col}': skewness={skewness:.2f}, using {strategy}")

# Impute categorical columns (fill with 'missing')
categorical_cols = train_data.drop(columns=['Diagnosis', 'DoctorInCharge']).select_dtypes(include=['object']).columns
train_data_imputed[categorical_cols] = train_data[categorical_cols].fillna('missing')

# Save imputers
with open('imputer.pkl', 'wb') as f:
    pickle.dump({'imputers': imputers, 'imputed_values': imputed_values, 'numerical_cols': numerical_cols}, f)
print("Saved imputers and imputed values to imputer.pkl")


# Feature Engineering Function
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


# Apply feature engineering
train_data = feature_engineering(train_data_imputed)


# Preprocessing: Parse MedicalHistory and Symptoms
def parse_nested(data, column):
    if column in data.columns:
        data[column] = data[column].apply(lambda x: eval(x) if pd.notnull(x) else [])
        dummies = pd.get_dummies(data[column].explode()).groupby(level=0).sum()
        return dummies
    return pd.DataFrame()


# Preprocessing: Parse MedicalHistory and Symptoms
train_data = pd.concat([
    train_data.drop(['MedicalHistory', 'Symptoms'], axis=1, errors='ignore'),
    parse_nested(train_data, 'MedicalHistory').add_prefix('MedHist_'),
    parse_nested(train_data, 'Symptoms').add_prefix('Symptom_')
], axis=1)

# Encode categorical columns
categorical_cols = train_data.drop(columns=['Diagnosis', 'DoctorInCharge']).select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[categorical_cols] = ordinal_encoder.fit_transform(train_data[categorical_cols])

# Save ordinal encoder
with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal_encoder, f)

# Prepare features and target
X = train_data.drop(columns=['Diagnosis', 'DoctorInCharge'])
y = train_data['Diagnosis'].astype(float)

# After scaling
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save scaler with feature names
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler to scaler.pkl")

# Feature Selection
# Numerical and categorical features
categorical_cols = X.columns[X.columns.str.contains('MedHist_|Symptom_')]
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.difference(categorical_cols)

# Remove constant features
variance_selector = VarianceThreshold(threshold=0.0)
X_var_filtered = variance_selector.fit_transform(X[numerical_cols])
filtered_numerical_cols = [
    col for col, keep in zip(list(numerical_cols), variance_selector.get_support()) if keep
]

# Numerical features: f_classif
selector_num = SelectKBest(score_func=f_classif, k=5)
X_num_selected = selector_num.fit_transform(X_var_filtered, y)
selected_num_indices = selector_num.get_support(indices=True)
selected_num_features = [filtered_numerical_cols[i] for i in selected_num_indices]
print("Selected numerical features:", selected_num_features)

# Categorical features: chi2
X_cat = X[categorical_cols].copy().astype(float)
X_cat = np.nan_to_num(X_cat, nan=0.0, posinf=0.0, neginf=0.0)
selector_cat = SelectKBest(score_func=chi2, k=5)
X_cat_selected = selector_cat.fit_transform(X_cat, y)
selected_cat_indices = selector_cat.get_support(indices=True)
selected_cat_features = [categorical_cols[i] for i in selected_cat_indices]
print("Selected categorical features:", selected_cat_features)

# Combine selected features
selected_features = selected_num_features + selected_cat_features
X_selected = X[selected_features]

# Save selected features
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
with open('selected_features.txt', 'w') as f:  # Also save as text for debugging
    f.write('\n'.join(selected_features))
print("Saved selected features to selected_features.pkl and selected_features.txt")

# Define models and hyperparameter grids
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
svm = SVC()

param_grid_log = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['saga']
}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

# Hyperparameter tuning
grids = [
    (log_reg, param_grid_log),
    (rf, param_grid_rf),
    (svm, param_grid_svm)
]
best_models = {}
train_times = {}
accuracies = {}

for model, param_grid in grids:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(X_selected, y)
    train_times[model.__class__.__name__] = time.time() - start_time

    print(f"\nAll trial results for {model.__class__.__name__}:")
    results = grid_search.cv_results_
    for mean_score, std_score, params in zip(
            results['mean_test_score'],
            results['std_test_score'],
            results['params']
    ):
        print(f"Parameters: {params}, Mean Accuracy: {mean_score:.4f}, Std Dev: {std_score:.4f}")

    print(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}")
    best_models[model.__class__.__name__] = grid_search.best_estimator_

# Ensemble methods
voting_clf = VotingClassifier(estimators=[
    ('lr', best_models['LogisticRegression']),
    ('rf', best_models['RandomForestClassifier']),
    ('svm', best_models['SVC'])
], voting='hard')
stacking_clf = StackingClassifier(estimators=[
    ('lr', best_models['LogisticRegression']),
    ('rf', best_models['RandomForestClassifier']),
    ('svm', best_models['SVC'])
], final_estimator=LogisticRegression())

for clf, name in [(voting_clf, 'Voting_Classifier'), (stacking_clf, 'Stacking_Classifier')]:
    start_time = time.time()
    clf.fit(X_selected, y)
    train_times[name] = time.time() - start_time

# Save models
for name, model in best_models.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

for clf, name in [(voting_clf, 'Voting_Classifier'), (stacking_clf, 'Stacking_Classifier')]:
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(clf, f)

# Get test accuracies (assuming classification_test.py exists)
from classification_test import evaluate_test_data

test_accuracies, test_times = evaluate_test_data('test.csv')
accuracies.update(test_accuracies)

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.title('Classification Accuracy of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('classification_accuracy.png')

plt.figure(figsize=(8, 6))
plt.bar(train_times.keys(), train_times.values(), color='lightgreen')
plt.title('Total Training Time of Models')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_time.png')