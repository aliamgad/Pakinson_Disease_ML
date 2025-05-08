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

# Load dataset (assuming file name based on project context)
# data = pd.read_csv('parkinsons_disease_data_cls.csv')

# Check if train.csv and test.csv exist, use them if available
# if os.path.exists('train.csv') and os.path.exists('test.csv'):

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# else:
#     # Split dataset into train and test (80% train, 20% test)
#     train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#     # Save the splits to CSV files
#     train_data.to_csv('train.csv', index=False)
#     test_data.to_csv('test.csv', index=False)


# Preprocessing

# 1- Fixing Data types and shape

def parse_nested(data, column):
    data[column] = data[column].apply(lambda x: eval(x) if pd.notnull(x) else [])
    return pd.get_dummies(data[column].explode()).groupby(level=0).sum()


# Apply parse_nested to the entire dataset before splitting
train_data = pd.concat([train_data.drop(['MedicalHistory', 'Symptoms'], axis=1),
                        parse_nested(train_data, 'MedicalHistory').add_prefix('MedHist_'),
                        parse_nested(train_data, 'Symptoms').add_prefix('Symptom_')], axis=1)

y = train_data['Diagnosis']
x = train_data.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])

# Feature Selection
# Check if selected features file exists
# if os.path.exists('selected_features.txt'):
#     with open('selected_features.txt', 'r') as f:
#         selected_features = f.read().splitlines()
#     print("Loaded selected features from file:", selected_features)
# else: (apply the selection)

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

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# # Imputer for missing values and save/load imputer
# # if os.path.exists('imputer.pkl'):
# #     with open('imputer.pkl', 'rb') as f:
# #         imputer = pickle.load(f)
# #     print("Loaded imputer from imputer.pkl")
# # else:

# 2- Imputing

imputer = SimpleImputer(strategy='mean')
imputer.fit(x[numerical_cols])
x[numerical_cols] = imputer.transform(x[numerical_cols])
x[categorical_cols] = x[categorical_cols].fillna('missing')

# Save for test script
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("Saved imputer to imputer.pkl")

# 3- Encoding categorical features

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
x[categorical_cols] = ordinal_encoder.fit_transform(x[categorical_cols])

with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal_encoder, f)

# if os.path.exists('scaler.pkl'):
#     with open('scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
#     print("Loaded scaler from scaler.pkl")
# else: (make new scaler)


# 4- Scaling

scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

# Save for test script
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler to scaler.pkl")

# 5- Feature Selection

variance_selector = VarianceThreshold(threshold=0.0)
X_var_filtered = variance_selector.fit_transform(x[numerical_cols])
filtered_numerical_cols = [
    col for col, keep in zip(list(numerical_cols), variance_selector.get_support()) if keep
]

print("Filtered numerical columns:", filtered_numerical_cols)

selector_num = SelectKBest(score_func=f_classif, k=5)
X_num_selected = selector_num.fit_transform(X_var_filtered, y)
selected_num_indices = selector_num.get_support(indices=True)
selected_num_features = [filtered_numerical_cols[i] for i in selected_num_indices]

print("Selected numerical features:", selected_num_features)

# Categorical features: Use Chi-Squared test
selector_cat = SelectKBest(score_func=chi2, k=5)  # Select top 5 categorical features
X_cat_selected = selector_cat.fit_transform(x[categorical_cols], y)
selected_cat_indices = selector_cat.get_support(indices=True)
selected_cat_features = x[categorical_cols].columns[selected_cat_indices].tolist() if hasattr(x[categorical_cols],
                                                                                              'columns') else x.columns[
    selected_cat_indices].tolist()

print("Selected categorical features:", selected_cat_features)

# Combine selected features
selected_features = selected_num_features + selected_cat_features

# Save for test script
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print("Saved selected features to selected_features.pkl")

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

# Hyperparameter tuning with cross-validation (cv=5)
grids = [
    (log_reg, param_grid_log),
    (rf, param_grid_rf),
    (svm, param_grid_svm)
]
best_models = {}
train_times = {}
accuracies = {}
best_score_overall = -1
best_model_name_overall = None

for model, param_grid in grids:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # 5-fold cross-validation
    start_time = time.time()
    grid_search.fit(x[selected_features], y)
    train_times[model.__class__.__name__] = time.time() - start_time

    # Print all trial results
    print(f"\nAll trial results for {model.__class__.__name__}:")
    results = grid_search.cv_results_
    for mean_score, std_score, params in zip(
            results['mean_test_score'],
            results['std_test_score'],
            results['params']
    ):
        print(f"Parameters: {params}, Mean Accuracy: {mean_score:.4f}, Std Dev: {std_score:.4f}")

    # Still print the best parameters for reference
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
    clf.fit(x[selected_features], y)
    train_times[name] = time.time() - start_time

# Save models with consistent naming
for name, model in best_models.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

for clf, name in [(voting_clf, 'Voting_Classifier'), (stacking_clf, 'Stacking_Classifier')]:
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(clf, f)

# Get test accuracies from test script
from classification_test import evaluate_test_data  # Import function from test script

test_accuracies, test_times = evaluate_test_data('test.csv')

# Update accuracies dictionary with test accuracies
accuracies.update(test_accuracies)

# Generate bar graphs (excluding test time, which will be in test script)
# Classification Accuracy
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.title('Classification Accuracy of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('classification_accuracy.png')

# Training Time
plt.figure(figsize=(8, 6))
plt.bar(train_times.keys(), train_times.values(), color='lightgreen')
plt.title('Total Training Time of Models')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('training_time.png')
