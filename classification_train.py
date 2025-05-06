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
# Handle nested data (MedicalHistory, Symptoms) - assuming they are strings of lists
def parse_nested(data, column):
    data[column] = data[column].apply(lambda x: eval(x) if pd.notnull(x) else [])
    return pd.get_dummies(data[column].explode()).groupby(level=0).sum()


# Apply parse_nested to the entire dataset before splitting
train_data = pd.concat([train_data.drop(['MedicalHistory', 'Symptoms'], axis=1),
                        parse_nested(train_data, 'MedicalHistory').add_prefix('MedHist_'),
                        parse_nested(train_data, 'Symptoms').add_prefix('Symptom_')], axis=1)

categorical_cols = train_data.drop(columns=['Diagnosis', 'DoctorInCharge']).select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_data[categorical_cols] = train_data[categorical_cols].fillna('missing')
train_data[categorical_cols] = ordinal_encoder.fit_transform(train_data[categorical_cols])

with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal_encoder, f)

X = train_data.drop(columns=['Diagnosis', 'DoctorInCharge'])
y = train_data['Diagnosis']


# Feature Selection
# Check if selected features file exists
# if os.path.exists('selected_features.txt'):
#     with open('selected_features.txt', 'r') as f:
#         selected_features = f.read().splitlines()
#     print("Loaded selected features from file:", selected_features)
# else: (apply the selection)


# Separate numerical and categorical features

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# Use dummy variables for categorical columns to ensure non-negative binary data
categorical_cols = X.columns[X.columns.str.contains('MedHist_|Symptom_')]
X_cat = X[categorical_cols].copy()  # Create a copy to avoid SettingWithCopyWarning
X_cat = X_cat.astype(float)  # Ensure float type for chi2
y = y.astype(float)  # Ensure y is numeric

# Validate and clean data
X_cat = np.nan_to_num(X_cat, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN/inf with 0
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN/inf with 0

# Debug: Print shapes and check for NaN
# print("X_cat shape:", X_cat.shape)
# print("y shape:", y.shape)
# print("X_cat NaN check:", np.isnan(X_cat).sum().sum())
# print("y NaN check:", np.isnan(y).sum())

# Numerical features: Use Pearson's correlation (f_classif in sklearn)
# Step 1: Remove constant features (zero variance)
variance_selector = VarianceThreshold(threshold=0.0)
X_var_filtered = variance_selector.fit_transform(X[numerical_cols])
filtered_numerical_cols = [
    col for col, keep in zip(list(numerical_cols), variance_selector.get_support()) if keep
]

# Pearson correlation
selector_num = SelectKBest(score_func=f_classif, k=5)
X_num_selected = selector_num.fit_transform(X_var_filtered, y)
selected_num_indices = selector_num.get_support(indices=True)
selected_num_features = [filtered_numerical_cols[i] for i in selected_num_indices]

print("Selected numerical features:", selected_num_features)

# Categorical features: Use Chi-Squared test
selector_cat = SelectKBest(score_func=chi2, k=5)  # Select top 5 categorical features
X_cat_selected = selector_cat.fit_transform(X_cat, y)
selected_cat_indices = selector_cat.get_support(indices=True)
selected_cat_features = X_cat.columns[selected_cat_indices].tolist() if hasattr(X_cat, 'columns') else X.columns[
    selected_cat_indices].tolist()
print("Selected categorical features:", selected_cat_features)

# Combine selected features
selected_features = selected_num_features + selected_cat_features
X_selected = X[selected_features]
numerical_cols = X_selected.select_dtypes(include=['int64', 'float64'], exclude=['string']).columns

# if os.path.exists('scaler.pkl'):
#     with open('scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
#     print("Loaded scaler from scaler.pkl")
# else: (make new scaler)

scaler = StandardScaler()
X_selected[numerical_cols] = scaler.fit_transform(X_selected[numerical_cols])

# Save for test script
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler to scaler.pkl")

# # Imputer for missing values and save/load imputer
# # if os.path.exists('imputer.pkl'):
# #     with open('imputer.pkl', 'rb') as f:
# #         imputer = pickle.load(f)
# #     print("Loaded imputer from imputer.pkl")
# # else:

imputer = SimpleImputer(strategy='mean')
X_selected[numerical_cols] = imputer.fit_transform(X_selected[numerical_cols])

# Save for test script
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print("Saved imputer to imputer.pkl")


# Save for test script
if not os.path.exists('selected_features.pkl'):
    with open('selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    print("Saved selected features to selected_features.pkl")

# Define models and hyperparameter grids
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
svm = SVC()

param_grid_log = {'C': [0.1, 1, 10]}  # try another one
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
    grid_search.fit(X_selected, y)
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
    clf.fit(X_selected, y)
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
