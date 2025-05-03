import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import time
from sklearn.metrics import accuracy_score

# Load dataset (assuming file name based on project context)
data = pd.read_csv('parkinsons_disease_data_cls.csv')

# Separate features and target
X = data.drop(columns=['Diagnosis', 'DoctorInCharge'])  # Drop confidential and target
y = data['Diagnosis']


# Preprocessing
# Handle nested data (MedicalHistory, Symptoms) - assuming they are strings of lists
def parse_nested(data, column):
    data[column] = data[column].apply(eval)  # Convert string to list
    return pd.get_dummies(data[column].explode()).groupby(level=0).sum()


X = pd.concat([X.drop(['MedicalHistory', 'Symptoms'], axis=1),
               parse_nested(X, 'MedicalHistory').add_prefix('MedHist_'),
               parse_nested(X, 'Symptoms').add_prefix('Symptom_')], axis=1)

#print(X)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna('missing'))  # Handle missing with 'missing' label

# # Scale numerical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
#
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputer for test data missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # Transform test data with same imputer

# Define models and hyperparameter grids
log_reg = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC()

param_grid_log = {'C': [0.1, 1, 10]}
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
test_times = {}
accuracies = {}

for model, param_grid in grids:
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_times[model] = time.time() - start_time
    best_models[model] = grid_search.best_estimator_
    start_time = time.time()
    y_pred = best_models[model].predict(X_test)
    test_times[model] = time.time() - start_time
    accuracies[model] = accuracy_score(y_test, y_pred)
    print(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Accuracy: {accuracies[model]}")

# Save models
for name, model in best_models.items():
    with open(f'{name.__class__.__name__}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Ensemble methods
voting_clf = VotingClassifier(
    estimators=[('lr', best_models[log_reg]), ('rf', best_models[rf]), ('svm', best_models[svm])], voting='hard')
stacking_clf = StackingClassifier(
    estimators=[('lr', best_models[log_reg]), ('rf', best_models[rf]), ('svm', best_models[svm])],
    final_estimator=LogisticRegression())

for clf, name in [(voting_clf, 'Voting'), (stacking_clf, 'Stacking')]:
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_times[name] = time.time() - start_time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_times[name] = time.time() - start_time
    accuracies[name] = accuracy_score(y_test, y_pred)
    with open(f'{name}_Classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)


# # Test script function
# def predict_new_data(test_file):
#     test_data = pd.read_csv(test_file)
#     X_new = test_data.drop(columns=['Diagnosis', 'DoctorInCharge'])
#     X_new = pd.concat([X_new.drop(['MedicalHistory', 'Symptoms'], axis=1),
#                        parse_nested(X_new, 'MedicalHistory').add_prefix('MedHist_'),
#                        parse_nested(X_new, 'Symptoms').add_prefix('Symptom_')], axis=1)
#     for col in X_new.select_dtypes(include=['object']).columns:
#         X_new[col] = le.transform(X_new[col].fillna('missing'))
#     X_new[numerical_cols] = scaler.transform(X_new[numerical_cols])
#     X_new = imputer.transform(X_new)  # Handle missing values
#     models = {}
#     for name in ['LogisticRegression', 'RandomForestClassifier', 'SVC', 'Voting', 'Stacking']:
#         with open(f'{name}.pkl', 'rb') as f:
#             models[name] = pickle.load(f)
#     predictions = {name: model.predict(X_new) for name, model in models.items()}
#     return predictions

# Example usage of test script (commented out to avoid execution)
# predictions = predict_new_data('new_test_data.csv')
# for name, pred in predictions.items():
#     print(f"{name} predictions: {pred}")
