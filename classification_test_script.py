import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from ast import literal_eval # For parsing nested lists

# --- IMPORTANT: FeatureEngineerAndParser Class Definition ---
# This class MUST be defined exactly as it was when the pipeline was trained and saved.
# It's copied here from your provided training script.
class FeatureEngineerAndParser(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Store unique values for one-hot encoding during fit
        self.medical_history_categories_ = None
        self.symptoms_categories_ = None
        # Store median/mean values for imputation/normalization if needed within engineering
        self.age_median_ = None
        self.updrs_median_ = None
        self.moca_median_ = None
        self.func_median_ = None
        self.ldl_median_ = None
        self.hdl_median_ = None
        self.total_chol_median_ = None
        self.systolic_median_ = None
        self.diastolic_median_ = None
        self.diet_median_ = None
        self.sleep_median_ = None
        self.weekly_activity_median_ = None
        self.diet_max_ = None
        self.sleep_max_ = None
        self.weekly_activity_max_ = None


    def fit(self, X, y=None):
        X_copy = X.copy() # Work on a copy

        # 1. Convert 'WeeklyPhysicalActivity (hr)' to hours (float)
        if 'WeeklyPhysicalActivity (hr)' in X_copy.columns:
            def time_string_to_hours(time_str):
                if pd.isnull(time_str):
                    return np.nan
                try:
                    parts = str(time_str).split(':')
                    if len(parts) == 2:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        return hours + minutes / 60
                    elif len(parts) == 1:
                         return int(parts[0])
                    else:
                        return np.nan
                except ValueError:
                    return np.nan
            X_copy.loc[:, 'WeeklyPhysicalActivity (hr)'] = X_copy['WeeklyPhysicalActivity (hr)'].apply(time_string_to_hours)
            self.weekly_activity_median_ = X_copy['WeeklyPhysicalActivity (hr)'].median()
            self.weekly_activity_max_ = X_copy['WeeklyPhysicalActivity (hr)'].max()


        # 2. Parse nested features and store categories for OHE
        if 'MedicalHistory' in X_copy.columns:
            parsed_medhist = X_copy['MedicalHistory'].apply(
                lambda x: [str(item) for item in literal_eval(x)] if pd.notnull(x) and isinstance(x, str) else []
            )
            all_medhist_items = [item for sublist in parsed_medhist for item in sublist]
            self.medical_history_categories_ = sorted(list(set(all_medhist_items)))

        if 'Symptoms' in X_copy.columns:
            parsed_symptoms = X_copy['Symptoms'].apply(
                lambda x: [str(item) for item in literal_eval(x)] if pd.notnull(x) and isinstance(x, str) else []
            )
            all_symptoms_items = [item for sublist in parsed_symptoms for item in sublist]
            self.symptoms_categories_ = sorted(list(set(all_symptoms_items)))

        # 3. Store medians/means for engineered features
        if 'Age' in X_copy.columns: self.age_median_ = X_copy['Age'].median()
        if 'UPDRS' in X_copy.columns: self.updrs_median_ = X_copy['UPDRS'].median()
        if 'MoCA' in X_copy.columns: self.moca_median_ = X_copy['MoCA'].median()
        if 'FunctionalAssessment' in X_copy.columns: self.func_median_ = X_copy['FunctionalAssessment'].median()
        if 'CholesterolLDL' in X_copy.columns: self.ldl_median_ = X_copy['CholesterolLDL'].median()
        if 'CholesterolHDL' in X_copy.columns: self.hdl_median_ = X_copy['CholesterolHDL'].median()
        if 'CholesterolTotal' in X_copy.columns: self.total_chol_median_ = X_copy['CholesterolTotal'].median()
        if 'SystolicBP' in X_copy.columns: self.systolic_median_ = X_copy['SystolicBP'].median()
        if 'DiastolicBP' in X_copy.columns: self.diastolic_median_ = X_copy['DiastolicBP'].median()
        if 'DietQuality' in X_copy.columns:
            self.diet_median_ = X_copy['DietQuality'].median()
            self.diet_max_ = X_copy['DietQuality'].max()
        if 'SleepQuality' in X_copy.columns:
            self.sleep_median_ = X_copy['SleepQuality'].median()
            self.sleep_max_ = X_copy['SleepQuality'].max()
        return self

    def transform(self, X):
        X_copy = X.copy() # Work on a copy

        # 1. Convert 'WeeklyPhysicalActivity (hr)' to hours (float)
        if 'WeeklyPhysicalActivity (hr)' in X_copy.columns:
            def time_string_to_hours(time_str):
                 if pd.isnull(time_str):
                    return np.nan
                 try:
                    parts = str(time_str).split(':')
                    if len(parts) == 2:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        return hours + minutes / 60
                    elif len(parts) == 1:
                         return int(parts[0])
                    else:
                        return np.nan
                 except ValueError:
                    return np.nan
            X_copy.loc[:, 'WeeklyPhysicalActivity (hr)'] = X_copy['WeeklyPhysicalActivity (hr)'].apply(time_string_to_hours)

        # 2. Parse nested features and create OHE columns
        medhist_df = pd.DataFrame(index=X_copy.index)
        if 'MedicalHistory' in X_copy.columns and self.medical_history_categories_ is not None:
            parsed_medhist = X_copy['MedicalHistory'].apply(
                lambda x: [str(item) for item in literal_eval(x)] if pd.notnull(x) and isinstance(x, str) else []
            )
            exploded = parsed_medhist.explode().fillna('missing_category_placeholder')
            dummies = pd.get_dummies(exploded, prefix='MedicalHistory')
            for cat in self.medical_history_categories_:
                 col_name = f'MedicalHistory_{cat}'
                 if col_name not in dummies.columns:
                     dummies[col_name] = 0
            if 'MedicalHistory_missing_category_placeholder' in dummies.columns:
                 dummies = dummies.drop(columns='MedicalHistory_missing_category_placeholder')
            dummies = dummies.groupby(level=0).sum()
            # Ensure consistent column order and presence as during fit
            medhist_cols_ordered = [f'MedicalHistory_{cat}' for cat in self.medical_history_categories_]
            medhist_df = dummies.reindex(columns=medhist_cols_ordered, fill_value=0)


        symptom_df = pd.DataFrame(index=X_copy.index)
        if 'Symptoms' in X_copy.columns and self.symptoms_categories_ is not None:
            parsed_symptoms = X_copy['Symptoms'].apply(
                lambda x: [str(item) for item in literal_eval(x)] if pd.notnull(x) and isinstance(x, str) else []
            )
            exploded = parsed_symptoms.explode().fillna('missing_category_placeholder')
            dummies = pd.get_dummies(exploded, prefix='Symptoms')
            for cat in self.symptoms_categories_:
                 col_name = f'Symptoms_{cat}'
                 if col_name not in dummies.columns:
                     dummies[col_name] = 0
            if 'Symptoms_missing_category_placeholder' in dummies.columns:
                 dummies = dummies.drop(columns='Symptoms_missing_category_placeholder')
            dummies = dummies.groupby(level=0).sum()
            # Ensure consistent column order and presence as during fit
            symptom_cols_ordered = [f'Symptoms_{cat}' for cat in self.symptoms_categories_]
            symptom_df = dummies.reindex(columns=symptom_cols_ordered, fill_value=0)


        cols_to_drop_nested = ['MedicalHistory', 'Symptoms']
        X_copy = X_copy.drop(columns=cols_to_drop_nested, errors='ignore')

        X_copy = pd.concat([X_copy, medhist_df], axis=1)
        X_copy = pd.concat([X_copy, symptom_df], axis=1)

        # 3. Feature engineering
        motor_symptoms = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability']
        # Create symptom columns if they don't exist from OHE (e.g., if a symptom was never in training data for Symptoms column)
        # but is expected as a direct column from the original CSV.
        # However, the current logic relies on OHE from 'Symptoms' list for these.
        # If 'Tremor', 'Rigidity' etc. are direct columns in the input CSV and NOT from the 'Symptoms' list,
        # this part needs adjustment to use those direct columns.
        # Assuming they are generated from 'Symptoms' list parsing:
        existing_motor_symptom_cols = [f'Symptoms_{s}' for s in motor_symptoms if f'Symptoms_{s}' in X_copy.columns]

        for col in existing_motor_symptom_cols: # Ensure numeric and fill NaNs if any (should be 0/1 from OHE)
            X_copy.loc[:, col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)

        if existing_motor_symptom_cols:
            X_copy.loc[:, 'MotorSymptomCount'] = X_copy[existing_motor_symptom_cols].sum(axis=1)
        else:
            X_copy.loc[:, 'MotorSymptomCount'] = 0

        if 'MoCA' in X_copy.columns and 'MotorSymptomCount' in X_copy.columns:
             moca_filled = X_copy['MoCA'].fillna(self.moca_median_ if self.moca_median_ is not None else 0)
             X_copy.loc[:, 'MoCA_MotorSymptomInteraction'] = moca_filled * X_copy['MotorSymptomCount']

        if 'Age' in X_copy.columns:
            age_filled = X_copy['Age'].fillna(self.age_median_ if self.age_median_ is not None else 0)
            if 'UPDRS' in X_copy.columns:
                updrs_filled = X_copy['UPDRS'].fillna(self.updrs_median_ if self.updrs_median_ is not None else 0)
                X_copy.loc[:, 'Age_UPDRS_Interaction'] = age_filled * updrs_filled
            if 'MoCA' in X_copy.columns:
                moca_filled = X_copy['MoCA'].fillna(self.moca_median_ if self.moca_median_ is not None else 0)
                X_copy.loc[:, 'Age_MoCA_Interaction'] = age_filled * moca_filled
            if 'FunctionalAssessment' in X_copy.columns:
                func_filled = X_copy['FunctionalAssessment'].fillna(self.func_median_ if self.func_median_ is not None else 0)
                X_copy.loc[:, 'Age_FunctionalAssessment_Interaction'] = age_filled * func_filled

        # For direct columns like 'FamilyHistoryParkinsons', 'TraumaticBrainInjury'
        # These are NOT from the 'Symptoms' or 'MedicalHistory' lists in the example data.
        hist_cols_direct = ['FamilyHistoryParkinsons', 'TraumaticBrainInjury']
        for col in hist_cols_direct:
            if col in X_copy.columns:
                 X_copy.loc[:, col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)
            else: # Add column if missing and fill with 0 (important for consistency)
                X_copy.loc[:, col] = 0


        if 'MotorSymptomCount' in X_copy.columns and 'FamilyHistoryParkinsons' in X_copy.columns:
            X_copy.loc[:, 'FamilyHistory_MotorSymptom_Interaction'] = X_copy['FamilyHistoryParkinsons'] * X_copy['MotorSymptomCount']
        if 'UPDRS' in X_copy.columns and 'TraumaticBrainInjury' in X_copy.columns:
            updrs_filled = X_copy['UPDRS'].fillna(self.updrs_median_ if self.updrs_median_ is not None else 0)
            X_copy.loc[:, 'TBI_UPDRS_Interaction'] = X_copy['TraumaticBrainInjury'] * updrs_filled

        # Symptom Combinations (assuming these are from the parsed 'Symptoms' list)
        # Example: 'Symptoms_Tremor', 'Symptoms_Rigidity'
        symptom_pairs = [('Tremor','Rigidity'), ('Bradykinesia','PosturalInstability'), ('SleepDisorders','Constipation')]
        for s1, s2 in symptom_pairs:
            s1_col = f'Symptoms_{s1}'
            s2_col = f'Symptoms_{s2}'
            if s1_col in X_copy.columns and s2_col in X_copy.columns:
                # Values should be 0/1 from OHE, fillna(0) just in case
                col_a_numeric = pd.to_numeric(X_copy[s1_col], errors='coerce').fillna(0)
                col_b_numeric = pd.to_numeric(X_copy[s2_col], errors='coerce').fillna(0)
                X_copy.loc[:, f"{s1}_and_{s2}"] = col_a_numeric * col_b_numeric
            else: # Ensure feature exists even if symptoms not present
                X_copy.loc[:, f"{s1}_and_{s2}"] = 0


        if 'Age' in X_copy.columns:
            age_filled = X_copy['Age'].fillna(self.age_median_ if self.age_median_ is not None else 1e-6)
            if 'UPDRS' in X_copy.columns:
                updrs_filled = X_copy['UPDRS'].fillna(self.updrs_median_ if self.updrs_median_ is not None else 0)
                X_copy.loc[:, 'UPDRS_per_Age'] = updrs_filled / (age_filled + 1e-6)
            if 'MoCA' in X_copy.columns:
                moca_filled = X_copy['MoCA'].fillna(self.moca_median_ if self.moca_median_ is not None else 0)
                X_copy.loc[:, 'MoCA_per_Age'] = moca_filled / (age_filled + 1e-6)

        if 'CholesterolLDL' in X_copy.columns and 'CholesterolHDL' in X_copy.columns:
            ldl_filled = X_copy['CholesterolLDL'].fillna(self.ldl_median_ if self.ldl_median_ is not None else 0)
            hdl_filled = X_copy['CholesterolHDL'].fillna(self.hdl_median_ if self.hdl_median_ is not None else 1e-6)
            X_copy.loc[:, 'LDL_to_HDL'] = ldl_filled / (hdl_filled + 1e-6)
        if 'CholesterolTotal' in X_copy.columns and 'CholesterolHDL' in X_copy.columns:
            total_filled = X_copy['CholesterolTotal'].fillna(self.total_chol_median_ if self.total_chol_median_ is not None else 0)
            hdl_filled = X_copy['CholesterolHDL'].fillna(self.hdl_median_ if self.hdl_median_ is not None else 1e-6)
            X_copy.loc[:, 'Total_to_HDL'] = total_filled / (hdl_filled + 1e-6)
        if 'SystolicBP' in X_copy.columns and 'DiastolicBP' in X_copy.columns:
            systolic_filled = X_copy['SystolicBP'].fillna(self.systolic_median_ if self.systolic_median_ is not None else 0)
            diastolic_filled = X_copy['DiastolicBP'].fillna(self.diastolic_median_ if self.diastolic_median_ is not None else 0)
            X_copy.loc[:, 'PulsePressure'] = systolic_filled - diastolic_filled

        # Comorbidity Score for direct history columns
        # These are direct columns in the input CSV, not from 'MedicalHistory' list
        comorb_cols_direct = ['FamilyHistoryParkinsons','TraumaticBrainInjury','Hypertension','Diabetes','Depression','Stroke']
        existing_comorb_direct = []
        for col in comorb_cols_direct:
            if col in X_copy.columns:
                X_copy.loc[:, col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)
                existing_comorb_direct.append(col)
            else: # Add column if missing and fill with 0
                X_copy.loc[:, col] = 0
                existing_comorb_direct.append(col) # Still include it for sum

        if existing_comorb_direct:
            X_copy.loc[:, 'ComorbidityScore'] = X_copy[existing_comorb_direct].sum(axis=1)
        else:
             X_copy.loc[:, 'ComorbidityScore'] = 0

        if 'MoCA' in X_copy.columns and 'FunctionalAssessment' in X_copy.columns:
            moca_filled = X_copy['MoCA'].fillna(self.moca_median_ if self.moca_median_ is not None else 0)
            func_filled = X_copy['FunctionalAssessment'].fillna(self.func_median_ if self.func_median_ is not None else 0)
            X_copy.loc[:, 'CogMotorInteraction'] = moca_filled * func_filled

        behavior_cols_list = ['DietQuality','SleepQuality','WeeklyPhysicalActivity (hr)']
        existing_behavior = []
        for c in behavior_cols_list:
            if c in X_copy.columns:
                X_copy.loc[:, c] = pd.to_numeric(X_copy[c], errors='coerce').fillna(getattr(self, f"{c.replace(' (hr)', '').lower()}_median_", 0))
                max_val = getattr(self, f"{c.replace(' (hr)', '').lower()}_max_", 1)
                max_val = 1 if pd.isna(max_val) or max_val == 0 else max_val # Handle potential NaN or zero max_val
                X_copy.loc[:, c] = X_copy[c] / max_val
                existing_behavior.append(c)
            else: # Add column if missing, fill with 0 (normalized)
                X_copy.loc[:, c] = 0
                existing_behavior.append(c)


        if existing_behavior:
            X_copy.loc[:, 'HealthBehaviorScore'] = X_copy[existing_behavior].sum(axis=1)
        else:
             X_copy.loc[:, 'HealthBehaviorScore'] = 0

        # Ensure all columns that were present after fit_transform are present here,
        # in the same order, and fill any missing ones with appropriate values (e.g., 0 or median).
        # This is crucial if the test set has a different shape or missing columns
        # compared to what the pipeline was trained on.
        # The `reindex` method can be useful here if you have a list of `fitted_columns_`.
        # For now, the preprocessor within the pipeline should handle most of this,
        # but explicit column creation/ordering here adds robustness.

        return X_copy

def load_testing_data(csv_file_path, target_column_name):
    """
    Loads testing data from a CSV file and separates features and target.

    Args:
        csv_file_path (str): Path to the CSV file.
        target_column_name (str): Name of the target variable column.

    Returns:
        tuple: (pd.DataFrame, pd.Series) or (None, None) if an error occurs.
               Returns X_test (features) and y_test (target).
    """
    try:
        df_test = pd.read_csv(csv_file_path)
        print(f"Successfully loaded testing data from: {csv_file_path}")
        print(f"Testing data shape: {df_test.shape}")

        # Drop identifiers if they exist, similar to training script
        cols_to_drop_initial = ['PatientID', 'DoctorInCharge']
        existing_cols_to_drop = [col for col in cols_to_drop_initial if col in df_test.columns]
        if existing_cols_to_drop:
            print(f"Dropping identifier columns from test data: {existing_cols_to_drop}")
            df_test.drop(columns=existing_cols_to_drop, inplace=True)

        if target_column_name not in df_test.columns:
            print(f"Error: Target column '{target_column_name}' not found in the testing data.")
            return None, None

        X_test_unseen = df_test.drop(columns=[target_column_name])
        y_test_unseen = df_test[target_column_name]
        print(f"Features (X_test_unseen) shape: {X_test_unseen.shape}")
        print(f"Target (y_test_unseen) shape: {y_test_unseen.shape}")
        return X_test_unseen, y_test_unseen
    except FileNotFoundError:
        print(f"Error: Testing data file not found at {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading testing data: {e}")
        return None, None

def load_pipeline(pipeline_path):
    """
    Loads a saved scikit-learn pipeline from a .pkl file.

    Args:
        pipeline_path (str): Path to the .pkl pipeline file.

    Returns:
        sklearn.pipeline.Pipeline: The loaded pipeline, or None if an error occurs.
    """
    try:
        loaded_pipeline = joblib.load(pipeline_path)
        print(f"Successfully loaded pipeline from: {pipeline_path}")
        return loaded_pipeline
    except FileNotFoundError:
        print(f"Error: Pipeline file not found at {pipeline_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the pipeline: {e}")
        return None

def main():
    test_data_path = '/parkinsons_disease_data_cls.csv'
    # Path to your saved pipeline .pkl file
    # Example: 'parkinsons_prediction_pipeline_voting_classifier.pkl'
    saved_pipeline_path = '/parkinsons_prediction_pipeline_voting_classifier.pkl'
    # Name of the target variable column in your dataset
    target_variable = 'Diagnosis'
    # --- End Configuration ---

    print("--- Starting Pipeline Testing Script ---")

    # 1. Load the unseen testing data
    print("\nStep 1: Loading unseen testing data...")
    X_test_unseen, y_test_unseen = load_testing_data(test_data_path, target_variable)

    if X_test_unseen is None or y_test_unseen is None:
        print("Exiting due to issues with loading testing data.")
        return

    # 2. Load the saved pipeline
    print("\nStep 2: Loading the saved pipeline...")
    pipeline = load_pipeline(saved_pipeline_path)

    if pipeline is None:
        print("Exiting due to issues with loading the pipeline.")
        return

    # 3. Make predictions on the unseen test data
    print("\nStep 3: Making predictions on the unseen test data...")
    try:
        predictions = pipeline.predict(X_test_unseen)
        print("Predictions made successfully.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        print("This might be due to inconsistencies between the training data and the new test data structure,")
        print("or issues within the pipeline's transformers when encountering new data patterns.")
        print("Ensure the test data has the exact same columns (excluding target) and data types as the training data fed into the pipeline.")
        return

    # 4. Calculate and display accuracy
    print("\nStep 4: Calculating accuracy...")
    try:
        accuracy = accuracy_score(y_test_unseen, predictions)
        print(f"\n--- Test Results ---")
        print(f"Accuracy of the pipeline on the unseen testing dataset: {accuracy:.4f}")
        print(f"Number of test samples: {len(y_test_unseen)}")
        print(f"Number of correct predictions: {int(accuracy * len(y_test_unseen))}")
    except Exception as e:
        print(f"An error occurred while calculating accuracy: {e}")

    print("\n--- Pipeline Testing Script Finished ---")

if __name__ == '__main__':
    main()