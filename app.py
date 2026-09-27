import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Feature Definitions and Defaults (derived from notebook analysis) ---
# Numerical features used in X_linear (and most of X_logistic)
NUMERICAL_FEATURES_LR = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Categorical features used in both models
CATEGORICAL_FEATURES_MAP = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

# Default values for numerical features (using mean from notebook's df.describe())
DEFAULT_NUMERICAL_VALUES = {
    'Age': 36.92,
    'DailyRate': 802.48,
    'DistanceFromHome': 9.19,
    'Education': 2.91, # Assuming 1-5 scale, mean for int is okay
    'EnvironmentSatisfaction': 2.72, # Assuming 1-4 scale, mean for int is okay
    'HourlyRate': 65.89,
    'JobInvolvement': 2.73, # Assuming 1-4 scale, mean for int is okay
    'JobLevel': 2.06, # Assuming 1-5 scale, mean for int is okay
    'JobSatisfaction': 2.73, # Assuming 1-4 scale, mean for int is okay
    'MonthlyRate': 14313.10,
    'NumCompaniesWorked': 2.69,
    'PercentSalaryHike': 15.21,
    'PerformanceRating': 3.15, # Assuming 1-4 scale, mean for int is okay
    'RelationshipSatisfaction': 2.71, # Assuming 1-4 scale, mean for int is okay
    'StockOptionLevel': 0.79, # Assuming 0-3 scale, mean for int is okay
    'TotalWorkingYears': 11.28,
    'TrainingTimesLastYear': 2.79,
    'WorkLifeBalance': 2.76, # Assuming 1-4 scale, mean for int is okay
    'YearsAtCompany': 7.00,
    'YearsInCurrentRole': 4.23,
    'YearsSinceLastPromotion': 2.19,
    'YearsWithCurrManager': 4.12,
    'MonthlyIncome': 6502.93 # Mean MonthlyIncome for general use if not predicted
}

# Default values for categorical features (using mode from notebook's df.describe(include='object'))
DEFAULT_CATEGORICAL_VALUES = {
    'BusinessTravel': 'Travel_Rarely',
    'Department': 'Research & Development',
    'EducationField': 'Life Sciences',
    'Gender': 'Male',
    'JobRole': 'Sales Executive',
    'MaritalStatus': 'Married',
    'OverTime': 'No'
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('linear_regression_model.joblib')
        log_model = joblib.load('logistic_regression_model.joblib')
        return lr_model, log_model
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'linear_regression_model.joblib' and 'logistic_regression_model.joblib' are in the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

lr_model, log_model = load_models()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Employee Analytics Dashboard", layout="wide")
st.title("Employee Analytics: Income & Attrition Prediction")
st.markdown("This dashboard allows you to predict an employee's **Monthly Income** and their likelihood of **Attrition** based on various employee characteristics.")

# --- Sidebar for User Input ---
st.sidebar.header('Employee Characteristics Input')

input_data = {}

# Input fields for numerical features
st.sidebar.subheader('Numerical Features')
for feature in NUMERICAL_FEATURES_LR:
    min_val = 0.0
    max_val = 100000.0 # Upper bound, adjust as needed
    step_val = 1.0
    format_val = "%.0f"

    if feature == 'DailyRate': min_val, max_val, step_val = 100.0, 1500.0, 1.0
    if feature == 'MonthlyRate': min_val, max_val, step_val = 2000.0, 30000.0, 1.0
    if feature == 'Age': min_val, max_val, step_val = 18.0, 60.0, 1.0
    if feature.startswith('Years'): max_val = 40.0 # Max years
    if feature.endswith('Satisfaction') or feature.startswith('JobInvolvement') or feature.startswith('Education') or feature.startswith('WorkLifeBalance') or feature.startswith('JobLevel') or feature.startswith('PerformanceRating') or feature.startswith('StockOptionLevel'):
        min_val, max_val, step_val = 1.0, 5.0, 1.0 # For 1-4 or 1-5 scales
        format_val = "%.0f"
    if feature == 'StockOptionLevel': min_val, max_val = 0.0, 3.0


    input_data[feature] = st.sidebar.number_input(
        f"**{feature.replace('_', ' ')}**",
        min_value=min_val,
        max_value=max_val,
        value=float(DEFAULT_NUMERICAL_VALUES.get(feature, min_val)), # Use default, else min
        step=step_val,
        format=format_val
    )

# Input fields for categorical features
st.sidebar.subheader('Categorical Features')
for feature, options in CATEGORICAL_FEATURES_MAP.items():
    default_value = DEFAULT_CATEGORICAL_VALUES.get(feature, options[0])
    input_data[feature] = st.sidebar.selectbox(
        f"**{feature.replace('_', ' ')}**",
        options=options,
        index=options.index(default_value) if default_value in options else 0
    )

# --- Prediction Logic ---
if st.sidebar.button('Predict'):
    # Create base DataFrame from input_data
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns matches the training data used for the preprocessors
    # These are the columns for X_linear (which is input_df before MonthlyIncome is added for LogR)
    X_linear_cols_ordered = NUMERICAL_FEATURES_LR + list(CATEGORICAL_FEATURES_MAP.keys())
    input_for_lr = input_df[X_linear_cols_ordered]

    st.subheader("Prediction Results")

    # 1. Predict Monthly Income (Linear Regression)
    st.markdown("### Predicted Monthly Income")
    predicted_income = None
    try:
        # The pipeline handles scaling and encoding internally for the LR model
        predicted_income = lr_model.predict(input_for_lr)[0]
        st.success(f"Predicted Monthly Income: **${predicted_income:,.2f}**")
    except Exception as e:
        st.error(f"Error predicting Monthly Income: {e}")
        st.warning("Using default Monthly Income for Attrition prediction.")
        predicted_income = DEFAULT_NUMERICAL_VALUES['MonthlyIncome'] # Fallback if LR fails

    # 2. Predict Attrition (Logistic Regression)
    st.markdown("### Predicted Attrition Likelihood")

    # Create input for Logistic Regression. It needs MonthlyIncome.
    input_for_log = input_df.copy()
    input_for_log['MonthlyIncome'] = predicted_income # Use predicted income for attrition

    # Ensure the order of columns matches the training data used for X_logistic
    X_logistic_cols_ordered = NUMERICAL_FEATURES_LR + ['MonthlyIncome'] + list(CATEGORICAL_FEATURES_MAP.keys())

    # Reorder columns to strictly match X_logistic during training. Necessary for ColumnTransformer consistency.
    # Note: ColumnTransformer relies on column names matching, but order can sometimes matter for some estimators.
    # Explicit reordering adds robustness.
    input_for_log = input_for_log[X_logistic_cols_ordered]

    try:
        # The pipeline handles scaling and encoding internally for the LogR model
        attrition_proba = log_model.predict_proba(input_for_log)[0][1] # Probability of 'Yes' attrition
        attrition_class = log_model.predict(input_for_log)[0]

        st.write(f"Probability of Attrition (Yes): **{attrition_proba:.2%}**")
        if attrition_class == 1:
            st.error("Predicted Attrition: **Yes** (Employee is likely to leave)")
            st.markdown("💡 *Consider targeted retention strategies based on this employee's profile.*")
        else:
            st.success("Predicted Attrition: **No** (Employee is likely to stay)")
            st.markdown("✅ *Employee is not predicted to attrite at this time.*")

        st.info(
            "*Note: The 'Yes'/'No' prediction is based on a default probability threshold of 0.5. "
            "This threshold can be adjusted based on the business costs of False Positives vs. False Negatives. "
            "A higher probability of attrition (e.g., > 20-30%) might already warrant attention, "
            "even if the classification is 'No' under the 0.5 threshold.*"
        )

    except Exception as e:
        st.error(f"Error predicting Attrition: {e}")


st.markdown("""
---
**Disclaimer:** This dashboard uses machine learning models trained on historical data.
Predictions are probabilistic and should be used as a tool to inform decisions, not as definitive forecasts.
This application does not store any user input data.
""")
