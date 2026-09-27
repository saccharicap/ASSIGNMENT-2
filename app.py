import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the models
@st.cache_resource
def load_models():
    lr_model = joblib.load('linear_regression_model.joblib')
    log_model = joblib.load('logistic_regression_model.joblib')
    return lr_model, log_model

lr_model, log_model = load_models()

# --- Streamlit App ----
st.set_page_config(page_title="Employee Analytics Dashboard", layout="wide")
st.title("Employee Analytics: Income & Attrition Prediction")

st.markdown("This dashboard allows you to predict an employee's **Monthly Income** and their likelihood of **Attrition** based on various employee characteristics.")

# --- Feature Definitions (from previous analysis) ---
numerical_features_template = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate', 
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

categorical_features_map = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

# Combine all features for input collection (logistic model features + MonthlyIncome for LR if predicting both)
all_input_features = numerical_features_template + list(categorical_features_map.keys())
# Add MonthlyIncome explicitly as it's a feature for Logistic Regression, but target for Linear Regression
# For LR prediction, we'll drop it if it's the target
all_input_features.append('MonthlyIncome') # Add MonthlyIncome for input, it will be removed for LR prediction

# Sidebar for user input
st.sidebar.header('Employee Characteristics Input')

input_data = {}

# Input fields for numerical features
st.sidebar.subheader('Numerical Features')
for feature in numerical_features_template:
    if feature != 'MonthlyIncome': # MonthlyIncome will be a prediction, not an input for LR
        min_val = df[feature].min() if feature in df.columns else 0 # Assuming df is available from kernel state
        max_val = df[feature].max() if feature in df.columns else 100000 # Adjust max based on typical ranges
        default_val = df[feature].mean() if feature in df.columns else 0 # Use mean for default
        input_data[feature] = st.sidebar.number_input(f"**{feature.replace('_', ' ')}**", 
                                                min_value=float(min_val), 
                                                max_value=float(max_val), 
                                                value=float(default_val), 
                                                step=1.0, format="%.2f")

# Input fields for categorical features
st.sidebar.subheader('Categorical Features')
for feature, options in categorical_features_map.items():
    default_index = 0
    if feature in df.columns: # Assuming df is available from kernel state
        # Try to set a more common default, e.g., the mode
        mode_val = df[feature].mode()[0]
        if mode_val in options:
            default_index = options.index(mode_val)

    input_data[feature] = st.sidebar.selectbox(f"**{feature.replace('_', ' ')}**", 
                                             options=options, 
                                             index=default_index)

# Add a placeholder for MonthlyIncome input for Logistic Regression if it's not being predicted by LR
# For the Streamlit app, we will predict MonthlyIncome first, then use that prediction for Attrition.
# So, MonthlyIncome will be an output first, then implicitly an input for attrition.

# Prediction button
if st.sidebar.button('Predict'):
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    st.subheader("Prediction Results")

    # --- Predict Monthly Income (Linear Regression) ---
    st.markdown("### Monthly Income Prediction")
    # Features for LR model should exclude 'MonthlyIncome' itself
    input_for_lr = input_df.drop(columns=['MonthlyIncome'], errors='ignore')
    # Ensure columns match the training data columns for the preprocessor
    lr_numerical_features = [f for f in numerical_features_template if f != 'MonthlyIncome']
    lr_categorical_features = list(categorical_features_map.keys())
    
    # Manually ensure all expected columns are present, fill with default if missing
    expected_lr_cols = lr_numerical_features + lr_categorical_features
    for col in expected_lr_cols:
        if col not in input_for_lr.columns:
            if col in numerical_features_template: # numerical default
                input_for_lr[col] = 0.0 # Or a more sensible default
            else: # categorical default
                input_for_lr[col] = categorical_features_map[col][0] # First category as default

    # Reorder columns to match training order if necessary (pipeline handles this somewhat, but good practice)
    # This assumes preprocessor was fitted on X_linear, which has a specific column order.
    # For robustness, we might need to recreate X_linear from original df to get exact column order
    # For simplicity here, we assume the pipeline will handle column matching internally for transformed features.

    try:
        predicted_income = lr_model.predict(input_for_lr)[0]
        st.success(f"Predicted Monthly Income: **${predicted_income:,.2f}**")
    except Exception as e:
        st.error(f"Error predicting Monthly Income: {e}")
        predicted_income = None # Set to None if prediction fails

    # --- Predict Attrition (Logistic Regression) ---
    st.markdown("### Attrition Prediction")
    
    # Features for Logistic Regression include 'MonthlyIncome'
    # If MonthlyIncome was predicted, use that value in the input_df for attrition prediction.
    if predicted_income is not None:
        input_df['MonthlyIncome'] = predicted_income
    else:
        # If income prediction failed, use a sensible default or original input for MonthlyIncome
        input_df['MonthlyIncome'] = input_data.get('MonthlyIncome', df['MonthlyIncome'].mean())

    # Ensure all columns expected by the logistic model's preprocessor are present
    # This assumes log_model's preprocessor was fitted on X_logistic
    log_numerical_features = numerical_features_template + ['MonthlyIncome'] # MonthlyIncome is a feature for logreg
    log_categorical_features = list(categorical_features_map.keys())

    expected_log_cols = log_numerical_features + log_categorical_features

    for col in expected_log_cols:
        if col not in input_df.columns:
            if col in numerical_features_template or col == 'MonthlyIncome':
                input_df[col] = 0.0 # Or a more sensible default
            else:
                input_df[col] = categorical_features_map[col][0]
    
    # Reorder columns to match training order for robustness if needed

    try:
        attrition_proba = log_model.predict_proba(input_df)[0][1] # Probability of 'Yes' attrition
        attrition_class = log_model.predict(input_df)[0]

        st.write(f"Probability of Attrition (Yes): **{attrition_proba:.2%}**")
        if attrition_class == 1:
            st.error("Predicted Attrition: **Yes** (Employee is likely to leave)")
            st.markdown("Consider targeted retention strategies.")
        else:
            st.success("Predicted Attrition: **No** (Employee is likely to stay)")
            st.markdown("Continue monitoring.")

        # Optional: Threshold guidance
        st.markdown(
            """
            *Note: The 'Yes'/'No' prediction is based on a default probability threshold of 0.5. 
            This threshold can be adjusted based on the business costs of False Positives vs. False Negatives.
            """
        )

    except Exception as e:
        st.error(f"Error predicting Attrition: {e}")


st.markdown("""
---
**Disclaimer:** This dashboard uses machine learning models trained on historical data. 
Predictions are probabilistic and should be used as a tool to inform decisions, not as definitive forecasts.
""")
