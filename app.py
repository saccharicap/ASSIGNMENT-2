%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Streamlit App Title
st.title('Monthly Income Prediction App')
st.write('Select a model and enter employee details to predict their monthly income.')

# Load all saved models
models = {}
try:
    models['Linear Regression'] = joblib.load('linear_regression_model.sav')
    models['Ridge Regression'] = joblib.load('ridge_regression_model.sav')
    models['Lasso Regression'] = joblib.load('lasso_regression_model.sav')
    st.success("All models loaded successfully!")
except FileNotFoundError:
    st.error("Error: One or more model files not found. Please ensure 'linear_regression_model.sav', 'ridge_regression_model.sav', and 'lasso_regression_model.sav' are in the correct directory.")
    st.stop()

# Define the exact order of original feature columns used during training
# This is crucial for consistency with the fitted ColumnTransformer
original_feature_columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Sidebar for model selection and feature input
with st.sidebar:
    st.header('Configuration')
    selected_model_name = st.selectbox('Choose a Model', list(models.keys()))
    model_pipeline = models[selected_model_name]

    st.header('Employee Features Input')

    age = st.slider('Age', 18, 60, 30)
    business_travel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    daily_rate = st.slider('Daily Rate', 100, 1500, 800)
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.slider('Distance From Home', 1, 29, 10)
    education = st.selectbox('Education', [1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}") # 1 'Below College' to 5 'Master'
    education_field = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
    environment_satisfaction = st.selectbox('Environment Satisfaction', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    gender = st.selectbox('Gender', ['Female', 'Male'])
    hourly_rate = st.slider('Hourly Rate', 30, 100, 60)
    job_involvement = st.selectbox('Job Involvement', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    job_level = st.selectbox('Job Level', [1, 2, 3, 4, 5], format_func=lambda x: f"Level {x}")
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    monthly_rate = st.slider('Monthly Rate', 2000, 27000, 12000)
    num_companies_worked = st.slider('Number of Companies Worked', 0, 9, 2)
    over_time = st.selectbox('Over Time', ['No', 'Yes'])
    percent_salary_hike = st.slider('Percent Salary Hike', 11, 25, 15)
    performance_rating = st.selectbox('Performance Rating', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    relationship_satisfaction = st.selectbox('Relationship Satisfaction', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    stock_option_level = st.selectbox('Stock Option Level', [0, 1, 2, 3], format_func=lambda x: f"Level {x}")
    total_working_years = st.slider('Total Working Years', 0, 40, 10)
    training_times_last_year = st.slider('Training Times Last Year', 0, 6, 3)
    work_life_balance = st.selectbox('Work Life Balance', [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    years_at_company = st.slider('Years At Company', 0, 40, 5)
    years_in_current_role = st.slider('Years In Current Role', 0, 18, 3)
    years_since_last_promotion = st.slider('Years Since Last Promotion', 0, 15, 1)
    years_with_curr_manager = st.slider('Years With Current Manager', 0, 17, 3)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'BusinessTravel': [business_travel],
    'DailyRate': [daily_rate],
    'Department': [department],
    'DistanceFromHome': [distance_from_home],
    'Education': [education],
    'EducationField': [education_field],
    'EnvironmentSatisfaction': [environment_satisfaction],
    'Gender': [gender],
    'HourlyRate': [hourly_rate],
    'JobInvolvement': [job_involvement],
    'JobLevel': [job_level],
    'JobRole': [job_role],
    'JobSatisfaction': [job_satisfaction],
    'MaritalStatus': [marital_status],
    'MonthlyRate': [monthly_rate],
    'NumCompaniesWorked': [num_companies_worked],
    'OverTime': [over_time],
    'PercentSalaryHike': [percent_salary_hike],
    'PerformanceRating': [performance_rating],
    'RelationshipSatisfaction': [relationship_satisfaction],
    'StockOptionLevel': [stock_option_level],
    'TotalWorkingYears': [total_working_years],
    'TrainingTimesLastYear': [training_times_last_year],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'YearsInCurrentRole': [years_in_current_role],
    'YearsSinceLastPromotion': [years_since_last_promotion],
    'YearsWithCurrManager': [years_with_curr_manager]
})

# IMPORTANT: Reindex the input_data to match the original feature columns order
# This ensures consistency with the fitted ColumnTransformer
input_data = input_data[original_feature_columns]

# Display input data
st.subheader('Input Features:')
st.write(input_data)

# Make prediction button
if st.button('Predict Monthly Income'):
    try:
        prediction = model_pipeline.predict(input_data)[0]
        st.subheader('Predicted Monthly Income (using ' + selected_model_name + '):')
        st.success(f'The predicted monthly income is: ${prediction:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
