import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Files
# ==========================
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ==========================
# Page
# ==========================
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰"
)

st.title("💰 Employee Salary Prediction")
st.write("Fill in the employee information to predict the salary category.")

# ==========================
# Education Mapping
# ==========================
education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

# ==========================
# Remove ? from options
# ==========================
workclass_options = [
    x for x in label_encoders["workclass"].classes_
    if x != "?"
]

occupation_options = [
    x for x in label_encoders["occupation"].classes_
    if x != "?"
]

country_options = [
    x for x in label_encoders["native-country"].classes_
    if x != "?"
]

# ==========================
# Inputs
# ==========================
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

workclass = st.selectbox(
    "Work Class",
    workclass_options
)

education = st.selectbox(
    "Education",
    label_encoders["education"].classes_
)

educational_num = education_map[education]

st.number_input(
    "Years of Education",
    value=educational_num,
    disabled=True
)

marital_status = st.selectbox(
    "Marital Status",
    label_encoders["marital-status"].classes_
)

occupation = st.selectbox(
    "Occupation",
    occupation_options
)

relationship = st.selectbox(
    "Relationship",
    label_encoders["relationship"].classes_
)

race = st.selectbox(
    "Race",
    label_encoders["race"].classes_
)

gender = st.selectbox(
    "Gender",
    label_encoders["gender"].classes_
)

hours_per_week = st.number_input(
    "Hours per Week",
    min_value=1,
    max_value=100,
    value=40
)

native_country = st.selectbox(
    "Native Country",
    country_options
)

# ==========================
# Prediction
# ==========================
if st.button("🔍 Predict Salary", use_container_width=True):

    workclass = label_encoders["workclass"].transform([workclass])[0]
    education = label_encoders["education"].transform([education])[0]
    marital_status = label_encoders["marital-status"].transform([marital_status])[0]
    occupation = label_encoders["occupation"].transform([occupation])[0]
    relationship = label_encoders["relationship"].transform([relationship])[0]
    race = label_encoders["race"].transform([race])[0]
    gender = label_encoders["gender"].transform([gender])[0]
    native_country = label_encoders["native-country"].transform([native_country])[0]

    data = pd.DataFrame([[
        age,
        workclass,
        education,
        educational_num,
        marital_status,
        occupation,
        relationship,
        race,
        gender,
        hours_per_week,
        native_country
    ]], columns=[
        "age",
        "workclass",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "hours-per-week",
        "native-country"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("💰 Estimated Income: Greater than $50,000 per year")
    else:
        st.info("💵 Estimated Income: $50,000 or Less per year")