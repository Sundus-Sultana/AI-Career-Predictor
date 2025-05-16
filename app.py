
import streamlit as st
import joblib
import numpy as np

# Load trained model and label encoder
model = joblib.load("model/career_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.set_page_config(page_title="Career Predictor", page_icon="🎯")
st.title("🎓 AI Career Predictor")
st.markdown("Answer the following questions to get a career recommendation based on your profile.")

# Collect user input
gpa = st.selectbox("GPA (4.0 scale)", [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.0])
certs = st.selectbox("Number of certifications completed", [0, 1, 2, 3, 4, 5, 6])
internship = st.selectbox("Have you completed an internship?", ["Yes", "No"])
experience = st.selectbox("Years of experience (including internships)", [0, 1, 2, 3, 4, 5])
salary = st.selectbox("Expected starting salary (₹ LPA)", ["Low", "Moderate", "High"])
linkedin = st.selectbox("How active is your LinkedIn profile?", ["Low", "Moderate", "High"])
github = st.selectbox("How active is your GitHub profile?", ["Low", "Moderate", "High"])
courses = st.selectbox("How many extra courses have you completed?", [0, 1, 2, 3, 4, 5, 6])

st.subheader("🧠 Psychometric Traits (1 = Low, 5 = High)")
openness = st.slider("Openness", 1, 5, 3)
conscientiousness = st.slider("Conscientiousness", 1, 5, 3)
extraversion = st.slider("Extraversion", 1, 5, 3)
agreeableness = st.slider("Agreeableness", 1, 5, 3)
neuroticism = st.slider("Neuroticism", 1, 5, 3)

# Helper encoders
def encode_level(val):
    return {"Low": 0, "Moderate": 1, "High": 2}.get(val, 0)

def encode_yes_no(val):
    return {"Yes": 1, "No": 0}.get(val, 0)

# Combine all inputs into a model-ready vector
input_vector = np.array([[
    gpa,
    certs,
    encode_yes_no(internship),
    experience,
    encode_level(salary),
    encode_level(linkedin),
    encode_level(github),
    courses,
    openness,
    conscientiousness,
    extraversion,
    agreeableness,
    neuroticism
]])

if st.button("🔍 Predict Career"):
    prediction = model.predict(input_vector)
    career = label_encoder.inverse_transform(prediction)[0]
    st.success(f"✅ Recommended Career Field: {career}")
