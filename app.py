import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ====================== STYLING & SETUP ======================
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #4a6cf7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Radio buttons */
    .stRadio>div {
        flex-direction: column;
        gap: 8px;
    }
    .stRadio>div>label {
        background: #f1f3ff;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stRadio>div>label:hover {
        background: #e0e5ff;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #4a6cf7;
        padding-bottom: 10px;
    }
    h2 {
        color: #3498db;
    }
    
    /* Expanders */
    .stExpander {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA LOADING & PREPROCESSING ======================
@st.cache_data
def load_data():
    # If CSV not found, use demo data
    try:
        data = pd.read_excel("Intial Dataset.xlsx")
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': ['Technology', 'Business', 'Arts', 'Engineering', 'Medical', 'Science', 'Education', 'Law'],
            'Work_Style': ['Independent', 'Collaborative', 'Flexible', 'Independent', 'Collaborative', 'Flexible', 'Collaborative', 'Independent'],
            'Strengths': ['Analytical', 'Creative', 'Strategic', 'Practical', 'Analytical', 'Analytical', 'Creative', 'Strategic'],
            'Communication_Skills': ['Medium', 'High', 'Low', 'Medium', 'High', 'High', 'High', 'High'],
            'Leadership_Skills': ['Medium', 'High', 'Low', 'Medium', 'High', 'Medium', 'High', 'High'],
            'Teamwork_Skills': ['High', 'Medium', 'Low', 'High', 'Medium', 'High', 'High', 'Medium'],
            'GPA': [3.5, 3.8, 3.2, 3.9, 3.6, 3.7, 3.4, 3.9],
            'Years_of_Experience': [5, 10, 2, 8, 12, 6, 9, 15],
            'Predicted_Career_Field': [
                'Software Developer', 
                'Marketing Manager', 
                'Graphic Designer',
                'Data Scientist',
                'Doctor',
                'Research Scientist',
                'Teacher/Professor',
                'Lawyer'
            ]
        })
    
    # Clean data
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

data = load_data()

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
    # Only encode columns that exist in the dataframe and are object type
    object_cols = [col for col in data.select_dtypes(include=['object']).columns 
                  if col in data.columns]
    
    for col in object_cols:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    
    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le

processed_data, target_le = preprocess_data(data.copy())

def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column 'Predicted_Career_Field' not found in data")
        return None, 0
    
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model(processed_data)

# ====================== QUESTIONNAIRE ======================
questions = [
    {
        "section": "Your Interests",
        "questions": [
            {
                "question": "Which of these fields interests you most?",
                "options": [
                    {"text": "Technology (Software, AI, IT)", "value": "Technology"},
                    {"text": "Business & Finance (Marketing, Banking)", "value": "Business"},
                    {"text": "Creative Arts (Design, Music, Writing)", "value": "Arts"},
                    {"text": "Engineering (Mechanical, Electrical)", "value": "Engineering"},
                    {"text": "Healthcare (Medicine, Nursing)", "value": "Medical"},
                    {"text": "Science (Physics, Biology, Research)", "value": "Science"},
                    {"text": "Education (Teaching, Training)", "value": "Education"},
                    {"text": "Law & Justice (Lawyer, Judge)", "value": "Law"}
                ],
                "feature": "Interest"
            },
            {
                "question": "What type of projects excite you?",
                "options": [
                    {"text": "Developing new software or apps", "value": "Technology"},
                    {"text": "Launching a new business/product", "value": "Business"},
                    {"text": "Creating artistic works", "value": "Arts"},
                    {"text": "Building physical structures/machines", "value": "Engineering"},
                    {"text": "Helping people with health issues", "value": "Medical"},
                    {"text": "Conducting scientific experiments", "value": "Science"},
                    {"text": "Teaching or mentoring others", "value": "Education"},
                    {"text": "Arguing cases or solving legal problems", "value": "Law"}
                ],
                "feature": "Interest"
            }
        ]
    },
    {
        "section": "Work Preferences",
        "questions": [
            {
                "question": "Your ideal work environment is:",
                "options": [
                    {"text": "Working alone with clear tasks", "value": "Independent"},
                    {"text": "Working closely with a team", "value": "Collaborative"},
                    {"text": "A mix of both with flexibility", "value": "Flexible"}
                ],
                "feature": "Work_Style"
            },
            {
                "question": "When facing a complex problem, you:",
                "options": [
                    {"text": "Prefer to solve it yourself", "value": "Independent"},
                    {"text": "Brainstorm with colleagues", "value": "Collaborative"},
                    {"text": "Depends on the situation", "value": "Flexible"}
                ],
                "feature": "Work_Style"
            }
        ]
    },
    {
        "section": "Your Skills",
        "questions": [
            {
                "question": "Your strongest skill is:",
                "options": [
                    {"text": "Analyzing data and patterns", "value": "Analytical"},
                    {"text": "Coming up with new ideas", "value": "Creative"},
                    {"text": "Planning long-term strategies", "value": "Strategic"},
                    {"text": "Hands-on problem solving", "value": "Practical"}
                ],
                "feature": "Strengths"
            },
            {
                "question": "You're particularly good at:",
                "options": [
                    {"text": "Math and logical reasoning", "value": "Analytical"},
                    {"text": "Artistic expression", "value": "Creative"},
                    {"text": "Seeing the big picture", "value": "Strategic"},
                    {"text": "Building physical solutions", "value": "Practical"}
                ],
                "feature": "Strengths"
            }
        ]
    },
    {
        "section": "Communication Style",
        "questions": [
            {
                "question": "How comfortable are you presenting ideas?",
                "options": [
                    {"text": "Very uncomfortable", "value": "Low"},
                    {"text": "Somewhat comfortable", "value": "Medium"},
                    {"text": "Very comfortable", "value": "High"}
                ],
                "feature": "Communication_Skills"
            },
            {
                "question": "In meetings, you typically:",
                "options": [
                    {"text": "Rarely speak up", "value": "Low"},
                    {"text": "Contribute when asked", "value": "Medium"},
                    {"text": "Frequently share ideas", "value": "High"}
                ],
                "feature": "Communication_Skills"
            }
        ]
    },
    {
        "section": "Leadership Approach",
        "questions": [
            {
                "question": "When assigned to lead a project, you:",
                "options": [
                    {"text": "Feel anxious about the responsibility", "value": "Low"},
                    {"text": "Manage but prefer not to lead", "value": "Medium"},
                    {"text": "Feel confident in your ability", "value": "High"}
                ],
                "feature": "Leadership_Skills"
            },
            {
                "question": "Your leadership style is:",
                "options": [
                    {"text": "Avoid leadership roles", "value": "Low"},
                    {"text": "Lead when necessary", "value": "Medium"},
                    {"text": "Naturally take charge", "value": "High"}
                ],
                "feature": "Leadership_Skills"
            }
        ]
    },
    {
        "section": "Team Dynamics",
        "questions": [
            {
                "question": "In group projects, you typically:",
                "options": [
                    {"text": "Work separately on your part", "value": "Low"},
                    {"text": "Coordinate some with teammates", "value": "Medium"},
                    {"text": "Actively collaborate throughout", "value": "High"}
                ],
                "feature": "Teamwork_Skills"
            },
            {
                "question": "When a teammate struggles, you:",
                "options": [
                    {"text": "Focus on your own work", "value": "Low"},
                    {"text": "Help if they ask directly", "value": "Medium"},
                    {"text": "Proactively offer assistance", "value": "High"}
                ],
                "feature": "Teamwork_Skills"
            }
        ]
    }
]

direct_input_features = {
    "GPA": {"question": "What is your GPA (0.0-4.0)?", "type": "number", "min": 0.0, "max": 4.0, "step": 0.1, "default": 3.0},
    "Years_of_Experience": {"question": "Years of professional experience:", "type": "number", "min": 0, "max": 50, "step": 1, "default": 5}
}

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    st.title("🧠 AI Career Prediction System")
    st.markdown("Discover your ideal career path based on your skills and preferences.")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This tool uses machine learning to match your profile with suitable careers.")
    st.sidebar.write(f"Model Accuracy: **{accuracy:.1%}**")
    
    # Tabs
    tab1, tab2 = st.tabs(["Career Prediction", "Data Insights"])
    
    with tab1:
        st.header("📝 Career Assessment")
        user_responses = {}
        
        # Direct inputs (GPA, Experience)
        with st.expander("Academic & Professional Background"):
            for feature, config in direct_input_features.items():
                user_responses[feature] = st.number_input(
                    config["question"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"]
                )
        
        # Questionnaire
        for section in questions:
            with st.expander(f"🔹 {section['section']}"):
                for question in section["questions"]:
                    selected_option = st.radio(
                        question["question"],
                        [opt["text"] for opt in question["options"]],
                        key=f"{question['feature']}_{question['question'][:20]}"
                    )
                    selected_value = question["options"][[opt["text"] for opt in question["options"]].index(selected_option)]["value"]
                    if question["feature"] not in user_responses:
                        user_responses[question["feature"]] = []
                    user_responses[question["feature"]].append(selected_value)
        
        # Prediction
        if st.button("🚀 Predict My Career"):
            if len(user_responses) < 3:
                st.warning("Please answer more questions for better accuracy.")
            else:
                with st.spinner("Analyzing your profile..."):
                    # Prepare input data
                    input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                    
                    # Create label encoders for categorical features
                    le_dict = {}
                    for col in data.select_dtypes(include=['object']).columns:
                        if col in data.columns and col != 'Predicted_Career_Field':
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            le_dict[col] = le
                    
                    for col in input_data.columns:
                        if col in user_responses:
                            if isinstance(user_responses[col], list):  # For question-based features
                                if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills']:
                                    # Handle Low/Medium/High scale
                                    level_map = {"Low": 0, "Medium": 1, "High": 2}
                                    avg_level = np.mean([level_map[val] for val in user_responses[col]])
                                    input_data[col] = avg_level
                                else:
                                    # For other categorical features, take the first response and encode it
                                    if col in le_dict:
                                        input_data[col] = le_dict[col].transform([user_responses[col][0]])[0]
                            else:  # For direct inputs (numerical)
                                input_data[col] = user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    # Make prediction
                    try:
                        prediction = model.predict(input_data)
                        predicted_career = target_le.inverse_transform(prediction)[0]
                        
                        # Explain prediction
                        st.success(f"🎯 **Recommended Career:** {predicted_career}")
                        
                        with st.expander("🔍 Why this recommendation?"):
                            st.write("The AI considered these key factors from your responses:")
                            
                            # Get feature importances
                            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                            top_features = feat_importances.sort_values(ascending=False).head(3)
                            
                            for feat in top_features.index:
                                # Map feature names to user-friendly descriptions
                                feature_descriptions = {
                                    "Interest": "Your interests and passions",
                                    "Work_Style": "Your preferred work environment",
                                    "Strengths": "Your strongest skills",
                                    "Communication_Skills": "Your communication style",
                                    "Leadership_Skills": "Your leadership approach",
                                    "Teamwork_Skills": "Your teamwork preferences",
                                    "GPA": "Your academic performance",
                                    "Years_of_Experience": "Your professional experience"
                                }
                                friendly_name = feature_descriptions.get(feat, feat.replace('_', ' '))
                                st.write(f"- **{friendly_name}** (importance: {top_features[feat]:.2f})")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
    
    with tab2:
        st.header("📊 Dataset Insights")
        st.write("Explore the data used for predictions.")
        
        if st.checkbox("Show raw data"):
            st.dataframe(data)
        
        st.subheader("Career Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        data['Predicted_Career_Field'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Most Common Careers in Dataset")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
