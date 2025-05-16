import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

# ====================== STYLING & SETUP ======================
st.set_page_config(
    page_title="Career Path Finder",
    page_icon="🧭",
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
    career_options = [
        'Software Developer', 'Data Scientist', 'AI Engineer', 
        'Cybersecurity Specialist', 'Cloud Architect',
        'Marketing Manager', 'Financial Analyst', 'HR Manager',
        'Entrepreneur', 'Investment Banker',
        'Graphic Designer', 'Video Editor', 'Music Producer',
        'Creative Writer', 'Art Director',
        'Mechanical Engineer', 'Electrical Engineer', 
        'Civil Engineer', 'Robotics Engineer',
        'Doctor', 'Nurse', 'Psychologist', 
        'Physical Therapist', 'Medical Researcher',
        'Biotechnologist', 'Research Scientist', 
        'Environmental Scientist', 'Physicist',
        'Teacher', 'Professor', 'Educational Consultant',
        'Curriculum Developer',
        'Lawyer', 'Judge', 'Legal Consultant',
        'UX Designer', 'Product Manager',
        'Journalist', 'Public Relations Specialist',
        'Architect', 'Urban Planner',
        'Chef', 'Event Planner', 'Fashion Designer'
    ]
    
    try:
        data = pd.read_excel("new_updated_data.xlsx")
        if len(data['Predicted_Career_Field'].unique()) < 20:
            data['Predicted_Career_Field'] = np.random.choice(career_options, size=len(data))
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': np.random.choice(['Technology', 'Business', 'Arts', 'Engineering', 'Medical', 'Science', 'Education', 'Law'], 200),
            'Work_Style': np.random.choice(['Independent', 'Collaborative', 'Flexible'], 200),
            'Strengths': np.random.choice(['Analytical', 'Creative', 'Strategic', 'Practical'], 200),
            'Communication_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Leadership_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Teamwork_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'GPA': np.round(np.random.uniform(2.0, 4.0, 200), 1),
            'Years_of_Experience': np.random.randint(0, 20, 200),
            'Predicted_Career_Field': np.random.choice(career_options, 200)
        })
    
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

data = load_data()

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
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
        st.error("Target column not found in data")
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
def get_all_questions():
    """Returns a pool of 30 questions"""
    return [
        # Interest Questions (1-8)
        {
            "question": "1. Which of these activities excites you most?",
            "options": [
                {"text": "Coding or working with technology", "value": "Technology"},
                {"text": "Analyzing market trends", "value": "Business"},
                {"text": "Creating art or designs", "value": "Arts"},
                {"text": "Building or fixing mechanical things", "value": "Engineering"},
                {"text": "Helping people with health issues", "value": "Medical"},
                {"text": "Conducting experiments", "value": "Science"},
                {"text": "Teaching others", "value": "Education"},
                {"text": "Debating or solving legal problems", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "2. What type of books/movies do you enjoy most?",
            "options": [
                {"text": "Sci-fi and technology", "value": "Technology"},
                {"text": "Business success stories", "value": "Business"},
                {"text": "Creative arts and design", "value": "Arts"},
                {"text": "How things work", "value": "Engineering"},
                {"text": "Medical dramas", "value": "Medical"},
                {"text": "Scientific discoveries", "value": "Science"},
                {"text": "Educational content", "value": "Education"},
                {"text": "Courtroom dramas", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "3. Which subject did you enjoy most in school?",
            "options": [
                {"text": "Computer Science", "value": "Technology"},
                {"text": "Economics/Business", "value": "Business"},
                {"text": "Art/Music", "value": "Arts"},
                {"text": "Physics/Engineering", "value": "Engineering"},
                {"text": "Biology/Medicine", "value": "Medical"},
                {"text": "Chemistry/Physics", "value": "Science"},
                {"text": "Education/Psychology", "value": "Education"},
                {"text": "Government/Law", "value": "Law"}
            ],
            "feature": "Interest"
        },
        {
            "question": "4. What type of projects do you enjoy most?",
            "options": [
                {"text": "Developing software/apps", "value": "Technology"},
                {"text": "Creating business plans", "value": "Business"},
                {"text": "Designing visuals/artwork", "value": "Arts"},
                {"text": "Building physical prototypes", "value": "Engineering"},
                {"text": "Helping people directly", "value": "Medical"},
                {"text": "Research experiments", "value": "Science"},
                {"text": "Teaching/mentoring", "value": "Education"},
                {"text": "Analyzing legal cases", "value": "Law"}
            ],
            "feature": "Interest"
        },

        # Work Style Questions (5-8)
        {
            "question": "5. How do you prefer to work?",
            "options": [
                {"text": "Alone with clear tasks", "value": "Independent"},
                {"text": "In a team environment", "value": "Collaborative"},
                {"text": "A flexible mix of both", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "6. Your ideal project would involve:",
            "options": [
                {"text": "Working independently on your part", "value": "Independent"},
                {"text": "Constant collaboration with others", "value": "Collaborative"},
                {"text": "Some teamwork with independent phases", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "7. When facing a difficult problem, you:",
            "options": [
                {"text": "Prefer to solve it yourself", "value": "Independent"},
                {"text": "Ask colleagues for input", "value": "Collaborative"},
                {"text": "Try yourself first, then ask for help", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },
        {
            "question": "8. Your ideal work schedule would be:",
            "options": [
                {"text": "Strict 9-5 with clear boundaries", "value": "Independent"},
                {"text": "Flexible hours with team coordination", "value": "Collaborative"},
                {"text": "Mix of structured and flexible time", "value": "Flexible"}
            ],
            "feature": "Work_Style"
        },

        # Strengths Questions (9-12)
        {
            "question": "9. What comes most naturally to you?",
            "options": [
                {"text": "Solving complex problems", "value": "Analytical"},
                {"text": "Coming up with creative ideas", "value": "Creative"},
                {"text": "Planning long-term strategies", "value": "Strategic"},
                {"text": "Building practical solutions", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "10. Others would describe you as:",
            "options": [
                {"text": "Logical and detail-oriented", "value": "Analytical"},
                {"text": "Imaginative and original", "value": "Creative"},
                {"text": "Visionary and forward-thinking", "value": "Strategic"},
                {"text": "Hands-on and resourceful", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "11. Your strongest skill is:",
            "options": [
                {"text": "Data analysis", "value": "Analytical"},
                {"text": "Creative thinking", "value": "Creative"},
                {"text": "Long-term planning", "value": "Strategic"},
                {"text": "Practical implementation", "value": "Practical"}
            ],
            "feature": "Strengths"
        },
        {
            "question": "12. In school projects, you typically:",
            "options": [
                {"text": "Handled the data/analysis parts", "value": "Analytical"},
                {"text": "Came up with creative concepts", "value": "Creative"},
                {"text": "Organized the overall strategy", "value": "Strategic"},
                {"text": "Built the practical deliverables", "value": "Practical"}
            ],
            "feature": "Strengths"
        },

        # Communication Skills (13-16)
        {
            "question": "13. In social situations, you:",
            "options": [
                {"text": "Prefer listening to speaking", "value": "Low"},
                {"text": "Speak when you have something to say", "value": "Medium"},
                {"text": "Easily engage in conversations", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "14. When explaining something complex, you:",
            "options": [
                {"text": "Struggle to put it in simple terms", "value": "Low"},
                {"text": "Can explain if you prepare", "value": "Medium"},
                {"text": "Naturally simplify complex ideas", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "15. In group presentations, you typically:",
            "options": [
                {"text": "Handle the background research", "value": "Low"},
                {"text": "Present your specific part", "value": "Medium"},
                {"text": "Take the lead in presenting", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },
        {
            "question": "16. When networking professionally, you:",
            "options": [
                {"text": "Find it challenging", "value": "Low"},
                {"text": "Can do it when necessary", "value": "Medium"},
                {"text": "Enjoy meeting new people", "value": "High"}
            ],
            "feature": "Communication_Skills"
        },

        # Leadership Skills (17-20)
        {
            "question": "17. When a group needs direction, you:",
            "options": [
                {"text": "Wait for someone else to step up", "value": "Low"},
                {"text": "Help if no one else does", "value": "Medium"},
                {"text": "Naturally take the lead", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "18. Your approach to responsibility is:",
            "options": [
                {"text": "Avoid taking charge", "value": "Low"},
                {"text": "Take charge when needed", "value": "Medium"},
                {"text": "Seek leadership roles", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "19. In team projects, you usually:",
            "options": [
                {"text": "Follow others' lead", "value": "Low"},
                {"text": "Share leadership duties", "value": "Medium"},
                {"text": "Organize the team's work", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },
        {
            "question": "20. When making group decisions, you:",
            "options": [
                {"text": "Go with the majority", "value": "Low"},
                {"text": "Voice your opinion when asked", "value": "Medium"},
                {"text": "Facilitate the decision-making", "value": "High"}
            ],
            "feature": "Leadership_Skills"
        },

        # Teamwork Skills (21-24)
        {
            "question": "21. In group settings, you usually:",
            "options": [
                {"text": "Focus on your individual tasks", "value": "Low"},
                {"text": "Coordinate when necessary", "value": "Medium"},
                {"text": "Actively collaborate with others", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "22. When a teammate needs help, you:",
            "options": [
                {"text": "Let them figure it out", "value": "Low"},
                {"text": "Help if they ask", "value": "Medium"},
                {"text": "Proactively offer assistance", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "23. Your view on teamwork is:",
            "options": [
                {"text": "Prefer working alone", "value": "Low"},
                {"text": "Teamwork has its benefits", "value": "Medium"},
                {"text": "Believe in the power of collaboration", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },
        {
            "question": "24. In conflict situations, you:",
            "options": [
                {"text": "Avoid getting involved", "value": "Low"},
                {"text": "Help mediate if needed", "value": "Medium"},
                {"text": "Actively work to resolve conflicts", "value": "High"}
            ],
            "feature": "Teamwork_Skills"
        },

        # Additional Career-Relevant Questions (25-30)
        {
            "question": "25. How do you handle deadlines?",
            "options": [
                {"text": "I often procrastinate", "value": "Low"},
                {"text": "I meet them with some effort", "value": "Medium"},
                {"text": "I consistently meet them early", "value": "High"}
            ],
            "feature": "Time_Management"
        },
        {
            "question": "26. When learning something new, you prefer:",
            "options": [
                {"text": "Hands-on practice", "value": "Practical"},
                {"text": "Theoretical understanding", "value": "Theoretical"},
                {"text": "Visual demonstrations", "value": "Visual"},
                {"text": "Group discussions", "value": "Social"}
            ],
            "feature": "Learning_Style"
        },
        {
            "question": "27. Your ideal work environment is:",
            "options": [
                {"text": "Structured and predictable", "value": "Structured"},
                {"text": "Dynamic and changing", "value": "Dynamic"},
                {"text": "Creative and open", "value": "Creative"},
                {"text": "Fast-paced and challenging", "value": "Challenging"}
            ],
            "feature": "Work_Environment"
        },
        {
            "question": "28. When faced with a problem, you:",
            "options": [
                {"text": "Follow established procedures", "value": "Procedural"},
                {"text": "Brainstorm creative solutions", "value": "Creative"},
                {"text": "Analyze data thoroughly", "value": "Analytical"},
                {"text": "Ask others for advice", "value": "Collaborative"}
            ],
            "feature": "Problem_Solving"
        },
        {
            "question": "29. When making decisions, you rely mostly on:",
            "options": [
                {"text": "Logic and facts", "value": "Logical"},
                {"text": "Gut feelings", "value": "Intuitive"},
                {"text": "Others' opinions", "value": "Social"},
                {"text": "Past experiences", "value": "Experiential"}
            ],
            "feature": "Decision_Making"
        },
        {
            "question": "30. You consider yourself more:",
            "options": [
                {"text": "Realistic and practical", "value": "Practical"},
                {"text": "Imaginative and innovative", "value": "Innovative"},
                {"text": "People-oriented", "value": "Social"},
                {"text": "Detail-oriented", "value": "Detail"}
            ],
            "feature": "Self_Perception"
        }
    ]

import random

@st.cache_data
def get_randomized_questions():
    """Selects 10 random questions from the pool of 20."""
    all_questions = get_all_questions()  # assuming this returns a list of questions
    features = list(set(q['feature'] for q in all_questions))  # Features represent different categories
    selected = []

    # First pick one from each feature category
    for feature in features:
        feature_questions = [q for q in all_questions if q['feature'] == feature]
        if feature_questions:
            selected.append(random.choice(feature_questions))

    # Remove selected questions from the pool
    remaining = [q for q in all_questions if q not in selected]

    # Calculate how many more we need to reach 10
    needed = 10 - len(selected)

    # Only sample if we have remaining questions and need more
    if needed > 0 and remaining:
        selected.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(selected)
    return selected


direct_input_features = {
    "GPA": {
        "question": "What is your approximate GPA (0.0-4.0)?",
        "type": "number", 
        "min": 0.0, 
        "max": 4.0, 
        "step": 0.1, 
        "default": 3.0
    },
    "Years_of_Experience": {
        "question": "Years of professional experience (if any):",
        "type": "number", 
        "min": 0, 
        "max": 50, 
        "step": 1, 
        "default": 0
    }
}

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    # Initialize session state
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'questions' not in st.session_state:
        st.session_state.questions = get_randomized_questions()
    
    st.title("🧭 Career Path Finder")
    st.markdown("Discover careers that match your unique strengths and preferences.")
    
    st.sidebar.title("About This Tool")
    st.sidebar.info("This assessment helps match your profile with suitable career options.")
    st.sidebar.write(f"*Based on analysis of {len(data)} career paths*")
    
    tab1, tab2 = st.tabs(["Take Assessment", "Career Insights"])
    
    with tab1:
        st.header("Career Compatibility Assessment")
        
        # Show restart button if we have previous responses
        if st.session_state.user_responses:
            if st.button("🔄 Start New Assessment"):
                st.session_state.user_responses = {}
                st.session_state.questions = get_randomized_questions()
                st.rerun()
        
        st.write("Answer these questions to discover careers that fit your profile.")
        
        with st.expander("Your Background"):
            for feature, config in direct_input_features.items():
                st.session_state.user_responses[feature] = st.number_input(
                    config["question"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    key=f"num_{feature}"
                )
        
        st.subheader("Personality and Preferences")
        for i, q in enumerate(st.session_state.questions):
            selected_option = st.radio(
                q["question"],
                [opt["text"] for opt in q["options"]],
                key=f"q_{i}"
            )
            selected_value = q["options"][[opt["text"] for opt in q["options"]].index(selected_option)]["value"]
            st.session_state.user_responses[q["feature"]] = selected_value
        
        if st.button("🔮 Find My Career Match"):
            required_fields = list(direct_input_features.keys()) + ['Interest', 'Work_Style', 'Strengths']
            filled_fields = [field for field in required_fields if field in st.session_state.user_responses]
            
            if len(filled_fields) < 3:
                st.warning("Please answer at least 3 questions (including GPA and Experience) for better results.")
            else:
                with st.spinner("Analyzing your unique profile..."):
                    input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                    
                    le_dict = {}
                    for col in data.select_dtypes(include=['object']).columns:
                        if col in data.columns and col != 'Predicted_Career_Field':
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            le_dict[col] = le
                    
                    for col in input_data.columns:
                        if col in st.session_state.user_responses:
                            if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills']:
                                level_map = {"Low": 0, "Medium": 1, "High": 2}
                                input_data[col] = level_map.get(st.session_state.user_responses[col], 1)
                            elif col in le_dict:
                                try:
                                    input_data[col] = le_dict[col].transform([st.session_state.user_responses[col]])[0]
                                except ValueError:
                                    input_data[col] = processed_data[col].mode()[0]
                            else:
                                input_data[col] = st.session_state.user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    try:
                        prediction = model.predict(input_data)
                        predicted_career = target_le.inverse_transform(prediction)[0]
                        
                        st.success(f"### Your Best Career Match: **{predicted_career}**")
                        
                        with st.expander("💡 Why this career matches you"):
                            st.write("This career aligns well with your profile because of:")
                            
                            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                            top_features = feat_importances.sort_values(ascending=False).head(3)
                            
                            for feat in top_features.index:
                                importance_desc = ""
                                if feat == "Interest":
                                    interest_val = st.session_state.user_responses.get("Interest", "Various")
                                    importance_desc = f"Your strong interest in {interest_val} fields"
                                elif feat == "Work_Style":
                                    style_val = st.session_state.user_responses.get("Work_Style", "Various")
                                    importance_desc = f"Your preference for {style_val.lower()} work environments"
                                elif feat == "Strengths":
                                    strength_val = st.session_state.user_responses.get("Strengths", "Various")
                                    importance_desc = f"Your {strength_val.lower()} strengths"
                                elif feat == "GPA":
                                    gpa_val = st.session_state.user_responses.get("GPA", 3.0)
                                    importance_desc = f"Your academic performance (GPA: {gpa_val})"
                                elif feat == "Years_of_Experience":
                                    exp_val = st.session_state.user_responses.get("Years_of_Experience", 0)
                                    importance_desc = f"Your professional experience ({exp_val} years)"
                                else:
                                    importance_desc = f"Your responses about {feat.replace('_', ' ').lower()}"
                                
                                st.write(f"- **{importance_desc}** (weight: {top_features[feat]:.2f})")
                            
                            st.write("\nThis career path typically requires these characteristics, which match well with your profile.")
                        
                        with st.expander("📚 Learn more about this career"):
                            career_data = data[data['Predicted_Career_Field'] == predicted_career]
                            if not career_data.empty:
                                st.write(f"**Typical profile for {predicted_career}:**")
                                cols = st.columns(3)
                                with cols[0]:
                                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                                    else:
                                        st.metric("Average GPA", "N/A")
                                with cols[1]:
                                    if 'Years_of_Experience' in career_data.columns:
                                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                                    else:
                                        st.metric("Avg. Experience", "N/A")
                                with cols[2]:
                                    if 'Interest' in career_data.columns:
                                        st.metric("Common Interest", career_data['Interest'].mode()[0])
                                    else:
                                        st.metric("Common Interest", "N/A")
                                
                                st.write("\n**Common characteristics:**")
                                if 'Work_Style' in career_data.columns:
                                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                                if 'Strengths' in career_data.columns:
                                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                                if 'Communication_Skills' in career_data.columns:
                                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
                            else:
                                st.write("No additional information available for this career in our dataset.")
                        
                    except Exception as e:
                        st.error(f"We encountered an issue analyzing your profile. Please try again.")
    
    with tab2:
        st.header("📊 Career Insights")
        st.write("Explore different career paths and their characteristics.")
        
        if st.checkbox("Show Career Options"):
            if 'Predicted_Career_Field' in data.columns:
                st.dataframe(data['Predicted_Career_Field'].value_counts().reset_index().rename(
                    columns={'index': 'Career', 'Predicted_Career_Field': 'Frequency'}))
            else:
                st.warning("Career field data not available.")
        
        st.subheader("Popular Career Paths")
        if 'Predicted_Career_Field' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            data['Predicted_Career_Field'].value_counts().head(15).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Most Common Career Paths")
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No career path data available.")
        
        st.subheader("Career Characteristics")
        if 'Predicted_Career_Field' in data.columns:
            selected_career = st.selectbox(
                "Select a career to learn more:",
                sorted(data['Predicted_Career_Field'].unique()),
                key="career_select"
            )
            
            career_data = data[data['Predicted_Career_Field'] == selected_career]
            
            if not career_data.empty:
                st.write(f"**Typical profile for {selected_career}:**")
                cols = st.columns(3)
                with cols[0]:
                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                    else:
                        st.metric("Average GPA", "N/A")
                with cols[1]:
                    if 'Years_of_Experience' in career_data.columns:
                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                    else:
                        st.metric("Avg. Experience", "N/A")
                with cols[2]:
                    if 'Interest' in career_data.columns:
                        st.metric("Common Interest", career_data['Interest'].mode()[0])
                    else:
                        st.metric("Common Interest", "N/A")

                st.write("\n**Common characteristics:**")
                if 'Work_Style' in career_data.columns:
                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                if 'Strengths' in career_data.columns:
                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                if 'Communication_Skills' in career_data.columns:
                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
            else:
                st.warning("No data available for this career.")
        else:
            st.warning("No career data available for analysis.")

if __name__ == "__main__":
    main()
