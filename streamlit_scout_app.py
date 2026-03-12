import streamlit as st
import random
import time

# 1. Page Configuration and Theme Setup
st.set_page_config(
    page_title="Gridiron Intelligence - Football LLM",
    page_icon="🏈",
    layout="wide"
)

# Custom CSS for a clean, football-inspired interface
st.markdown("""
<style>
    /* Main app background */
    .reportview-container {
        background-color: #f0f2f6; 
    }
    /* Customize buttons */
    .stButton>button {
        background-color: #1a531f; /* Dark Green */
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2e7d32; /* Lighter Green on hover */
        color: white;
    }
    /* Customize text input area */
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 5px;
    }
    /* Title styling */
    .football-title {
        color: #1a531f;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        padding-bottom: 0px;
    }
    /* Subtitle styling */
    .football-subtitle {
        color: #495057;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 400;
        padding-top: 0px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Page Content - Header Section

# Using markdown for custom styled title and subtitle
st.markdown("<h1 class='football-title'>Gridiron Intelligence 🏈</h1>", unsafe_allow_html=True)
st.markdown("<p class='football-subtitle'>The Advanced Football LLM Analysist, powered by Gemini Backend</p>", unsafe_allow_html=True)

# Sidebar with logo and navigation placeholder
with st.sidebar:
    # Use a placeholder image URL for the logo
    st.image("https://raw.githubusercontent.com/shaverm96/Gridiron-Intelligence/main/Logos/Main.svg", width=150)
    st.title("Gridiron Intel")
    st.write("---")
    st.subheader("Navigation")
    st.radio("Go to", ["Analyze Query", "Feature Overview", "About Us"])
    st.write("---")
    st.write("**Model Version:** GI-Gemini-v1.0")

# 3. Main Query Interface

st.markdown("### Ask Your Football Question:")
st.write("Enter your query about stats, strategy, historical data, or draft prospects.")

# User Text Input Area
user_query = st.text_area(
    label="Your Football Query",
    placeholder="Example: 'Analyze the impact of pre-snap motion on Kansas City's offensive success in 2023.'",
    height=100,
    label_visibility="collapsed" # Hide default label for cleaner look
)

# Run Button
if st.button("Run Intelligence Analysis"):
    if user_query:
        # --- Mockup Backend Logic (Start) ---
        # This is where you would typically call the Gemini API
        
        # 1. Simulate API Call Processing
        with st.spinner('Accessing Gemini backend and analyzing game data...'):
            time.sleep(2) # Simulate processing time

        # 2. Simulate Gemini Response (Mockup Data)
        mock_responses = [
            f"**Analysis for:** '{user_query}'\n\nBased on Gridiron Intelligence's analysis powered by Gemini, the pre-snap motion utilized by the Chiefs was a critical component. Data shows a 12% increase in success rate on plays with motion versus those without, primarily by stressing defensive communication...",
            f"**Stat Breakdown for:** '{user_query}'\n\nQuerying player stats... Our model indicates that Travis Kelce saw a 30% target share in third-down situations, the highest among tight ends in 2023. His EPA (Expected Points Added) contribution remains elite...",
            f"**Strategic Insight for:** '{user_query}'\n\nAnalyzing defensive schemes... When facing a standard 4-3 defense, your query suggests implementing more 12-personnel packages to exploit potential mismatches in the running game, particularly off-the-tackle runs...",
            f"**Historical Comparison for:** '{user_query}'\n\nThe 1985 Chicago Bears defense, when compared to modern units using our adjusted performance metrics, ranks 1st in historical dominance. Their '46 defense' would likely generate a 45% pressure rate even against today's sophisticated offensive lines..."
        ]
        
        # Simple simulation logic (not actual LLM processing)
        if "motion" in user_query.lower() or "kc" in user_query.lower():
            response = mock_responses[0]
        elif "stat" in user_query.lower() or "kelce" in user_query.lower():
            response = mock_responses[1]
        elif "strategy" in user_query.lower() or "defense" in user_query.lower():
            response = mock_responses[2]
        elif "history" in user_query.lower() or "bears" in user_query.lower():
            response = mock_responses[3]
        else:
            response = random.choice(mock_responses) # Random fallback

        # --- Mockup Backend Logic (End) ---

        # 3. Display Output
        st.markdown("---")
        st.markdown("#### 🤖 Gridiron Intelligence Response:")
        st.info(response)

    else:
        st.warning("Please enter a football-related query before running the analysis.")

# 4. Feature Showcase Section

st.markdown("---")
st.markdown("## Core Capabilities of Gridiron Intelligence")

# Use columns for features layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📊 Deep Stat Analysis")
    st.write(
        "Leverage Gemini's processing power to go beyond box scores. Analyze EPA, "
        "CPOE, success rates, and situational performance metrics instantly."
    )

with col2:
    st.markdown("#### 🔮 Predictive Modeling")
    st.write(
        "Utilize advanced machine learning to forecast draft outcomes, game "
        "results, and player performance trajectories based on historical data."
    )

with col3:
    st.markdown("#### 🧠 Strategic Breakdown")
    st.write(
        "Get insights into coaching strategies, play-calling tendencies, and "
        "schematic advantages/disadvantages for any team or game."
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>Powered by Google Gemini | © 2024 Gridiron Intelligence | This is a demonstration mockup.</div>", 
    unsafe_allow_html=True
)