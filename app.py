import streamlit as st
import random
import time
import os
import google.generativeai as genai

# --- CONFIGURATION (Load API Key) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    genai.configure(api_key=GEMINI_API_KEY)

# Import Logic from separate module
from scout_card_engine import (
    Measurables, 
    HighSchoolStats, 
    PlayerContext, 
    run_quant_engine, 
    retrieve_relevant_insights, 
    generate_scout_report_llm, 
    ANALYTICS_FACT_BANK
)

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
        color: #000000; /* Force black text */
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
    st.title("Gridiron Intelligence")
    st.write("---")
    st.subheader("Navigation")
    st.radio("Go to", ["Analyze Query", "Feature Overview", "About Us"])
    st.write("---")
    st.write("**Model Version:** GI-Gemini-v1.0")

# Main Query Interface
st.markdown("### Player Search / Stats Query:")
st.write("Enter a player's name to generate a full scouting report, or ask a general football question.")

# User Text Input Area
user_query = st.text_area(
    label="Search",
    placeholder="e.g., 'Arch Manning', 'Shedeur Sanders', or 'Analyze the Chiefs redzone efficiency...'",
    height=100,
    label_visibility="collapsed"
)

# Run Button
if st.button("Run Intelligence Analysis"):
    if user_query:
        # --- Mockup Backend Logic (Start) ---
        with st.spinner('Accessing Gemini backend and analyzing game data...'):
            time.sleep(1.5) # Simulate processing time
            
            # Use the query as the player name
            player_name = user_query.strip()
            if len(player_name) < 3: player_name = "Marcus Johnson"

            # Create Mock Player Context
            # In a real app, this would query a database
            measurables = Measurables(6, 4, 215, "TX")
            stats = HighSchoolStats(3450, 42, 8, 68.5, 4)
            
            player_ctx = PlayerContext(
                player_name=player_name,
                position="QB",
                high_school="Westlake High",
                measurables=measurables,
                stats=stats,
                target_school="Texas A&M",
                target_school_tier="Power 4"
            )
            
            # Run Engine
            player_ctx.quant_output = run_quant_engine(player_ctx)
            player_ctx.rag_insights = retrieve_relevant_insights(player_ctx, ANALYTICS_FACT_BANK)
            
            try:
                report = generate_scout_report_llm(player_ctx)
            except Exception as e:
                report = f"⚠️ AI Report Generation Error: {str(e)}\n\nUsing fallback template..."
                st.error(f"LLM Error: {e}")

        # 3. Display Output - Scorecard Style
        st.markdown("---")
        st.subheader(f"📝 Scouting Report: {player_ctx.player_name}")
        st.caption(f"{player_ctx.position} | {player_ctx.high_school} | {player_ctx.measurables.state}")
        
        # Metrics Row
        col_score, col_tier, col_conf = st.columns(3)
        
        with col_score:
            st.metric("Quant Score", f"{player_ctx.quant_output.raw_score}/100")
        with col_tier:
            st.metric("Projected Tier", player_ctx.quant_output.tier)
        with col_conf:
            st.metric("Model Confidence", f"{int(player_ctx.quant_output.confidence * 100)}%")
            
        st.divider()
        
        # Report Content
        st.markdown(report)
        
        st.info(f"💡 insight: {player_ctx.player_name}'s TD:INT ratio is in the top 10% of recruits from {player_ctx.measurables.state}.")

    else:
        st.warning("Please enter a player name (e.g., 'Marcus Johnson') to generate a scout card.")

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