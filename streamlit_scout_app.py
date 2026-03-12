import streamlit as st
import os
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="Gridiron Intelligence: Scout Card",
    page_icon="🏈",
    layout="wide"
)

# --- MOCKED DATA STRUCTURES ---
# (Temporarily defined here to avoid importing heavy dependencies from scout_card_engine)

class Measurables:
    """Mock Measurables"""
    def __init__(self, height_feet, height_inches, weight_lbs, state):
        self.height_feet = height_feet
        self.height_inches = height_inches
        self.weight_lbs = weight_lbs
        self.state = state
    
    @property
    def height_display(self):
        return f"{self.height_feet}'{self.height_inches}\""

class HighSchoolStats:
    """Mock Stats"""
    def __init__(self, passing_yards, passing_tds, interceptions, completion_pct, star_rating):
        self.passing_yards = passing_yards
        self.passing_tds = passing_tds
        self.interceptions = interceptions
        self.completion_pct = completion_pct
        self.star_rating = star_rating

class XGBoostOutput:
    """Mock Output"""
    def __init__(self, raw_score, tier, confidence):
        self.raw_score = raw_score
        self.tier = tier
        self.confidence = confidence

class PlayerContext:
    """Mock Context"""
    def __init__(self, player_name, position, high_school, measurables, stats, target_school, target_school_tier):
        self.player_name = player_name
        self.position = position
        self.high_school = high_school
        self.measurables = measurables
        self.stats = stats
        self.target_school = target_school
        self.target_school_tier = target_school_tier
        self.quant_output = None
        self.rag_insights = []

# --- MOCKED LOGIC ENGINES ---

def run_quant_engine(player):
    """Mock Quantitative Engine"""
    return XGBoostOutput(85.0, "Power 4 Contributor", 0.82)

ANALYTICS_FACT_BANK = []

def retrieve_relevant_insights(player, fact_bank):
    """Mock RAG"""
    return [
        "Insight 1: Strong production indicates potential.",
        "Insight 2: Measurables align with position standards."
    ]

def generate_scout_report_llm(player):
    """Mock LLM Report"""
    return f"""
    ### Scouting Report for {player.player_name}
    
    **Summary:**
    This is a preliminary report based on the provided data. The full AI analysis engine is currently being optimized.
    
    **Strengths:**
    - Good size for the position ({player.measurables.height_display}, {player.measurables.weight_lbs} lbs).
    - Solid production with {player.stats.passing_yards} yards.
    
    **Projection:**
    Based on the mock score of 85.0, this player projects as a Power 4 Contributor.
    """

# --- CONFIGURATION & SETUP ---
# Load environment variables
load_dotenv("GEMINI_API_KEY.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.warning("⚠️ GEMINI_API_KEY not found. Running in mock mode.")

st.title("🏈 Gridiron Intelligence: Scout Scorecard")
st.markdown("### AI-Powered Recruitment Analysis")

# Sidebar - Player Input
with st.sidebar:
    st.header("Player Profile")
    
    # Personal
    p_name = st.text_input("Player Name", "Marcus Johnson")
    p_high_school = st.text_input("High School", "Riverside High")
    p_pos = st.selectbox("Position", ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"])
    
    # Measurables
    st.subheader("Measurables")
    col1, col2 = st.columns(2)
    with col1:
        p_height_ft = st.number_input("Height (ft)", 5, 7, 6)
        p_weight = st.number_input("Weight (lbs)", 150, 400, 215)
    with col2:
        p_height_in = st.number_input("Height (in)", 0, 11, 4)
        p_state = st.text_input("State", "NC")
        
    # Stats
    st.subheader("Performance Data")
    p_yards = st.number_input("Passing Yards", 0, 10000, 3850)
    p_tds = st.number_input("Passing TDs", 0, 100, 42)
    p_ints = st.number_input("Interceptions", 0, 50, 8)
    p_comp_pct = st.slider("Completion %", 40.0, 100.0, 64.2)
    p_stars = st.slider("Star Rating", 1, 5, 4)
    
    # Target
    st.subheader("Recruitment Target")
    p_target_school = st.text_input("Target School", "UNC Charlotte")
    p_school_tier = st.selectbox("School Tier", ["Power 4", "G5", "FCS"])
    
    if st.button("Generate Scout Card", type="primary"):
        # Build Player Context
        measurables = Measurables(p_height_ft, p_height_in, p_weight, p_state)
        stats = HighSchoolStats(p_yards, p_tds, p_ints, p_comp_pct, p_stars)
        
        player_ctx = PlayerContext(
            player_name=p_name,
            position=p_pos,
            high_school=p_high_school,
            measurables=measurables,
            stats=stats,
            target_school=p_target_school,
            target_school_tier=p_school_tier
        )
        
        # Run Engine A (Quant)
        player_ctx.quant_output = run_quant_engine(player_ctx)
        
        # Run RAG
        player_ctx.rag_insights = retrieve_relevant_insights(player_ctx, ANALYTICS_FACT_BANK)
        
        # Store in session state to persist across reruns
        st.session_state['player_context'] = player_ctx
        
        # Clear previous report to force regeneration

        if 'report_text' in st.session_state:
            del st.session_state['report_text']
            
        st.rerun()

# Display Area
if 'player_context' in st.session_state:
    player = st.session_state['player_context']
    
    # Display Scorecard
    st.header(f"{player.player_name} - {player.position}")
    st.caption(f"{player.high_school} | {player.measurables.height_display}, {player.measurables.weight_lbs} lbs")
    
    col_score, col_tier, col_conf = st.columns(3)
    
    with col_score:
        st.metric("Quant Score", f"{player.quant_output.raw_score}/100")
    with col_tier:
        st.metric("Projected Tier", player.quant_output.tier)
    with col_conf:
        st.metric("Model Confidence", f"{int(player.quant_output.confidence * 100)}%")
        
    st.divider()
    
    # Check if report already generated
    if 'report_text' not in st.session_state:
        with st.spinner(f"Drafting scouting report for {player.player_name}..."):
            st.session_state['report_text'] = generate_scout_report_llm(player)
            
    st.subheader("📝 Scouting Report")
    st.markdown(st.session_state['report_text'])
    
    st.divider()
    st.subheader("🔍 Analytical Insights (RAG Context)")
    for insight in player.rag_insights:
        st.info(f"💡 {insight}")

else:
    st.info("👈 Enter player details in the sidebar and click 'Generate Scout Card' to begin.")
