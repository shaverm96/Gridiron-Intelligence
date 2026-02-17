import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv("GEMINI_API_KEY.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- DATA STRUCTURES ---

@dataclass
class Measurables:
    """Physical measurements and location."""
    height_feet: int
    height_inches: int
    weight_lbs: int
    state: str
    
    @property
    def height_display(self) -> str:
        return f"{self.height_feet}'{self.height_inches}\""
    
    @property
    def total_height_inches(self) -> int:
        return (self.height_feet * 12) + self.height_inches


@dataclass
class HighSchoolStats:
    """High school performance metrics."""
    passing_yards: int
    passing_tds: int
    interceptions: int
    completion_pct: float
    star_rating: int  # 3-5 stars
    
    @property
    def td_int_ratio(self) -> float:
        return self.passing_tds / max(self.interceptions, 1)


@dataclass
class XGBoostOutput:
    """Quantitative model predictions."""
    raw_score: float  # 0-100
    tier: str  # e.g., "Power 4 Multi-Year Starter"
    confidence: float  # 0.0-1.0


@dataclass
class PlayerContext:
    """Complete player profile for scouting."""
    player_name: str
    position: str
    high_school: str
    measurables: Measurables
    stats: HighSchoolStats
    target_school: str
    target_school_tier: str  # "Power 4" or "G5"
    quant_output: Optional[XGBoostOutput] = None
    rag_insights: List[str] = field(default_factory=list)


# --- LOGIC ENGINES ---

def run_quant_engine(player: PlayerContext) -> XGBoostOutput:
    """Simulated XGBoost scoring engine."""
    score = 50.0  # Baseline
    
    # Height factor (QBs)
    if player.position == "QB" and player.measurables.total_height_inches >= 76:
        score += 8.0
    
    # Weight-to-height ratio
    height_m = player.measurables.total_height_inches * 0.0254
    weight_kg = player.measurables.weight_lbs * 0.453592
    bmi = weight_kg / (height_m ** 2)
    if 23 <= bmi <= 27:  # Athletic build
        score += 5.0
    
    # Production
    if player.stats.passing_yards > 3500:
        score += 7.0
    if player.stats.passing_tds > 35:
        score += 6.0
    
    # Decision-making
    if player.stats.completion_pct > 62.0:
        score += 5.0
    if player.stats.td_int_ratio > 4.0:
        score += 8.0
    
    # Star rating
    score += (player.stats.star_rating - 3) * 3.0
    
    # Geographic fit
    if player.measurables.state in ["NC", "SC", "VA"]:
        score += 3.0
    
    # Cap score at 100
    score = min(score, 100.0)
    
    # Tier classification
    if score >= 90:
        tier = "Future NFL Draft Pick"
    elif score >= 80:
        tier = "Power 4 Multi-Year Starter"
    elif score >= 70:
        tier = "Power 4 Contributor / Competition Window"
    elif score >= 60:
        tier = "G5 Potential / Power 4 Depth"
    else:
        tier = "G5 / FBS Depth Option"
    
    # Simulated confidence
    confidence = 0.75 + (player.stats.star_rating - 3) * 0.05
    
    return XGBoostOutput(
        raw_score=round(score, 1),
        tier=tier,
        confidence=round(confidence, 2)
    )

# Analytical Fact Bank
ANALYTICS_FACT_BANK = [
    {"fact": "QBs with height ≥ 6'4\" have a 28% higher success rate at Power 4 programs.", "tags": ["QB", "Height", "Power 4", "Measurables"]},
    {"fact": "In-state recruits to G5 programs show 18% better retention and development outcomes.", "tags": ["G5", "Geographic", "Retention"]},
    {"fact": "QBs with completion % > 62% in high school transition to college-level reads 35% faster.", "tags": ["QB", "Completion", "Decision-Making"]},
    {"fact": "TD:INT ratio > 4.0 correlates with 0.42 probability of earning starting role within 2 years.", "tags": ["QB", "TD:INT", "Decision-Making", "Statistical"]},
    {"fact": "4-star QBs have only 12% higher success rate than 3-star QBs when controlling for measurables.", "tags": ["QB", "Star Rating", "Recruiting Bias"]},
    {"fact": "G5 programs develop QBs at comparable rates to Power 4 when adjusted for input talent.", "tags": ["G5", "QB", "Development"]},
    {"fact": "Production volume (3500+ yards) indicates durability and offensive scheme fit.", "tags": ["QB", "Production", "Yards"]}
]

def retrieve_relevant_insights(player: PlayerContext, fact_bank: List[Dict]) -> List[str]:
    """Tag-based fact retrieval."""
    insights = []
    player_tags = [player.position, player.target_school_tier]
    
    if player.measurables.total_height_inches >= 76: player_tags.append("Height")
    if player.stats.completion_pct > 62: player_tags.append("Completion")
    if player.stats.td_int_ratio > 4.0: player_tags.append("TD:INT")
    if player.stats.passing_yards > 3500: player_tags.append("Production")
    if player.stats.star_rating in [3, 4, 5]: player_tags.append("Star Rating")
    
    for fact_obj in fact_bank:
        if any(tag in fact_obj["tags"] for tag in player_tags):
            insights.append(fact_obj["fact"])
            
    return insights[:5]

# --- LLM INTEGRATION ---

def generate_scout_report_llm(player: PlayerContext):
    """
    Direct LLM Generation using the Prompt from the Notebook.
    Instead of a full agent (which requires tools), we can use a direct Chain 
    since all data is available in the context.
    """
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05", # Updated model name
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=2000
    )
    
    # Prepare Data Block
    rag_text = "\n".join([f"- {i}" for i in player.rag_insights])
    
    prompt_text = f"""You are acting as a college football scout. Your task is to evaluate high school players for college recruitment based on player data, quantitative model outputs, and analytical research findings.

    ======== PLAYER DATA ========
    Name: {player.player_name}
    Position: {player.position}
    High School: {player.high_school}
    Measurables: {player.measurables.height_display} | {player.measurables.weight_lbs} lbs | {player.measurables.state}

    ======== HIGH SCHOOL PERFORMANCE ========
    Passing Yards: {player.stats.passing_yards}
    Passing TDs: {player.stats.passing_tds}
    Interceptions: {player.stats.interceptions}
    Completion %: {player.stats.completion_pct}%
    TD:INT Ratio: {player.stats.td_int_ratio:.1f}
    Star Rating: {player.stats.star_rating} stars

    ======== MODEL ASSESSMENT (Proprietary Quant Engine) ========
    Raw Score: {player.quant_output.raw_score}/100
    Tier: {player.quant_output.tier}
    Model Confidence: {player.quant_output.confidence}

    ======== TARGET SCHOOL ========
    School: {player.target_school}
    Tier: {player.target_school_tier}

    ======== ANALYTICAL RESEARCH FINDINGS ========
    {rag_text}

    ======== YOUR TASK (Scout's Report) ========
    Write a professional scouting summary (3-4 paragraphs, 300-400 words) following these guidelines:

    1. **ACKNOWLEDGE THE SCORE**: Reference the Quant Engine's score and tier.
    2. **JUSTIFY THE SCORE**: Use 2-3 of the Analytical Research Findings above to explain WHY the model projects this outcome.
    3. **FIT AT TARGET**: Assess the player's fit at {player.target_school}.
    4. **COMPARISON**: Briefly compare to historical player archetypes if relevant.
    5. **TONE**: Confident, measured, professional. No "According to the data".

    Generate the report now:
    """
    
    response = llm.invoke(prompt_text)
    return response.content
