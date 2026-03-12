updaimport os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

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
    Generates a scouting report using a LangChain Agent with tool calling,
    replicating the logic from the notebook template.
    """
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05", 
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=2000
    )
    
    # Define Tool (Closing over the 'player' context)
    @tool
    def get_player_data(query: str) -> str:
        """Retrieve scouting data, stats, and model projections for the player."""
        p = player
        rag_text = "\n".join([f"- {i}" for i in p.rag_insights])
        
        return f"""
        PLAYER PROFILE: {p.player_name}
        Position: {p.position}
        Measurables: {p.measurables.height_display}, {p.measurables.weight_lbs} lbs, {p.measurables.state}
        
        STATS:
        Passing: {p.stats.passing_yards} yds, {p.stats.passing_tds} TDs
        INTs: {p.stats.interceptions}
        Completion %: {p.stats.completion_pct}
        TD:INT Ratio: {p.stats.td_int_ratio:.1f}
        
        QUANT MODEL:
        Score: {p.quant_output.raw_score}/100
        Tier: {p.quant_output.tier}
        Confidence: {p.quant_output.confidence}
        
        TARGET SCHOOL:
        {p.target_school} ({p.target_school_tier})
        
        RESEARCH INSIGHTS:
        {rag_text}
        """

    tools = [get_player_data]

    # Agent System Message (Exact replica from notebook)
    agent_system_message = """You are acting as a college football scout. Your task is to evaluate high school players for college recruitment based on player data, quantitative model outputs, and analytical research findings.

When you receive the player data from the tool, write a professional scouting summary (2-3 paragraphs, 250-350 words) following these guidelines:

1. **ACKNOWLEDGE THE SCORE**: Reference the Quant Engine's score and tier.
2. **JUSTIFY THE SCORE**: Use the Analytical Research Findings to explain WHY the model projects this outcome.
3. **INITIAL THOUGHTS**: Based on the player's measurables, stats, and research insights, discuss the player's potential.
4. **POSSIBLE FALL BACKS**: Based on how high the models confidence is (if below .5 mention fall backs more, if between (.5 and .75) mention fall backs less, if above .75 mention fall backs least), mention potential "floor" scenarios or development paths for the player that are common in scouting but appear in the data.
5. **FIT AT TARGET SCHOOL**: Assess the player's fit at their target school.
6. **HISTORICAL COMPARISON**: Compare the player to 2 - 3 historical recruits by name with similar profiles and outcomes in one short sentence stating players and 2-3 traits they share.
7. **SCOUT LANGUAGE**: Use professional terminology naturally (e.g., "High ceiling", "Twitchy", "Processes quickly").
8. **TONE**: Confident but measured.

Do NOT use phrases like "According to the research", "The data shows.", "a potential "floor" scenario", and "a potential "ceiling" scenario" scenario Integrate findings naturally.
Make sure to focus on the player's potential and fit, not just their limitations. Acknowledge both strengths and areas for development, but maintain an overall positive and constructive tone as a scout would when discussing a recruit with coaches.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoke Agent
    scout_task = f"Generate a comprehensive scouting report for {player.player_name}. Use the get_player_data tool to retrieve his profile first."
    
    result = agent_executor.invoke({"input": scout_task})
    return result["output"]
