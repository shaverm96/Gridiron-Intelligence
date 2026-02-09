# Gridiron Intelligence: AI-Powered Football Recruiting Assistant

## Executive Summary

**Gridiron Intelligence** is an AI-powered system that generates professional "Scouting Cards" for high school football players, combining statistical predictions with narrative analysis to guide college recruiting decisions.

The system outputs two key components:
1. **Statistical Projection**: A probability score predicting a recruit's college tier (e.g., "85% chance of being a Power 4 Starter")
2. **Narrative Scouting Report**: Natural-language analysis explaining the statistical prediction with historical comparisons and key performance indicators

---

## The Dual-Engine Architecture

To overcome the fundamental limitation that large language models struggle with quantitative analysis, Gridiron Intelligence employs a split-brain approach:

### Engine A: The Quant (XGBoost)
- **Role**: The Calculator
- **Input**: Hard statistics (Height, Weight, 40-time, High School Production, Star Rating, Measurables)
- **Output**: Probability score (0-100) predicting player tier and performance ceiling
- **Purpose**: Provides accurate numerical predictions based on historical patterns

### Engine B: The Scout (Gemini + RAG)
- **Role**: The Storyteller
- **Input**: Quant Engine score + Historical Player Comparisons + Research Facts + Context
- **Output**: Professional scouting narrative justifying the quantitative projection
- **Example**: "Despite his 3-star rating, the model projects him as a starter because his height-to-weight ratio matches 90% of successful SEC linebackers."
- **Purpose**: Bridges the gap between raw numbers and actionable recruiting intelligence

---

## Current Status: Phase 2.5 - LangChain Orchestration

### Phase 1 - Data Engineering ✅ COMPLETED

#### Completed Tasks

#### Data Collection
- **Recruitment Data**: High school football recruits from Class of 2015-2028
  - Location: `/data/Football Reruitment Tables/Recruitment Classes/`
  - Format: Individual CSV files per year (e.g., `recruits_2015.csv` through `recruits_2028.csv`)
  - Content: Recruit names, positions, high school stats, star ratings, committed schools

- **Scouting Reports**: Professional evaluations from recruitment services
  - Location: `/data/Football Reruitment Tables/Scouting Reports/`
  - Format: Year-by-year CSV files matching recruitment classes
  - Content: Detailed analyst notes and evaluations

- **College Statistics**: Comprehensive player performance data
  - Location: `/data/Stats/` with subdirectories by category:
    - `rosters/`: Team rosters by school and year (2015-2025)
    - `passing/`, `rushing/`, `receiving/`: Offensive performance metrics
    - `defense/`: Defensive player statistics
    - `kicking/`, `punting/`: Specialist data
  - Content: Position-specific performance metrics for outcome validation

- **Compiled Player Stats**: Integrated dataset combining college performance
  - Location: `/data/compiled_player_stats_with_defense.csv`
  - Purpose: Ground truth for model training

#### Name Matching & Entity Resolution
- **Primary Matcher**: School mapping with exact name matching
  - Location: `/data/name_matching/matches_primary_layer.csv`
  
- **Backup Matcher**: Fuzzy matching with 5-variant strategies
  - Location: `/data/name_matching/matches_backup_layer.csv`
  - Variants: Last name matching, phonetic matching, partial matching, etc.
  
- **Final Matcher**: Position-based and initial matching refinement
  - Location: `/data/name_matching/matches_final_layer.csv`
  
- **Consolidated Results**: All successful matches
  - Location: `/data/name_matching/all_matches_combined.csv`
  - Unmatched Recruits: `/data/name_matching/unmatched_recruits.csv`

- **Support Files**:
  - School mapping dictionary: `college_to_recruit_schools_mapping.json`
  - Nickname standardization: `nickname_map.json`
  - Cleaned datasets: `all_recruits_cleaned.csv`, `all_rosters_cleaned.csv`

#### Processing Infrastructure
- **Sports Reference Scraper** (`notebooks/sports_ref_scraper.ipynb`): Automated data collection from Sports-Reference.com
  - Handles passing, rushing, receiving, kicking, punting categories
  - Years covered: 2015-2025
  
- **Recruit-to-Roster Matcher** (`notebooks/recruit_to_roster_matcher_v2.ipynb`): Multi-strategy name resolution
  - Matches high school recruits to college rosters
  - Three-layer matching strategy for maximum accuracy

### Key Achievements
✓ Built "Ground Truth" dataset linking high school recruits (2015-2021 classes) to college outcomes  
✓ Implemented sophisticated name-matching pipeline with 3 fallback strategies  
✓ Collected comprehensive college statistics across all major schools (2015-2025)  
✓ Standardized naming conventions and school mappings  
✓ Refactored all processing scripts to use portable relative paths for team collaboration  

### Phase 2 - LLM Integration ✅ COMPLETED

#### Key Achievements
✓ Successfully integrated Gemini API (gemini-2.0-flash-exp) for narrative generation  
✓ Developed sophisticated prompt engineering with persona-driven approach  
✓ Implemented 7-metric validation framework (85%+ pass rate required)  
✓ Established baseline performance: <2s latency, ~400-500 tokens per report  
✓ Created dual-engine architecture separating quantitative (XGBoost simulation) from qualitative (LLM) reasoning  
✓ Prototype notebook: `notebooks/gemini_scout_engine.ipynb`  

#### Current Implementation
- **Scout Engine**: Direct Gemini API integration with structured prompt template
- **RAG System**: Tag-based fact retrieval (6 curated QB insights)
- **Validation**: Quality checks for terminology, structure, and context usage
- **Cost**: ~$0.08 per 1M input tokens, ~$0.4 per 1M output tokens

### Project Structure
```
Gridiron_Intelligence/
├── notebooks/
│   ├── sports_ref_scraper.ipynb           # Data collection
│   ├── recruit_to_roster_matcher_v2.ipynb # Name resolution
│   ├── gemini_scout_engine.ipynb          # Phase 2: Direct Gemini API prototype
│   └── langchain_scout_template.ipynb     # Phase 2.5: LangChain orchestration
├── data/
│   ├── compiled_player_stats_with_defense.csv  # Integrated ground truth
│   ├── Football Reruitment Tables/
│   │   ├── Recruitment Classes/          # Raw recruit data (2015-2028)
│   │   └── Scouting Reports/             # Professional evaluations
│   ├── Stats/
│   │   ├── rosters/                      # College team rosters
│   │   ├── passing/, rushing/, etc/      # Player stats by category
│   │   └── defense/                      # Defensive metrics
│   └── name_matching/                    # Matcher outputs & mappings
├── src/                                  # Future: Model training code
├── requirements.txt
└── README.md
```

---

## Roadmap: Phases 2.5 - 6

### Phase 2.5: LangChain Orchestration 🔄 IN PROGRESS

**Objective:** Modernize Scout Engine with LangChain for modularity, testing, and conversational capabilities.

**Notebook:** `notebooks/langchain_scout_template.ipynb`

#### Implemented Features
- [x] `PromptTemplate`: Modular prompt engineering with template variables (replaces string concatenation)
- [x] `ChatGoogleGenerativeAI`: Gemini wrapper for model abstraction and easy swapping
- [x] `LLMChain`: Sequential orchestration of Quant → RAG → Scout pipeline
- [x] `ConversationBufferMemory`: Store player context for follow-up Q&A without regenerating reports
- [x] Tag-based RAG with ChromaDB migration path documented
- [x] Validation framework ported from Phase 2 prototype
- [x] Synthetic data factory with real-data integration hooks

#### Key Improvements Over Phase 2
- **Model Flexibility**: Abstract interface for testing Claude, GPT-4, or other LLMs
- **Prompt Versioning**: Template system enables A/B testing and prompt iteration
- **Conversational Context**: Ask follow-up questions ("Why did he score 78?", "What's his biggest weakness?") without full regeneration
- **Production Ready**: Structured for real data integration with load_real_player() stubs

#### Next Steps
- [ ] Test with multiple player profiles (different positions, star ratings, school tiers)
- [ ] Benchmark LangChain vs. direct API (latency, token usage)
- [ ] Document prompt template variants for A/B testing
- [ ] Add OutputParser for structured validation

---

### Phase 3: Quantitative Model Training (XGBoost Production)

**Objective:** Replace simulated Quant Engine with trained XGBoost classifier.

#### Tasks
- [ ] Feature engineering from `all_matches_combined.csv`:
  - Measurables (height, weight, BMI)
  - Production metrics (yards, TDs, efficiency)
  - Contextual factors (star rating, geography, school tier)
- [ ] Train/test split on recruit classes (2015-2019 train, 2020-2021 test)
- [ ] Target variable: College outcome tier (NFL Draft, Power 4 Starter, G5, etc.)
- [ ] Hyperparameter tuning (GridSearchCV, cross-validation)
- [ ] Feature importance analysis and SHAP values
- [ ] Model persistence and versioning

**Expected Output:** `models/quant_engine_v1.pkl` with 70%+ accuracy on test set

---

### Phase 4: Advanced RAG with Vector Search

**Objective:** Upgrade from tag-based retrieval to semantic search with ChromaDB.

#### Tasks
- [ ] Build vector database from historical player profiles:
  - Embed player archetypes from `all_matches_combined.csv`
  - Store college outcomes, development trajectories, and performance data
  - Index by position, measurables, school tier
- [ ] Implement semantic search:
  - Convert player profiles to embeddings (sentence-transformers)
  - Query for top-k similar players (k=5)
  - Return comparison players with outcome context
- [ ] Enhance RAG insights:
  - Replace hardcoded fact bank with dynamic retrieval
  - Include historical player comparisons ("Similar to [Player X] who became...")
  - Cite data sources and statistical significance
- [ ] ChromaDB persistence: `data/chromadb/player_vectors/`

**Expected Output:** RAG retrieval with semantic relevance, historical player comparisons

---

### Phase 5: Multi-Model Experimentation & Cost Optimization

**Objective:** Compare LLM performance and optimize for cost/quality tradeoff.

#### Tasks
- [ ] Implement multi-model testing:
  - Gemini (current baseline)
  - Claude (Anthropic)
  - GPT-4 (OpenAI)
  - Llama 3 (local deployment option)
- [ ] Benchmark framework:
  - Output quality (domain expert scoring)
  - Cost per report
  - Latency (p50, p95, p99)
  - Terminology accuracy
- [ ] Prompt optimization per model (different models may need tailored prompts)
- [ ] Fallback chain for reliability (if primary model fails, use backup)

**Expected Output:** Model recommendation matrix with cost/quality tradeoffs

---

### Phase 6: Agent Architecture & Advanced Features

**Objective:** Enable interactive, tool-augmented scouting with multi-turn reasoning.

#### Features
- [ ] **ConversationalRetrievalChain**: Advanced Q&A with document retrieval
- [ ] **Agent Tools**:
  - `query_player_stats`: Fetch specific stats on-demand from CSV data
  - `compare_players`: Side-by-side differential analysis
  - `generate_depth_chart`: Position-specific rankings for recruiting classes
  - `explain_quant_score`: Break down XGBoost feature contributions
- [ ] **Memory Systems**:
  - `ConversationBufferWindowMemory`: Multi-session context
  - `VectorStoreRetrieverMemory`: Reference historical evaluations
- [ ] **Player Comparison Feature** (Stretch Goal):
  - Compare two players with differential Quant analysis
  - Recruitment priority recommendation
  - Fit analysis for specific team schemes

**Expected Output:** Interactive scouting assistant with tool use and multi-turn reasoning

---

## Dependencies

See `requirements.txt` for full list. Key packages:

**Data Engineering (Phase 1):**
- `pandas`: Data manipulation
- `requests`, `beautifulsoup4`: Web scraping
- `fuzzywuzzy`: Fuzzy string matching

**LLM Integration (Phase 2 & 2.5):**
- `google-generativeai`: Direct Gemini API (Phase 2 prototype)
- `langchain`: Orchestration framework (Phase 2.5)
- `langchain-google-genai`: LangChain Gemini wrapper
- `python-dotenv`: Environment variable management

**RAG & Vector Search (Phase 4):**
- `chromadb`: Vector database for semantic search
- `sentence-transformers`: Embedding generation (future)

**Quantitative Modeling (Phase 3):**
- `xgboost`: Gradient boosting classifier
- `scikit-learn`: Model evaluation and preprocessing

---

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Data Collection** (Phase 1): Run `notebooks/sports_ref_scraper.ipynb` to gather college statistics
3. **Name Matching** (Phase 1): Run `notebooks/recruit_to_roster_matcher_v2.ipynb` to build ground truth dataset
4. **LLM Prototype** (Phase 2): Explore `notebooks/gemini_scout_engine.ipynb` for direct Gemini API implementation
5. **LangChain Template** (Phase 2.5): Run `notebooks/langchain_scout_template.ipynb` for orchestrated scout engine with conversational memory

---

## Team & Attribution

**Spring 2026 - DSBA 6010: Large Language Models**  
Group 4: Gridiron Intelligence Project  
UNC Charlotte

---

## Notes for Developers

- All file paths use relative path resolution via `BASE_DATA_DIR` and `BASE_PROJECT_DIR` for portability
- Notebook environment configured for collaborative development
- Data files contain historical records (2015-2028) with no personal identifiable information
- Naming convention: `{team}_{year}_{stat_type}.csv`