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

## Current Status: Phase 1 - Data Engineering

### Completed Tasks

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

### Project Structure
```
Gridiron_Intelligence/
├── notebooks/
│   ├── sports_ref_scraper.ipynb          # Data collection
│   └── recruit_to_roster_matcher_v2.ipynb # Name resolution
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

## Next Phase: Phase 2 - LLM Integration & Model Selection

### Milestone: LLM Experimentation & Baseline Establishment

#### Task 1: Multi-Model Experimentation
- [ ] Configure and test multiple LLM candidates:
  - **Gemini API** (Primary candidate): Evaluate for quality and cost-effectiveness
  - Comparison models: Claude, GPT-4, other domain-relevant alternatives
- [ ] Establish consistent testing harness with representative examples
- [ ] Test prompt engineering approaches specific to scouting domain

#### Task 2: Representative Test Cases
- [ ] Build test dataset with:
  - Sample recruit profiles (High school stats, measurables, star rating)
  - Quant Engine scores (projected outcomes)
  - Desired narrative output (scout perspective)
- [ ] Test prompt: "Analyze this recruit as a college scout would. Explain why the model projects [SCORE]. Reference similar players and key indicators."
- [ ] Iteratively refine prompts for output quality, consistency, and reasoning clarity

#### Task 3: Model Selection Rationale
Document comparison across:
- **Output Quality**: Realism of scouting language, accuracy of reasoning, coherence
- **Cost**: Per-token pricing, cost per scouting card generation
- **Latency**: Response time for real-time card generation
- **Reliability**: Uptime, API stability, documentation quality

#### Task 4: Baseline Performance Metrics
- [ ] Establish performance benchmarks for selected model(s):
  - Average response quality (scored by domain experts)
  - Consistency across similar player profiles
  - Factual accuracy verification
  - Output length and readability metrics

---

## Future Roadmap

### Phase 3: Quantitative Model Development
- Build and train XGBoost classifier on matched recruit-to-outcome dataset
- Hyperparameter tuning and cross-validation
- Feature importance analysis

### Phase 4: RAG & Context Enhancement
- Integrate historical player comparison retrieval
- Build vector database of similar player archetypes
- Enhance prompt context with relevant statistics

### Phase 5: Production System
- Build full scouting card generation pipeline
- API interface for card requests
- Performance monitoring and model retraining

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`: Data manipulation
- `requests`, `beautifulsoup4`: Web scraping
- `fuzzywuzzy`: Fuzzy string matching
- `xgboost`: Quantitative model (Phase 3)
- `google-cloud-generativeai`: Gemini API integration (Phase 2)

---

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Data Collection**: Run `notebooks/sports_ref_scraper.ipynb` to gather college statistics
3. **Name Matching**: Run `notebooks/recruit_to_roster_matcher_v2.ipynb` to build ground truth dataset
4. **LLM Testing** (Phase 2): Follow experimentation protocol in Phase 2 section

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