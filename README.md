#  Intelligent Candidate Discovery System
### Redrob AI Hackathon — The Data & AI Challenge

> **"Beyond Keyword Matching."** An agentic two-stage hybrid AI pipeline that ranks candidates the way a great recruiter would — by understanding context, not counting keywords.

---

##  Architecture Overview

```
Raw Job Description (any text)
         │
         ▼
┌─────────────────────────────┐
│  Stage 0 — LLM JD Parser   │  ← Gemini 2.5 Flash extracts
│  (jd_parser.py)             │    5 semantic dimensions from JD
└────────────┬────────────────┘
             │  JD Dimension Embeddings
             ▼
┌─────────────────────────────┐
│  Stage 1 — FAISS Retrieval  │  ← Sentence-Transformers embed
│  (retrieval_faiss.py)       │    all candidates into a flat
│                             │    inner-product index.
│  + Semantic Title Filter    │    Filters out irrelevant roles
│  All candidates → Top 300   │    (HR, Sales, etc.) before scoring
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2 — 8-Dimensional    │  ← Fully semantic scoring:
│  Multi-Score Engine         │    • TechnicalScorer (cosine sim)
│  (technical/seniority/      │    • SeniorityScorer (concept emb)
│   behavioral/signal/        │    • BehavioralScorer (decay math)
│   education_scorer.py)      │    • SignalScorer (all 23 signals)
│                             │    • EducationScorer (tier + field)
│  Top 300 → Top 150          │    Zero regex. Zero keyword lists.
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3 — LLM Re-Ranker   │  ← Gemini 2.5 Flash acts as the
│  (llm_reranker.py)          │    ultimate recruiter judge.
│                             │    Outputs: score 0-100 + bespoke
│  Top 150 → Final 100 ranked │    contextual reasoning per candidate
└────────────┬────────────────┘    Resilient fallback scoring included
             │
             ▼
   team_submission.csv + explainability_logs.json
```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Vector Index | FAISS (`IndexFlatIP`, inner-product cosine) |
| LLM | Gemini 2.5 Flash (JD parsing + re-ranking) |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker |
| Data Processing | Pandas, NumPy |

---

##  Key Features

- **Zero Keyword Matching** — every scoring component uses cosine similarity over dense embeddings, not string matching or regex
- **Dynamic JD Understanding** — Gemini parses any raw job description into 5 structured semantic dimensions; no hardcoded JD required  
- **Semantic Title Filtering** — prevents irrelevant roles (HR, Sales, Mechanical) from polluting rankings using embedding similarity
- **Two-Brain Architecture** — FAISS provides millisecond-speed retrieval at scale; Gemini provides deep contextual reasoning for precision
- **8-Dimensional Scoring** — Technical Fit, Seniority, Founding Fit, Signal Score, Education, Evidence Strength, Hiring Probability, Behavioral Fit
- **All 23 Behavioral Signals** — comprehensive integration of every Redrob platform signal across 6 sub-groups
- **Education & Certification Scoring** — semantic field-of-study matching, institution tier, and certification relevance
- **Resilient by Design** — circuit-breaker fallback estimates LLM-equivalent scores locally if API quota is exhausted (no score collapse)

---

##  Setup & Installation

### Prerequisites
- Python 3.9+
- A Gemini API Key ([get one free](https://ai.google.dev/))

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Set environment variables:
```bash
export GEMINI_API_KEY="your_key_here"
export DATA_DIR="/path/to/challenge/data"  # Optional, defaults to challenge_data/
```

Or edit `src/config.py` directly.

### 3. Run the Ranking Pipeline
```bash
python src/build_ranking.py
```

The pipeline will:
1. Parse the JD dynamically via Gemini into 5 semantic dimensions
2. Embed all candidates and index into FAISS
3. Filter out irrelevant titles using semantic similarity
4. Retrieve Top 300 semantically relevant candidates
5. Score all 300 across 8 dimensions
6. LLM re-rank the Top 100 with contextual reasoning
7. Output `outputs/team_submission.csv` and `outputs/explainability_logs.json`

### 4. Launch the FastAPI Dashboard
```bash
python dashboard/app.py
```
Open `http://localhost:5000` in your browser.

API Endpoints:
- `GET /` — Interactive dashboard UI with radar charts and score breakdowns
- `GET /api/dashboard` — Full ranked data (JSON)
- `GET /api/health` — Pipeline status and model info

---

##  Docker

```bash
# Build
docker build -t ai-recruiter .

# Run
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key ai-recruiter
```

---

##  Repository Structure

```
Intelligent-Candidate-Discovery/
├── src/
│   ├── build_ranking.py        # Main pipeline orchestrator
│   ├── config.py               # All configuration in one place
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── jd_parser.py        # LLM-powered JD dimension extractor (5 dimensions)
│   │   ├── data_loader.py      # Stream-based JSONL candidate loader
│   │   ├── embedding_engine.py # Sentence-Transformers wrapper
│   │   ├── retrieval_faiss.py  # FAISS index builder + searcher
│   │   ├── llm_reranker.py     # Gemini re-ranker with resilient fallback
│   │   └── explainer.py        # Rich reasoning generator
│   └── scoring/
│       ├── __init__.py
│       ├── technical_scorer.py # Multi-dimensional JD cosine similarity
│       ├── seniority_scorer.py # Semantic concept embedding scorer
│       ├── behavioral_scorer.py# Hire-Ready Index with decay modeling
│       ├── signal_scorer.py    # All 23 Redrob signals (NEW)
│       └── education_scorer.py # Education + certification scorer (NEW)
├── dashboard/
│   ├── app.py                  # FastAPI application
│   ├── templates/index.html    # Premium dark-theme dashboard with radar charts
│   └── static/                 # CSS + JS assets
├── outputs/
│   ├── team_submission.csv     # Final ranked candidate list (100 candidates)
│   └── explainability_logs.json# Per-candidate score breakdowns + signal details
├── tools/
│   └── validate_submission.py  # Submission format validator
├── docs/
│   └── walkthrough.md          # Methodology & architecture documentation
├── challenge_data/             # Challenge dataset (JSONL not committed)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

##  Output Format

`team_submission.csv` (100 candidates):
```
candidate_id,rank,score,reasoning
CAND_0002025,1,0.8442,"Senior AI engineer with 5.9 years building production ML systems. Strong semantic alignment with ML/Retrieval requirements. Exceptional technical match for embeddings, retrieval, and ranking."
...
```

`explainability_logs.json`:
```json
{
  "CAND_0002025": {
    "rank": 1,
    "total_score": 0.8442,
    "scores": {
      "technical_fit": 0.86,
      "seniority_fit": 0.70,
      "founding_fit": 0.65,
      "signal_score": 0.72,
      "education_fit": 0.80,
      "evidence_strength": 0.75,
      "hiring_probability": 0.78,
      "behavioral_fit": 0.61
    },
    "signal_breakdown": {
      "engagement": 0.68,
      "market_demand": 0.55,
      "availability": 0.72,
      "platform_trust": 0.81,
      "hiring_track_record": 0.65,
      "technical_signals": 0.60
    },
    "llm_reasoning": "..."
  }
}
```

---

##  Scoring Dimensions

| Dimension | Weight | What It Measures |
|---|---|---|
| Technical Fit | 30% | Semantic cosine similarity between JD dimensions and candidate career chunks |
| Seniority Fit | 15% | Evidence of architecture ownership, leadership, and scale |
| Signal Score | 15% | Composite of all 23 Redrob behavioral signals |
| Hiring Probability | 10% | Availability, response rate, notice period, location fit |
| Behavioral Fit | 10% | Platform engagement, profile completeness, interview track record |
| Founding Fit | 10% | Startup DNA, 0-to-1 experience, open-source activity |
| Education Fit | 5% | Field of study relevance, institution tier, certifications |
| Evidence Strength | 5% | Quantifiable impact metrics in career descriptions |

---

##  Validate Submission
```bash
python tools/validate_submission.py
```
