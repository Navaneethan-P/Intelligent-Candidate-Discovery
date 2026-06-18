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
│  Stage 0 — LLM JD Parser   │  ← Gemini 2.0 Flash extracts
│  (jd_parser.py)             │    4 semantic dimensions from JD
└────────────┬────────────────┘
             │  JD Dimension Embeddings
             ▼
┌─────────────────────────────┐
│  Stage 1 — FAISS Retrieval  │  ← Sentence-Transformers embed
│  (retrieval_faiss.py)       │    all 5,000+ candidates into
│                             │    a flat inner-product index.
│  5000 candidates → Top 200  │    Multi-dimensional JD query
└────────────┬────────────────┘    narrows to Top 200 in ~ms
             │
             ▼
┌─────────────────────────────┐
│  Stage 2 — Multi-Score      │  ← Fully semantic scoring:
│  (technical/seniority/      │    • TechnicalScorer (cosine sim)
│   behavioral_scorer.py)     │    • SeniorityScorer (concept emb)
│                             │    • BehavioralScorer (decay math)
│  Top 200 → Top 50           │    Zero regex. Zero keyword lists.
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3 — LLM Re-Ranker   │  ← Gemini 2.0 Flash acts as the
│  (llm_reranker.py)          │    ultimate recruiter judge.
│                             │    Outputs: score 0-100 + bespoke
│  Top 50 → Final 50 ranked   │    contextual reasoning per candidate
└────────────┬────────────────┘    Circuit breaker fallback included
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
| LLM | Gemini 2.0 Flash (JD parsing + re-ranking) |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker |
| Data Processing | Pandas, NumPy |

---

##  Key Features

- **Zero Keyword Matching** — every scoring component uses cosine similarity over dense embeddings, not string matching or regex
- **Dynamic JD Understanding** — Gemini parses any raw job description into structured semantic dimensions; no hardcoded JD required  
- **Two-Brain Architecture** — FAISS provides millisecond-speed retrieval at scale; Gemini provides deep contextual reasoning for precision
- **Hire-Ready Index™** — proprietary behavioral signal that combines recruiter response rate, notice period, inactivity decay, and GitHub activity into one metric
- **Resilient by Design** — circuit-breaker fallback generates clean structured reasoning locally if LLM quota is exhausted

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
Open `src/config.py` and set:
```python
GEMINI_API_KEY = "your_key_here"   # or set as env variable GEMINI_API_KEY
DATA_DIR = "/path/to/challenge/data"
MAX_CANDIDATES = 5000              # Set to None to process all candidates
```

### 3. Run the Ranking Pipeline
```bash
cd ai_recruiter_submission
python src/build_ranking.py
```

The pipeline will:
1. Parse the JD dynamically via Gemini
2. Embed all candidates and index into FAISS
3. Retrieve Top 200 semantically relevant candidates
4. Score all 200 across 6 dimensions
5. LLM re-rank the Top 50 with contextual reasoning
6. Output `outputs/team_submission.csv` and `outputs/explainability_logs.json`

### 4. Launch the FastAPI Dashboard
```bash
python dashboard/app.py
```
Open `http://localhost:5000` in your browser.

API Endpoints:
- `GET /` — Interactive dashboard UI
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
ai_recruiter_submission/
├── src/
│   ├── build_ranking.py        # Main pipeline orchestrator
│   ├── config.py               # All configuration in one place
│   ├── pipeline/
│   │   ├── jd_parser.py        # LLM-powered JD dimension extractor
│   │   ├── data_loader.py      # Stream-based JSONL candidate loader
│   │   ├── embedding_engine.py # Sentence-Transformers wrapper
│   │   ├── retrieval_faiss.py  # FAISS index builder + searcher
│   │   ├── llm_reranker.py     # Gemini re-ranker with circuit breaker
│   │   └── explainer.py        # Fallback reasoning generator
│   └── scoring/
│       ├── technical_scorer.py # Multi-dimensional JD cosine similarity
│       ├── seniority_scorer.py # Semantic concept embedding scorer
│       └── behavioral_scorer.py# Hire-Ready Index with decay modeling
├── dashboard/
│   ├── app.py                  # FastAPI application
│   ├── templates/index.html    # Dashboard UI
│   └── static/                 # CSS + JS assets
├── outputs/
│   ├── team_submission.csv     # Final ranked candidate list
│   └── explainability_logs.json# Per-candidate score breakdowns
├── Dockerfile
└── requirements.txt
```

---

##  Output Format

`team_submission.csv`:
```
candidate_id,rank,score,reasoning
CAND_0001056,1,0.4821,"Senior ML Engineer with 8 years at Flipkart. Strong semantic alignment with ML/Retrieval requirements. High GitHub activity (87) signals active open-source engagement."
...
```

`explainability_logs.json`:
```json
{
  "CAND_0001056": {
    "rank": 1,
    "total_score": 0.4821,
    "scores": {
      "technical_fit": 0.82,
      "seniority_fit": 0.70,
      "founding_fit": 0.65,
      "hiring_probability": 0.78,
      "behavioral_fit": 0.61,
      "evidence_strength": 0.75
    },
    "llm_reasoning": "..."
  }
}
```

---

##  Validate Submission
```bash
python tools/validate_submission.py
```
