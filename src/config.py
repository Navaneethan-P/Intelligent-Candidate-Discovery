import os

# Base Directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "challenge_data", "India_runs_data_and_ai_challenge"))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Files
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
SUBMISSION_FILE = os.path.join(OUTPUTS_DIR, "team_submission.csv")
EXPLAINABILITY_LOGS = os.path.join(OUTPUTS_DIR, "explainability_logs.json")

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MAX_CANDIDATES = None  # Process ALL candidates for full dataset coverage

# --- Scoring Weights ---
# 8 scoring dimensions — weights must sum to 1.0
WEIGHTS = {
    'technical_fit':       0.30,   # Core JD semantic match
    'seniority_fit':       0.15,   # Leadership, architecture, scale evidence
    'founding_fit':        0.10,   # Startup / 0-to-1 DNA
    'signal_score':        0.15,   # All 23 redrob behavioral signals composite
    'education_fit':       0.05,   # Degree relevance + institution tier
    'evidence_strength':   0.05,   # Quantifiable impact in career descriptions
    'hiring_probability':  0.10,   # Availability, response rate, notice period
    'behavioral_fit':      0.10,   # Platform engagement, profile completeness
}

# FAISS retrieval settings
FAISS_TOP_K = 300         # Retrieve top 300 from FAISS
RERANK_TOP_K = 150        # Score top 150 with multi-dimensional scoring
SUBMISSION_TOP_K = 100    # Final output: top 100 candidates
LLM_RERANK_TOP_K = 100   # LLM re-ranks the top 100

# The raw job description text to be parsed by the LLM
RAW_JOB_DESCRIPTION = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Experience Required: 5–9 years

We need someone who is simultaneously comfortable with two things:
1. Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
2. Scrappy product-engineering attitude — willing to ship a working ranker in a week.

What you'd actually be doing: Own the intelligence layer of Redrob's product — ranking, retrieval, and matching systems.

Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
- Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
- Strong Python. Yes really, we care about code quality.
- Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

Things we'd like you to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space

Things we explicitly do NOT want:
- Title-chasers switching companies every 1.5 years
- Framework enthusiasts whose GitHub is full of LangChain tutorials
- People who have ONLY worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in their entire career
- People whose primary expertise is computer vision, speech, or robotics WITHOUT significant NLP/IR exposure
- People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation

The ideal candidate: 6-8 years total experience, 4-5 in applied ML/AI at product companies. Has shipped at least one end-to-end ranking, search, or recommendation system. Located in or willing to relocate to Noida or Pune. Active on Redrob platform.
"""
