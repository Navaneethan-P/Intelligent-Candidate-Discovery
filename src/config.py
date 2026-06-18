import os

# Base Directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = r"d:\AI Candidate Ranking System Implementation\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Files
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.jsonl")
SUBMISSION_FILE = os.path.join(OUTPUTS_DIR, "team_submission.csv")
EXPLAINABILITY_LOGS = os.path.join(OUTPUTS_DIR, "explainability_logs.json")

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MAX_CANDIDATES = 5000  # Set to None for full run

# Weights
WEIGHTS = {
    'technical_fit': 0.40,
    'seniority_fit': 0.20,
    'founding_fit': 0.15,
    'hiring_probability': 0.10,
    'behavioral_fit': 0.10,
    'evidence_strength': 0.05
}

# The job description text chunked by dimension to allow finer-grained semantic search
JD_DIMENSIONS = {
    "role_core": "Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.",
    "infrastructure": "Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.",
    "evaluation": "Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.",
    "engineering": "Strong Python. Background in distributed systems or large-scale inference optimization. Built scalable ranking, search, or recommendation system to real users at meaningful scale."
}

# The raw job description text to be parsed by the LLM
RAW_JOB_DESCRIPTION = """
We are looking for a Senior AI/ML Engineer with deep technical depth in modern ML systems, specifically around embeddings, retrieval, ranking, LLMs, and fine-tuning.
You should have production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
Strong experience with vector databases or hybrid search infrastructure is required (Pinecone, Weaviate, Qdrant, Milvus, FAISS).
You must have hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B testing).
We need strong Python skills and a background in distributed systems or large-scale inference optimization.
Ideally, you have led a team of engineers, architected systems from scratch, and built scalable ranking or search systems for real users at a meaningful scale.
Experience in 0 to 1 startup environments and the ability to wear multiple hats is a huge plus.
"""
