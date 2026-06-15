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
# We are using BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

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
