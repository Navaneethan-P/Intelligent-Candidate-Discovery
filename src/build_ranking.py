import os
import json
import time
import pandas as pd
import numpy as np
from collections import defaultdict

from config import CANDIDATES_FILE, SUBMISSION_FILE, EXPLAINABILITY_LOGS, EMBEDDING_MODEL_NAME, WEIGHTS, RAW_JOB_DESCRIPTION
from pipeline.data_loader import load_candidates, extract_candidate_chunks
from pipeline.embedding_engine import EmbeddingEngine
from pipeline.retrieval_faiss import FAISSRetriever
from scoring.technical_scorer import TechnicalScorer
from scoring.seniority_scorer import SeniorityScorer
from scoring.behavioral_scorer import BehavioralScorer
from pipeline.explainer import Explainer
from pipeline.jd_parser import JDParser
from pipeline.llm_reranker import LLMReranker

def main():
    print("Initializing Agentic Hybrid Semantic Retrieval Engine...")
    start_time = time.time()
    
    # 1. Initialize Components & Parse JD
    print("Parsing Job Description via LLM...")
    jd_parser = JDParser()
    JD_DIMENSIONS = jd_parser.parse_jd(RAW_JOB_DESCRIPTION)
    print("Parsed JD Dimensions:", list(JD_DIMENSIONS.keys()))
    
    engine = EmbeddingEngine(model_name=EMBEDDING_MODEL_NAME)
    faiss_retriever = FAISSRetriever()
    llm_reranker = LLMReranker()
    
    print("Embedding Job Description Dimensions...")
    jd_embeddings = {}
    for dim, text in JD_DIMENSIONS.items():
        jd_embeddings[dim] = engine.encode([text])[0]
        
    tech_scorer = TechnicalScorer(jd_embeddings, engine=engine)
    seniority_scorer = SeniorityScorer(engine)
    behavioral_scorer = BehavioralScorer()
    explainer = Explainer()
    
    # 2. Stage 1: Fast Retrieval via FAISS
    print(f"Loading candidates from {CANDIDATES_FILE} and indexing in FAISS...")
    candidates_store = {}
    candidate_chunks_store = {}
    candidate_chunk_embeddings_store = {}
    
    batch_size = 32
    for batch in load_candidates(CANDIDATES_FILE, batch_size=batch_size):
        for candidate in batch:
            c_id = candidate['candidate_id']
            candidates_store[c_id] = candidate
            
            chunks = extract_candidate_chunks(candidate)
            if not chunks:
                continue
                
            chunk_texts = [c['text'] for c in chunks]
            chunk_embeddings = engine.encode(chunk_texts)
            
            candidate_chunks_store[c_id] = chunks
            candidate_chunk_embeddings_store[c_id] = chunk_embeddings
            
            faiss_retriever.add_chunks(c_id, chunks, chunk_embeddings)
            
    # Query FAISS to narrow down to top 200 candidates
    print("Querying FAISS to retrieve top 200 candidates based on JD dimensions...")
    candidate_hits = defaultdict(float)
    for dim, q_emb in jd_embeddings.items():
        results = faiss_retriever.search(q_emb, k=200)
        for res in results:
            # Accumulate scores for candidates across all dimensions
            candidate_hits[res['candidate_id']] += res['score']
            
    # Get top 200 candidate IDs
    top_200_ids = sorted(candidate_hits.keys(), key=lambda k: candidate_hits[k], reverse=True)[:200]
    print(f"Retrieved {len(top_200_ids)} candidates from FAISS.")
    
    # 3. Base Scoring for Top 200
    print("Computing base semantic & behavioral scores for Top 200...")
    scored_results = []
    
    for c_id in top_200_ids:
        candidate = candidates_store[c_id]
        chunks = candidate_chunks_store[c_id]
        chunk_embeddings = candidate_chunk_embeddings_store[c_id]
        
        tech_fit, best_matches = tech_scorer.score_candidate(candidate, chunks, chunk_embeddings)
        senior_fit = seniority_scorer.score_seniority(candidate, chunk_embeddings)
        founding_fit = seniority_scorer.score_founding_fit(candidate, chunk_embeddings)
        evidence_str = seniority_scorer.score_evidence(candidate)
        hiring_prob = behavioral_scorer.score_hiring_probability(candidate)
        behav_fit = behavioral_scorer.score_behavioral_fit(candidate)
        
        base_score = (
            (WEIGHTS['technical_fit'] * tech_fit) +
            (WEIGHTS['seniority_fit'] * senior_fit) +
            (WEIGHTS['founding_fit'] * founding_fit) +
            (WEIGHTS['hiring_probability'] * hiring_prob) +
            (WEIGHTS['behavioral_fit'] * behav_fit)
        ) + (WEIGHTS['evidence_strength'] * evidence_str)
        
        scores_dict = {
            'technical_fit': float(tech_fit),
            'seniority_fit': float(senior_fit),
            'founding_fit': float(founding_fit),
            'hiring_probability': float(hiring_prob),
            'behavioral_fit': float(behav_fit),
            'evidence_strength': float(evidence_str),
            'confidence': int(((tech_fit + senior_fit + evidence_str + hiring_prob) / 4.0) * 100)
        }
        
        scored_results.append((base_score, c_id, scores_dict, candidate, best_matches))
        
    # Sort and take top 50 for LLM Re-ranking
    scored_results.sort(key=lambda x: (-x[0], x[1]))
    top_50 = scored_results[:50]
    
    # 4. Stage 2: LLM Re-Ranking
    print("Stage 2: LLM Re-Ranking Top 50 candidates...")
    final_results = []
    logs = {}
    
    for rank, (base_score, c_id, s_dict, c_data, best_matches) in enumerate(top_50, 1):
        print(f"Re-ranking {rank}/50: {c_id}")
        llm_score, llm_reasoning = llm_reranker.rerank_candidate(c_data, RAW_JOB_DESCRIPTION, s_dict)
        
        # Combine LLM score with base score for final rank, or just use LLM score. 
        # The prompt says "Ask the LLM to act as the ultimate judge, outputting a final score". We will use LLM score as the primary, but mix in base behavioral metrics.
        # Let's normalize LLM score to 0-1
        normalized_llm_score = llm_score / 100.0 if isinstance(llm_score, (int, float)) else 0.5
        
        # The true "AI Agent" score
        final_score = (normalized_llm_score * 0.7) + (base_score * 0.3)
        
        final_results.append((final_score, c_id, s_dict, c_data, best_matches, llm_reasoning))

    # Re-sort based on final score
    final_results.sort(key=lambda x: (-x[0], x[1]))
    
    # 5. Generate Output
    print("Generating submission CSV and Explainability Logs...")
    out_csv = ["candidate_id,rank,score,reasoning"]
    
    for rank, (score, c_id, s_dict, c_data, best_matches, reasoning) in enumerate(final_results, 1):
        formatted_score = f"{score:.4f}"
        # Sanitize: strip newlines and escape quotes for valid CSV
        reasoning_clean = reasoning.replace('\r', ' ').replace('\n', ' ').replace('"', '""')
        out_csv.append(f'{c_id},{rank},{formatted_score},"{reasoning_clean}"')
        
        logs[c_id] = {
            'rank': rank,
            'total_score': float(score),
            'scores': s_dict,
            'factual_evidence': best_matches,
            'llm_reasoning': reasoning
        }
        
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_csv) + "\n")
        
    with open(EXPLAINABILITY_LOGS, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)
        
    end_time = time.time()
    print(f"Successfully saved {SUBMISSION_FILE} and {EXPLAINABILITY_LOGS}.")
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
