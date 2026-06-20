import os
import json
import time
import pandas as pd
import numpy as np
from collections import defaultdict

from config import (
    CANDIDATES_FILE, SUBMISSION_FILE, EXPLAINABILITY_LOGS, 
    EMBEDDING_MODEL_NAME, WEIGHTS, RAW_JOB_DESCRIPTION,
    FAISS_TOP_K, RERANK_TOP_K, SUBMISSION_TOP_K, LLM_RERANK_TOP_K
)
from pipeline.data_loader import load_candidates, extract_candidate_chunks
from pipeline.embedding_engine import EmbeddingEngine
from pipeline.retrieval_faiss import FAISSRetriever
from scoring.technical_scorer import TechnicalScorer
from scoring.seniority_scorer import SeniorityScorer
from scoring.behavioral_scorer import BehavioralScorer
from scoring.signal_scorer import SignalScorer
from scoring.education_scorer import EducationScorer
from pipeline.explainer import Explainer
from pipeline.jd_parser import JDParser
from pipeline.llm_reranker import LLMReranker


def is_relevant_title(title, engine, target_embedding, threshold=0.25):
    """
    Semantic title relevance check. Returns True if the candidate's title
    is semantically close enough to engineering/tech/ML roles.
    Prevents HR Managers, Sales Executives, etc. from being ranked.
    """
    if not title or not engine:
        return True  # Don't filter if no data
    
    title_emb = engine.encode([title])[0]
    norm1 = np.linalg.norm(title_emb)
    norm2 = np.linalg.norm(target_embedding)
    if norm1 == 0 or norm2 == 0:
        return True
    sim = float(np.dot(title_emb, target_embedding) / (norm1 * norm2))
    return sim >= threshold


def main():
    print("=" * 70)
    print("  Intelligent Candidate Discovery — Agentic Hybrid Ranking Engine")
    print("=" * 70)
    start_time = time.time()
    
    # 1. Initialize Components & Parse JD
    print("\n[Stage 0] Parsing Job Description via LLM...")
    jd_parser = JDParser()
    JD_DIMENSIONS = jd_parser.parse_jd(RAW_JOB_DESCRIPTION)
    print(f"  Parsed JD Dimensions: {list(JD_DIMENSIONS.keys())}")
    
    engine = EmbeddingEngine(model_name=EMBEDDING_MODEL_NAME)
    faiss_retriever = FAISSRetriever()
    llm_reranker = LLMReranker()
    
    print("  Embedding Job Description Dimensions...")
    jd_embeddings = {}
    for dim, text in JD_DIMENSIONS.items():
        jd_embeddings[dim] = engine.encode([text])[0]
    
    # Pre-compute the target role concept embedding for title filtering
    target_role_concept = (
        "Software Engineer Machine Learning AI Data Scientist Backend Developer "
        "ML Engineer AI Engineer NLP Engineer Research Engineer Platform Engineer "
        "Senior Staff Principal Lead Architect DevOps SRE Infrastructure Engineer "
        "Full Stack Developer Computer Science Technology"
    )
    target_role_embedding = engine.encode([target_role_concept])[0]
    norm = np.linalg.norm(target_role_embedding)
    if norm > 0:
        target_role_embedding = target_role_embedding / norm
        
    tech_scorer = TechnicalScorer(jd_embeddings, engine=engine)
    seniority_scorer = SeniorityScorer(engine)
    behavioral_scorer = BehavioralScorer()
    signal_scorer = SignalScorer()
    education_scorer = EducationScorer(engine=engine)
    explainer = Explainer()
    
    # 2. Stage 1: Load candidates and build FAISS index
    print(f"\n[Stage 1] Loading candidates from {CANDIDATES_FILE}...")
    candidates_store = {}
    candidate_chunks_store = {}
    candidate_chunk_embeddings_store = {}
    
    total_loaded = 0
    total_filtered_out = 0
    
    batch_size = 32
    for batch in load_candidates(CANDIDATES_FILE, batch_size=batch_size):
        for candidate in batch:
            c_id = candidate['candidate_id']
            total_loaded += 1
            
            # Semantic title relevance filter — drop clearly irrelevant roles
            title = candidate.get('profile', {}).get('current_title', '')
            if not is_relevant_title(title, engine, target_role_embedding, threshold=0.22):
                total_filtered_out += 1
                continue
            
            candidates_store[c_id] = candidate
            
            chunks = extract_candidate_chunks(candidate)
            if not chunks:
                continue
                
            chunk_texts = [c['text'] for c in chunks]
            chunk_embeddings = engine.encode(chunk_texts)
            
            candidate_chunks_store[c_id] = chunks
            candidate_chunk_embeddings_store[c_id] = chunk_embeddings
            
            faiss_retriever.add_chunks(c_id, chunks, chunk_embeddings)
    
    print(f"  Loaded {total_loaded} candidates total.")
    print(f"  Filtered out {total_filtered_out} irrelevant title candidates.")
    print(f"  Indexed {len(candidates_store)} relevant candidates in FAISS.")
    
    # Query FAISS to narrow down to top candidates
    print(f"\n[Stage 2] Querying FAISS to retrieve top {FAISS_TOP_K} candidates...")
    candidate_hits = defaultdict(float)
    for dim, q_emb in jd_embeddings.items():
        results = faiss_retriever.search(q_emb, k=FAISS_TOP_K)
        for res in results:
            candidate_hits[res['candidate_id']] += res['score']
            
    # Get top N candidate IDs
    top_ids = sorted(candidate_hits.keys(), key=lambda k: candidate_hits[k], reverse=True)[:RERANK_TOP_K]
    print(f"  Retrieved {len(top_ids)} candidates from FAISS for detailed scoring.")
    
    # 3. Multi-Dimensional Scoring
    print(f"\n[Stage 3] Computing 8-dimensional scores for top {len(top_ids)} candidates...")
    scored_results = []
    
    for idx, c_id in enumerate(top_ids):
        if (idx + 1) % 50 == 0:
            print(f"  Scored {idx + 1}/{len(top_ids)}...")
        
        candidate = candidates_store[c_id]
        chunks = candidate_chunks_store.get(c_id, [])
        chunk_embeddings = candidate_chunk_embeddings_store.get(c_id, np.array([]))
        
        # Core semantic scoring
        tech_fit, best_matches = tech_scorer.score_candidate(candidate, chunks, chunk_embeddings)
        senior_fit = seniority_scorer.score_seniority(candidate, chunk_embeddings)
        founding_fit = seniority_scorer.score_founding_fit(candidate, chunk_embeddings)
        evidence_str = seniority_scorer.score_evidence(candidate, chunk_embeddings)
        
        # Behavioral scoring
        hiring_prob = behavioral_scorer.score_hiring_probability(candidate)
        behav_fit = behavioral_scorer.score_behavioral_fit(candidate)
        
        # New scorers
        signal_score, signal_breakdown = signal_scorer.compute_composite_signal_score(candidate)
        education_fit = education_scorer.compute_education_score(candidate)
        
        # Weighted composite score across all 8 dimensions
        base_score = (
            (WEIGHTS['technical_fit'] * tech_fit) +
            (WEIGHTS['seniority_fit'] * senior_fit) +
            (WEIGHTS['founding_fit'] * founding_fit) +
            (WEIGHTS['signal_score'] * signal_score) +
            (WEIGHTS['education_fit'] * education_fit) +
            (WEIGHTS['evidence_strength'] * evidence_str) +
            (WEIGHTS['hiring_probability'] * hiring_prob) +
            (WEIGHTS['behavioral_fit'] * behav_fit)
        )
        
        # Confidence = average of the strongest signal dimensions
        confidence = int(((tech_fit + senior_fit + evidence_str + signal_score) / 4.0) * 100)
        
        scores_dict = {
            'technical_fit': float(tech_fit),
            'seniority_fit': float(senior_fit),
            'founding_fit': float(founding_fit),
            'hiring_probability': float(hiring_prob),
            'behavioral_fit': float(behav_fit),
            'evidence_strength': float(evidence_str),
            'signal_score': float(signal_score),
            'education_fit': float(education_fit),
            'confidence': confidence,
        }
        
        scored_results.append((base_score, c_id, scores_dict, candidate, best_matches, signal_breakdown))
        
    # Sort and take top candidates for LLM Re-ranking
    scored_results.sort(key=lambda x: (-x[0], x[1]))
    top_for_llm = scored_results[:LLM_RERANK_TOP_K]
    
    # 4. Stage 4: LLM Re-Ranking
    print(f"\n[Stage 4] LLM Re-Ranking top {len(top_for_llm)} candidates...")
    final_results = []
    logs = {}
    
    for rank, (base_score, c_id, s_dict, c_data, best_matches, sig_breakdown) in enumerate(top_for_llm, 1):
        if rank % 10 == 0:
            print(f"  Re-ranking {rank}/{len(top_for_llm)}: {c_id}")
        
        llm_score, llm_reasoning = llm_reranker.rerank_candidate(
            c_data, RAW_JOB_DESCRIPTION, s_dict, base_score=base_score
        )
        
        # Combine LLM score with base score for the final rank
        normalized_llm_score = llm_score / 100.0 if isinstance(llm_score, (int, float)) else 0.5
        
        # 60/40 blend: LLM judgment + semantic base score
        final_score = (normalized_llm_score * 0.60) + (base_score * 0.40)
        
        final_results.append((final_score, c_id, s_dict, c_data, best_matches, llm_reasoning, sig_breakdown))

    # Re-sort based on final score
    final_results.sort(key=lambda x: (-x[0], x[1]))
    
    # Take exactly SUBMISSION_TOP_K candidates
    final_results = final_results[:SUBMISSION_TOP_K]
    
    # 5. Generate Output
    print(f"\n[Stage 5] Generating submission CSV ({SUBMISSION_TOP_K} candidates) and Explainability Logs...")
    out_csv = ["candidate_id,rank,score,reasoning"]
    
    for rank, (score, c_id, s_dict, c_data, best_matches, reasoning, sig_breakdown) in enumerate(final_results, 1):
        formatted_score = f"{score:.4f}"
        
        # Use LLM reasoning if available, otherwise generate from explainer
        if reasoning and len(reasoning) > 50:
            final_reasoning = reasoning
        else:
            final_reasoning = explainer.generate_reasoning(rank, s_dict, c_data, best_matches)
        
        # Sanitize: strip newlines and escape quotes for valid CSV
        reasoning_clean = final_reasoning.replace('\r', ' ').replace('\n', ' ').replace('"', '""')
        out_csv.append(f'{c_id},{rank},{formatted_score},"{reasoning_clean}"')
        
        logs[c_id] = {
            'rank': rank,
            'total_score': float(score),
            'scores': s_dict,
            'signal_breakdown': sig_breakdown,
            'factual_evidence': best_matches,
            'llm_reasoning': reasoning
        }
        
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_csv) + "\n")
        
    with open(EXPLAINABILITY_LOGS, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)
        
    end_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete!")
    print(f"  Candidates processed: {total_loaded}")
    print(f"  Irrelevant titles filtered: {total_filtered_out}")
    print(f"  FAISS indexed: {len(candidates_store)}")
    print(f"  Final ranked output: {len(final_results)} candidates")
    print(f"  Saved: {SUBMISSION_FILE}")
    print(f"  Saved: {EXPLAINABILITY_LOGS}")
    print(f"  Execution time: {end_time - start_time:.2f} seconds")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
