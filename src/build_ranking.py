import os
import json
import time
import pandas as pd

from config import CANDIDATES_FILE, SUBMISSION_FILE, EXPLAINABILITY_LOGS, EMBEDDING_MODEL_NAME, JD_DIMENSIONS, WEIGHTS
from pipeline.data_loader import load_candidates, extract_candidate_chunks
from pipeline.embedding_engine import EmbeddingEngine
from pipeline.retrieval_faiss import FAISSRetriever
from scoring.technical_scorer import TechnicalScorer
from scoring.seniority_scorer import SeniorityScorer
from scoring.behavioral_scorer import BehavioralScorer
from pipeline.explainer import Explainer

def main():
    print("Initializing Modular Hybrid Semantic Retrieval Engine...")
    start_time = time.time()
    
    # 1. Initialize Components
    engine = EmbeddingEngine(model_name=EMBEDDING_MODEL_NAME)
    
    # Pre-compute JD dimension embeddings
    print("Embedding Job Description Dimensions...")
    jd_embeddings = {}
    for dim, text in JD_DIMENSIONS.items():
        jd_embeddings[dim] = engine.encode([text])[0]
        
    tech_scorer = TechnicalScorer(jd_embeddings)
    seniority_scorer = SeniorityScorer()
    behavioral_scorer = BehavioralScorer()
    explainer = Explainer()
    
    # 2. Process Candidates and Score
    results = []
    logs = {}
    
    print(f"Loading and processing candidates from {CANDIDATES_FILE}...")
    batch_size = 32
    
    for batch in load_candidates(CANDIDATES_FILE, batch_size=batch_size):
        for candidate in batch:
            c_id = candidate['candidate_id']
            chunks = extract_candidate_chunks(candidate)
            
            # Embed candidate chunks
            chunk_texts = [c['text'] for c in chunks]
            chunk_embeddings = engine.encode(chunk_texts)
            
            # Semantic Scoring
            tech_fit, best_matches = tech_scorer.score_candidate(candidate, chunks, chunk_embeddings)
            
            # If technical fit is too low, we can early exit to save computation, but we'll compute full for explainability
            if tech_fit < 0.2:
                continue
                
            senior_fit = seniority_scorer.score_seniority(candidate)
            founding_fit = seniority_scorer.score_founding_fit(candidate)
            evidence_str = seniority_scorer.score_evidence(candidate)
            
            # Behavioral Scoring
            hiring_prob = behavioral_scorer.score_hiring_probability(candidate)
            behav_fit = behavioral_scorer.score_behavioral_fit(candidate)
            
            # Final Weighted Score
            base_score = (
                (WEIGHTS['technical_fit'] * tech_fit) +
                (WEIGHTS['seniority_fit'] * senior_fit) +
                (WEIGHTS['founding_fit'] * founding_fit) +
                (WEIGHTS['hiring_probability'] * hiring_prob) +
                (WEIGHTS['behavioral_fit'] * behav_fit)
            )
            final_score = base_score + (WEIGHTS['evidence_strength'] * evidence_str)
            
            conf_score = int(((tech_fit + senior_fit + evidence_str + hiring_prob) / 4.0) * 100)
            
            scores_dict = {
                'technical_fit': float(tech_fit),
                'seniority_fit': float(senior_fit),
                'founding_fit': float(founding_fit),
                'hiring_probability': float(hiring_prob),
                'behavioral_fit': float(behav_fit),
                'evidence_strength': float(evidence_str),
                'confidence': conf_score
            }
            
            results.append((final_score, c_id, scores_dict, candidate, best_matches))
            
    print(f"Finished processing. Total valid candidates: {len(results)}")
    
    # 3. Sort and select top 100
    results.sort(key=lambda x: (-x[0], x[1]))
    top_100 = results[:100]
    
    # 4. Generate Explainability and Export
    print("Generating explainability logs and submission CSV...")
    out_csv = ["candidate_id,rank,score,reasoning"]
    
    for rank, (score, c_id, s_dict, c_data, best_matches) in enumerate(top_100, 1):
        formatted_score = f"{score:.4f}"
        reasoning = explainer.generate_reasoning(rank, s_dict, c_data, best_matches)
        
        # Escape quotes for CSV
        reasoning_escaped = reasoning.replace('"', '""')
        out_csv.append(f'{c_id},{rank},{formatted_score},"{reasoning_escaped}"')
        
        logs[c_id] = {
            'rank': rank,
            'total_score': float(score),
            'scores': s_dict,
            'factual_evidence': best_matches
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
