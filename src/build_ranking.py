import json
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import sys

JD_TEXT = """
Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python.
Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
LLM fine-tuning experience (LoRA, QLoRA, PEFT)
Experience with learning-to-rank models (XGBoost-based or neural)
Background in distributed systems or large-scale inference optimization
Open-source contributions in the AI/ML space.
Built scalable ranking, search, or recommendation system to real users at meaningful scale.
"""

def get_candidate_text(candidate):
    text_parts = [
        candidate['profile'].get('summary', ''),
        candidate['profile'].get('headline', '')
    ]
    for role in candidate.get('career_history', []):
        text_parts.append(role.get('title', ''))
        text_parts.append(role.get('description', ''))
    return " ".join(text_parts)

def get_technical_fit_score(candidate, candidate_emb, jd_embedding):
    cos_sim = np.dot(candidate_emb, jd_embedding) / (np.linalg.norm(candidate_emb) * np.linalg.norm(jd_embedding))
    cos_sim = max(0, (cos_sim + 1) / 2)
    
    core_skills = ['python', 'embeddings', 'retrieval', 'llm', 'fine-tuning', 'vector database', 'faiss', 'pinecone', 'milvus', 'qdrant', 'ranking', 'search', 'recommendation', 'rag']
    skill_score = 0
    for skill in candidate.get('skills', []):
        s_name = skill.get('name', '').lower()
        if any(c in s_name for c in core_skills):
            skill_score += (skill.get('endorsements', 0) * 0.01) + (skill.get('duration_months', 0) * 0.01)
            
    skill_score = min(1.0, skill_score / 5.0) 
    
    return (0.7 * cos_sim) + (0.3 * skill_score)

def get_seniority_fit_score(candidate):
    text_parts = []
    for role in candidate.get('career_history', []):
        text_parts.append(role.get('description', '').lower())
        text_parts.append(role.get('title', '').lower())
    full_text = " ".join(text_parts)
    
    score = 0.0
    if re.search(r'\b(led|architected|designed|owned|built from scratch|founded|managed team|spearheaded|drove)\b', full_text):
        score += 0.4
    if re.search(r'\b(senior|staff|principal|lead|head|director|founder|architect)\b', full_text):
        score += 0.3
    if re.search(r'\b(scalable|production|deployed|infrastructure)\b', full_text):
        score += 0.3
        
    return min(1.0, score)

def get_founding_fit_score(candidate):
    text_parts = []
    for role in candidate.get('career_history', []):
        text_parts.append(role.get('description', '').lower())
    full_text = " ".join(text_parts)
    
    score = 0.0
    if re.search(r'\b(0 to 1|zero to one|startup|early stage|founding team|multiple hats|wear many hats|fast-paced|mvp|prototype|open source|contributor)\b', full_text):
        score += 0.5
    if re.search(r'\b(full stack|frontend|backend|devops|ci/cd|infrastructure|deployment)\b', full_text):
        score += 0.3
        
    github = candidate.get('redrob_signals', {}).get('github_activity_score', -1)
    if github > 50:
        score += 0.2
    elif github > 0:
        score += 0.1
        
    service_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree']
    company_names = [role.get('company', '').lower() for role in candidate.get('career_history', [])]
    all_service = len(company_names) > 0 and all(any(sc in c for sc in service_companies) for c in company_names)
    if all_service:
        score -= 0.5
    else:
        score += 0.2
        
    return max(0.0, min(1.0, score))

def get_hiring_probability_score(candidate):
    signals = candidate.get('redrob_signals', {})
    score = 0.0
    
    score += signals.get('recruiter_response_rate', 0) * 0.4
    
    notice = signals.get('notice_period_days', 90)
    if notice <= 30:
        score += 0.3
    elif notice <= 60:
        score += 0.15
        
    if signals.get('open_to_work_flag', False):
        score += 0.15
        
    location = candidate['profile'].get('location', '').lower()
    if any(loc in location for loc in ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr']):
        score += 0.15
    elif signals.get('willing_to_relocate', False):
        score += 0.15
        
    return min(1.0, score)

def get_behavioral_fit_score(candidate):
    signals = candidate.get('redrob_signals', {})
    score = 0.0
    
    completeness = signals.get('profile_completeness_score', 0)
    score += (completeness / 100.0) * 0.4
    
    score += signals.get('interview_completion_rate', 0) * 0.3
    
    last_active = signals.get('last_active_date', '2020-01-01')
    try:
        last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
        reference_dt = datetime(2026, 6, 12)
        days_inactive = (reference_dt - last_active_dt).days
        if days_inactive < 30:
            score += 0.3
        elif days_inactive < 90:
            score += 0.15
        elif days_inactive > 180:
            score -= 0.5
    except:
        pass
        
    return max(0.0, min(1.0, score))

def get_evidence_strength_score(candidate):
    text_parts = []
    for role in candidate.get('career_history', []):
        text_parts.append(role.get('description', '').lower())
    full_text = " ".join(text_parts)
    
    metrics = re.findall(r'\b(?:[0-9]+(?:\.[0-9]+)?(?:%|x|m|k|b|tb|gb)|[0-9]+(?:,[0-9]+)+|\$[0-9]+[mkb]?)\b', full_text)
    scale_indicators = re.findall(r'\b(?:latency|requests/sec|rps|qps|dau|mau|users|million|billion)\b', full_text)
    
    total_evidence = len(metrics) + len(scale_indicators)
    score = min(1.0, total_evidence / 4.0)
    return score

def generate_reasoning(c_id, rank, scores, candidate, conf_score):
    tech = scores['technical_fit']
    senior = scores['seniority_fit']
    founding = scores['founding_fit']
    hiring = scores['hiring_probability']
    evidence = scores['evidence_strength']
    
    yoe = candidate['profile'].get('years_of_experience', 0)
    title = candidate['profile'].get('current_title', 'Engineer')
    
    if conf_score >= 80:
        conf_str = "95% Confidence"
    elif conf_score >= 60:
        conf_str = "80% Confidence"
    else:
        conf_str = "50% Confidence"
        
    metrics = []
    if tech > 0.8:
        metrics.append("exceptional technical match for ML/Retrieval")
    elif tech > 0.6:
        metrics.append("strong technical foundation")
        
    if senior > 0.8:
        metrics.append("proven scale/architecture ownership")
        
    if founding > 0.7:
        metrics.append("strong startup/0-to-1 DNA")
        
    if evidence > 0.6:
        metrics.append("quantifiable impact metrics")
        
    if hiring > 0.8:
        metrics.append("highly responsive and immediately available")
        
    reason_parts = [f"Ranked #{rank} ({conf_str}):", f"{title} with {yoe} yrs exp."]
    if metrics:
        reason_parts.append("Demonstrates " + ", ".join(metrics) + ".")
    else:
        reason_parts.append("Solid overall fit.")
        
    return " ".join(reason_parts)

def process_batch(batch, model, jd_emb, results):
    if not batch:
        return
        
    texts = [get_candidate_text(c) for c in batch]
    embeddings = model.encode(texts) # batched encoding
    
    for i, c in enumerate(batch):
        candidate_emb = embeddings[i]
        tech = get_technical_fit_score(c, candidate_emb, jd_emb)
        if tech < 0.3:
            continue
            
        senior = get_seniority_fit_score(c)
        founding = get_founding_fit_score(c)
        hiring = get_hiring_probability_score(c)
        behavior = get_behavioral_fit_score(c)
        evidence = get_evidence_strength_score(c)
        
        base_score = (0.40 * tech) + (0.20 * senior) + (0.20 * founding) + (0.10 * hiring) + (0.10 * behavior)
        final_score = base_score + (0.10 * evidence)
        conf_score = int(((tech + senior + evidence + hiring) / 4.0) * 100)
        
        scores_dict = {
            'technical_fit': tech,
            'seniority_fit': senior,
            'founding_fit': founding,
            'hiring_probability': hiring,
            'behavioral_fit': behavior,
            'evidence_strength': evidence,
            'confidence': conf_score
        }
        
        results.append((final_score, c['candidate_id'], scores_dict, c))

def main():
    print("Loading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_emb = model.encode(JD_TEXT)
    
    import os
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'challenge_data', 'India_runs_data_and_ai_challenge')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    
    candidates_file = os.path.join(DATA_DIR, "candidates.jsonl")
    print(f"Processing candidates from {candidates_file}...")
    
    results = []
    irrelevant_titles = ['marketing', 'hr ', 'human resources', 'accountant', 'sales', 'finance', 'customer support', 'graphic designer', 'content writer', 'recruiter', 'business analyst', 'operations manager']
    
    batch = []
    batch_size = 64
    count = 0
    
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} candidates...")
                
            c = json.loads(line)
            title = c['profile'].get('current_title', '').lower()
            if any(it in title for it in irrelevant_titles):
                continue
                
            batch.append(c)
            if len(batch) >= batch_size:
                process_batch(batch, model, jd_emb, results)
                batch = []
                
        if batch:
            process_batch(batch, model, jd_emb, results)
            
    print(f"Finished processing. Total valid candidates: {len(results)}")
    results.sort(key=lambda x: (-x[0], x[1]))
    top_100 = results[:100]
    
    out_csv = []
    out_csv.append("candidate_id,rank,score,reasoning")
    logs = {}
    
    for rank, (score, c_id, s_dict, c_data) in enumerate(top_100, 1):
        formatted_score = f"{score:.4f}"
        reasoning = generate_reasoning(c_id, rank, s_dict, c_data, s_dict['confidence'])
        reasoning = reasoning.replace('"', '""')
        out_csv.append(f'{c_id},{rank},{formatted_score},"{reasoning}"')
        
        logs[c_id] = {
            'rank': rank,
            'total_score': score,
            'scores': s_dict
        }
        
    out_csv_path = os.path.join(OUTPUTS_DIR, 'team_submission.csv')
    with open(out_csv_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_csv) + "\n")
        
    out_json_path = os.path.join(OUTPUTS_DIR, 'explainability_logs.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        for cid in logs:
            for k in logs[cid]['scores']:
                logs[cid]['scores'][k] = float(logs[cid]['scores'][k])
        json.dump(logs, f, indent=2)
        
    print(f"Saved {out_csv_path} and {out_json_path}.")

if __name__ == "__main__":
    main()
