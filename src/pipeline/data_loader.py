import json

from config import MAX_CANDIDATES

def load_candidates(filepath, batch_size=64):
    """
    Generator that yields batches of candidates from a JSONL file.
    No keyword-based pre-filtering — semantic scoring via FAISS handles relevance.
    """
    batch = []
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if MAX_CANDIDATES and count >= MAX_CANDIDATES:
                break
            if not line.strip():
                continue
            count += 1
            if count % 1000 == 0:
                print(f"Loaded {count} candidates...")

            c = json.loads(line)
            batch.append(c)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

def extract_candidate_chunks(candidate):
    """
    Extract meaningful semantic chunks from the candidate profile.
    Separates summary, skills, each career role, education, and certifications
    into distinct chunks for fine-grained multi-dimensional matching.
    """
    chunks = []
    
    # Profile summary — the candidate's own narrative
    summary = candidate.get('profile', {}).get('summary', '')
    if summary:
        chunks.append({"type": "summary", "text": summary})
    
    # Headline — condensed professional identity
    headline = candidate.get('profile', {}).get('headline', '')
    if headline:
        chunks.append({"type": "headline", "text": headline})
        
    # Skills — joined as a semantic block
    skills = candidate.get('skills', [])
    skills_text = ", ".join([s.get('name', '') for s in skills])
    if skills_text:
        chunks.append({"type": "skills", "text": skills_text})
        
    # Career history — each role is an independent chunk
    for idx, role in enumerate(candidate.get('career_history', [])):
        title = role.get('title', '')
        company = role.get('company', '')
        desc = role.get('description', '')
        industry = role.get('industry', '')
        duration = role.get('duration_months', 0)
        
        text = f"{title} at {company}"
        if industry:
            text += f" ({industry})"
        if duration:
            text += f", {duration} months"
        if desc:
            text += f". {desc}"
        
        if text.strip():
            chunks.append({"type": f"role_{idx}", "text": text})
    
    # Education — each degree is a chunk for semantic matching
    for idx, edu in enumerate(candidate.get('education', [])):
        degree = edu.get('degree', '')
        field = edu.get('field_of_study', '')
        institution = edu.get('institution', '')
        tier = edu.get('tier', 'unknown')
        
        text = f"{degree} in {field} from {institution}"
        if tier and tier != 'unknown':
            text += f" ({tier})"
        
        if text.strip():
            chunks.append({"type": f"education_{idx}", "text": text})
    
    # Certifications — each cert is a chunk
    for idx, cert in enumerate(candidate.get('certifications', [])):
        name = cert.get('name', '')
        issuer = cert.get('issuer', '')
        year = cert.get('year', '')
        
        text = f"{name} by {issuer}"
        if year:
            text += f" ({year})"
        
        if text.strip():
            chunks.append({"type": f"cert_{idx}", "text": text})
    
    # Skill assessment scores — verified platform data as semantic text
    signals = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {})
    if assessments:
        assessment_text = ", ".join([
            f"{skill}: {score}/100" for skill, score in assessments.items()
        ])
        chunks.append({"type": "assessments", "text": f"Verified skill assessments: {assessment_text}"})
    
    return chunks
