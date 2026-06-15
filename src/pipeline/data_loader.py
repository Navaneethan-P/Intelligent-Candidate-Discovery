import json

from config import MAX_CANDIDATES

def load_candidates(filepath, batch_size=64):
    """
    Generator that yields batches of candidates.
    Filters out obvious non-matches based on current title.
    """
    irrelevant_titles = [
        'marketing', 'hr ', 'human resources', 'accountant', 'sales', 
        'finance', 'customer support', 'graphic designer', 'content writer', 
        'recruiter', 'business analyst', 'operations manager'
    ]
    
    batch = []
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if MAX_CANDIDATES and count >= MAX_CANDIDATES:
                break
            if not line.strip(): continue
            count += 1
            if count % 1000 == 0:
                print(f"Loaded {count} candidates...")
                
            c = json.loads(line)
            title = c.get('profile', {}).get('current_title', '').lower()
            if any(it in title for it in irrelevant_titles):
                continue
                
            batch.append(c)
            if len(batch) >= batch_size:
                yield batch
                batch = []
                
        if batch:
            yield batch

def extract_candidate_chunks(candidate):
    """
    Instead of one large blob of text, we extract meaningful chunks from the candidate.
    We separate 'skills', 'summary', and each 'career_history' item.
    """
    chunks = []
    
    summary = candidate.get('profile', {}).get('summary', '')
    if summary:
        chunks.append({"type": "summary", "text": summary})
        
    skills = candidate.get('skills', [])
    skills_text = ", ".join([s.get('name', '') for s in skills])
    if skills_text:
        chunks.append({"type": "skills", "text": skills_text})
        
    for idx, role in enumerate(candidate.get('career_history', [])):
        title = role.get('title', '')
        desc = role.get('description', '')
        text = f"{title}. {desc}"
        if text.strip() != ".":
            chunks.append({"type": f"role_{idx}", "text": text})
            
    return chunks
