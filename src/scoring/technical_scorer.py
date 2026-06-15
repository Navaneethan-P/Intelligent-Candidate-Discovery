import numpy as np

class TechnicalScorer:
    def __init__(self, jd_dimension_embeddings):
        """
        jd_dimension_embeddings: dict mapping dimension name to its embedding
        """
        self.jd_embeddings = jd_dimension_embeddings
        
    def score_candidate(self, candidate, candidate_chunks_metadata, candidate_chunk_embeddings):
        """
        Scores the technical fit of the candidate based on multi-dimensional semantic matching.
        """
        if len(candidate_chunk_embeddings) == 0:
            return 0.0, {}
            
        # Normalize chunk embeddings
        chunk_embeddings = np.array(candidate_chunk_embeddings)
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        chunk_embeddings = chunk_embeddings / norms
        
        dimension_scores = {}
        best_matches = {}
        
        for dim, jd_emb in self.jd_embeddings.items():
            # Normalize jd_emb
            jd_emb_norm = jd_emb / np.linalg.norm(jd_emb)
            
            # Compute cosine similarities for all chunks against this dimension
            similarities = np.dot(chunk_embeddings, jd_emb_norm)
            
            # Use max similarity across chunks for this dimension
            max_idx = np.argmax(similarities)
            max_sim = float(similarities[max_idx])
            
            # Rescale similarity to be somewhat positive
            score = max(0, (max_sim + 1) / 2)
            dimension_scores[dim] = score
            best_matches[dim] = candidate_chunks_metadata[max_idx]
            
        # Overall technical score is a weighted average of dimension scores
        # We value 'role_core' and 'infrastructure' highly
        weights = {
            "role_core": 0.4,
            "infrastructure": 0.3,
            "evaluation": 0.2,
            "engineering": 0.1
        }
        
        overall_score = sum(dimension_scores[dim] * weights.get(dim, 0) for dim in dimension_scores)
        
        # Add explicit keyword bonus for hard requirements (to ground the semantic search)
        core_skills = ['python', 'embeddings', 'retrieval', 'llm', 'fine-tuning', 'vector database', 'faiss', 'pinecone', 'milvus', 'qdrant', 'ranking', 'rag']
        skill_bonus = 0
        for skill in candidate.get('skills', []):
            s_name = skill.get('name', '').lower()
            if any(c in s_name for c in core_skills):
                skill_bonus += 0.05
                
        final_score = min(1.0, overall_score + skill_bonus)
        
        return final_score, best_matches
