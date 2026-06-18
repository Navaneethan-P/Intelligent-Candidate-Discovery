import numpy as np

class TechnicalScorer:
    def __init__(self, jd_dimension_embeddings, engine=None):
        """
        jd_dimension_embeddings: dict mapping dimension name to its embedding
        engine: EmbeddingEngine instance for semantic skill scoring
        """
        self.jd_embeddings = jd_dimension_embeddings
        self.engine = engine

        # Pre-compute a "hard technical skills" concept embedding for bonus scoring
        self.core_skills_concept = (
            "Python machine learning embeddings vector retrieval LLM fine-tuning "
            "FAISS Pinecone Milvus Qdrant Weaviate RAG ranking recommendation systems "
            "transformer models semantic search production inference"
        )
        # Pre-compute a "target titles/roles" concept embedding to detect profile relevance and penalize unrelated roles
        self.target_titles_concept = (
            "Senior AI ML Engineer Machine Learning Data Scientist Computer Vision "
            "Backend Software Developer Architect Principal Staff Lead Head Director"
        )
        if self.engine:
            self.core_skills_embedding = self.engine.encode([self.core_skills_concept])[0]
            norm = np.linalg.norm(self.core_skills_embedding)
            if norm > 0:
                self.core_skills_embedding = self.core_skills_embedding / norm

            self.target_titles_embedding = self.engine.encode([self.target_titles_concept])[0]
            norm_titles = np.linalg.norm(self.target_titles_embedding)
            if norm_titles > 0:
                self.target_titles_embedding = self.target_titles_embedding / norm_titles

    def _cosine_sim(self, v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def score_candidate(self, candidate, candidate_chunks_metadata, candidate_chunk_embeddings):
        """
        Scores the technical fit of the candidate based on multi-dimensional semantic matching.
        Separates skills-based matching from actual career experience to prevent keyword-stuffing hacks.
        All scoring is purely semantic — no keyword lists.
        """
        if len(candidate_chunk_embeddings) == 0:
            return 0.0, {}

        # Normalize chunk embeddings
        chunk_embeddings = np.array(candidate_chunk_embeddings)
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_embeddings = chunk_embeddings / norms

        # Identify chunk types
        exp_indices = []
        skills_idx = None
        for idx, metadata in enumerate(candidate_chunks_metadata):
            if metadata.get('type') == 'skills':
                skills_idx = idx
            else:
                exp_indices.append(idx)

        dimension_scores = {}
        best_matches = {}

        for dim, jd_emb in self.jd_embeddings.items():
            # Normalize jd_emb
            jd_emb_norm = jd_emb / (np.linalg.norm(jd_emb) or 1)

            # Compute cosine similarities for all chunks against this dimension
            similarities = np.dot(chunk_embeddings, jd_emb_norm)

            skills_sim = float(similarities[skills_idx]) if skills_idx is not None else 0.0
            exp_sims = [float(similarities[i]) for i in exp_indices]
            exp_sim = max(exp_sims) if exp_sims else 0.0

            # Combine experience match and skills match: experience holds 70% weight to prevent fake/irrelevant skill matching
            if exp_sims and skills_idx is not None:
                combined_sim = 0.7 * exp_sim + 0.3 * max(exp_sim, skills_sim)
                best_idx = exp_indices[np.argmax(exp_sims)]
            elif exp_sims:
                combined_sim = exp_sim
                best_idx = exp_indices[np.argmax(exp_sims)]
            else:
                combined_sim = 0.3 * skills_sim
                best_idx = skills_idx

            # Rescale similarity to be somewhat positive
            score = max(0.0, (combined_sim + 1) / 2)
            dimension_scores[dim] = score
            best_matches[dim] = candidate_chunks_metadata[best_idx]

        # Overall technical score is a weighted average of dimension scores
        weights = {
            "role_core": 0.40,
            "infrastructure": 0.30,
            "evaluation": 0.20,
            "engineering": 0.10
        }

        overall_score = sum(
            dimension_scores.get(dim, 0) * w for dim, w in weights.items()
        )

        # Modulate score based on profile headline and current title relevance to target titles
        title_text = f"{candidate.get('profile', {}).get('current_title', '')} {candidate.get('profile', {}).get('headline', '')}"
        title_factor = 1.0
        if self.engine and hasattr(self, 'target_titles_embedding') and title_text.strip():
            title_emb = self.engine.encode([title_text])[0]
            title_sim = self._cosine_sim(self.target_titles_embedding, title_emb)
            if title_sim >= 0.50:
                title_factor = 1.0
            elif title_sim < 0.35:
                title_factor = 0.0
            else:
                title_factor = (title_sim - 0.35) / 0.15

        # Modulate the overall technical score
        modulated_score = overall_score * (0.5 + 0.5 * title_factor)

        # Semantic skill bonus: compare candidate skill chunks against core skills concept
        # This is fully semantic — no keyword lists
        skill_bonus = 0.0
        if self.engine and hasattr(self, 'core_skills_embedding'):
            skill_sims = [
                self._cosine_sim(self.core_skills_embedding, chunk_embeddings[i])
                for i in range(len(chunk_embeddings))
            ]
            # Top skill similarity becomes a bonus capped at 0.15
            skill_bonus = min(0.15, max(skill_sims) * 0.2)

        final_score = min(1.0, modulated_score + skill_bonus)
        return final_score, best_matches
