import numpy as np


class EducationScorer:
    """
    Scores candidates based on education relevance and certification alignment.
    Uses semantic embeddings for field-of-study matching against JD requirements,
    plus structured scoring for institution tier.
    """

    def __init__(self, engine=None):
        self.engine = engine

        # Concept embedding for relevant fields of study
        self.relevant_fields_concept = (
            "Computer Science Artificial Intelligence Machine Learning "
            "Data Science Information Technology Software Engineering "
            "Mathematics Statistics Electrical Engineering Electronics "
            "Computational Linguistics Natural Language Processing "
            "Information Retrieval Operations Research"
        )

        # Concept embedding for relevant certifications
        self.relevant_certs_concept = (
            "Machine Learning Deep Learning AI Artificial Intelligence "
            "TensorFlow PyTorch Data Science NLP Natural Language Processing "
            "AWS ML Google Cloud AI Azure AI MLOps Cloud Computing "
            "Python Data Engineering Big Data Distributed Systems"
        )

        if self.engine:
            self.fields_embedding = self.engine.encode([self.relevant_fields_concept])[0]
            norm = np.linalg.norm(self.fields_embedding)
            if norm > 0:
                self.fields_embedding = self.fields_embedding / norm

            self.certs_embedding = self.engine.encode([self.relevant_certs_concept])[0]
            norm = np.linalg.norm(self.certs_embedding)
            if norm > 0:
                self.certs_embedding = self.certs_embedding / norm

    def _cosine_sim(self, v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def score_education(self, candidate):
        """
        Scores education relevance:
        - Institution tier (tier_1 > tier_2 > tier_3 > tier_4)
        - Field of study semantic relevance to ML/AI/CS
        - Degree level (PhD > Masters > Bachelors)
        """
        education = candidate.get('education', [])
        if not education:
            return 0.3  # Neutral — no education data shouldn't heavily penalize

        score = 0.0
        best_tier_score = 0.0
        best_field_score = 0.0
        best_degree_score = 0.0

        for edu in education:
            # Institution tier scoring
            tier = edu.get('tier', 'unknown')
            tier_scores = {
                'tier_1': 1.0,    # IIT, IIIT, NIT, IISc, top global
                'tier_2': 0.70,
                'tier_3': 0.40,
                'tier_4': 0.20,
                'unknown': 0.30,
            }
            tier_score = tier_scores.get(tier, 0.30)
            best_tier_score = max(best_tier_score, tier_score)

            # Field of study — semantic match
            field = edu.get('field_of_study', '')
            if field and self.engine:
                field_emb = self.engine.encode([field])[0]
                field_sim = self._cosine_sim(self.fields_embedding, field_emb)
                # Rescale: sim > 0.5 is highly relevant, < 0.2 is unrelated
                field_score = max(0.0, min(1.0, (field_sim - 0.1) / 0.5))
                best_field_score = max(best_field_score, field_score)
            elif field:
                # Fallback: simple check for CS/ML/Math related terms
                relevant_terms = ['computer', 'software', 'data', 'machine', 'artificial',
                                  'information', 'math', 'statistics', 'electrical', 'electronics']
                field_lower = field.lower()
                if any(term in field_lower for term in relevant_terms):
                    best_field_score = max(best_field_score, 0.7)
                else:
                    best_field_score = max(best_field_score, 0.2)

            # Degree level scoring
            degree = edu.get('degree', '').lower()
            if any(d in degree for d in ['phd', 'doctorate', 'ph.d']):
                best_degree_score = max(best_degree_score, 1.0)
            elif any(d in degree for d in ['master', 'mtech', 'm.tech', 'ms', 'm.s', 'mba']):
                best_degree_score = max(best_degree_score, 0.7)
            elif any(d in degree for d in ['bachelor', 'btech', 'b.tech', 'bs', 'b.s', 'be', 'b.e']):
                best_degree_score = max(best_degree_score, 0.5)
            else:
                best_degree_score = max(best_degree_score, 0.3)

        # Combine: field relevance matters most, then tier, then degree level
        score = (best_field_score * 0.50) + (best_tier_score * 0.30) + (best_degree_score * 0.20)
        return min(1.0, score)

    def score_certifications(self, candidate):
        """
        Scores certification relevance using semantic matching against ML/AI concepts.
        """
        certifications = candidate.get('certifications', [])
        if not certifications:
            return 0.0  # No certs = no bonus (not a penalty)

        if not self.engine:
            return 0.1  # Minimal credit if we can't do semantic matching

        best_sim = 0.0
        cert_count_relevant = 0

        for cert in certifications:
            cert_text = f"{cert.get('name', '')} {cert.get('issuer', '')}"
            if not cert_text.strip():
                continue

            cert_emb = self.engine.encode([cert_text])[0]
            sim = self._cosine_sim(self.certs_embedding, cert_emb)

            if sim > 0.35:  # Threshold for "relevant" certification
                cert_count_relevant += 1
            best_sim = max(best_sim, sim)

        # Score based on best cert relevance + quantity bonus
        score = 0.0
        if best_sim > 0.5:
            score += 0.60
        elif best_sim > 0.35:
            score += 0.35
        elif best_sim > 0.2:
            score += 0.15

        # Bonus for multiple relevant certs
        if cert_count_relevant >= 3:
            score += 0.25
        elif cert_count_relevant >= 1:
            score += 0.10

        return min(1.0, score)

    def compute_education_score(self, candidate):
        """
        Combined education + certification score for a candidate.
        Education has higher weight since it's a more fundamental signal.
        """
        edu_score = self.score_education(candidate)
        cert_score = self.score_certifications(candidate)

        # Education = 75%, Certifications = 25%
        combined = edu_score * 0.75 + cert_score * 0.25
        return combined
