import numpy as np

class SeniorityScorer:
    def __init__(self, engine=None):
        self.engine = engine

        # Pre-compute concept embeddings for semantic scoring — no regex, no keywords
        self.leadership_concept = "Led a team of engineers, architected systems from scratch, infrastructure scale, managed team, spearheaded, drove"
        self.titles_concept = "senior staff principal lead head director founder architect"
        self.scale_concept = "scalable production deployed infrastructure high volume systems"
        self.founding_concept = "0 to 1 zero to one startup early stage founding team multiple hats mvp prototype contributor"
        self.fullstack_concept = "full stack frontend backend devops ci/cd infrastructure deployment"

        if self.engine:
            self.concept_embeddings = {
                "leadership": self.engine.encode([self.leadership_concept])[0],
                "titles":     self.engine.encode([self.titles_concept])[0],
                "scale":      self.engine.encode([self.scale_concept])[0],
                "founding":   self.engine.encode([self.founding_concept])[0],
                "fullstack":  self.engine.encode([self.fullstack_concept])[0],
            }

    def _cosine_sim(self, v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def score_seniority(self, candidate, chunk_embeddings=None):
        """
        Evidence of scale, architecture, and leadership — pure semantic cosine similarity.
        """
        if not self.engine or chunk_embeddings is None or len(chunk_embeddings) == 0:
            return 0.0

        score = 0.0
        max_leadership = max(self._cosine_sim(self.concept_embeddings["leadership"], e) for e in chunk_embeddings)
        max_titles     = max(self._cosine_sim(self.concept_embeddings["titles"],     e) for e in chunk_embeddings)
        max_scale      = max(self._cosine_sim(self.concept_embeddings["scale"],      e) for e in chunk_embeddings)

        if max_leadership > 0.4: score += 0.4
        if max_titles     > 0.4: score += 0.3
        if max_scale      > 0.4: score += 0.3

        return min(1.0, score)

    def score_founding_fit(self, candidate, chunk_embeddings=None):
        """
        Evidence of zero-to-one engineering — pure semantic cosine similarity.
        """
        score = 0.0

        if self.engine and chunk_embeddings is not None and len(chunk_embeddings) > 0:
            max_founding  = max(self._cosine_sim(self.concept_embeddings["founding"],  e) for e in chunk_embeddings)
            max_fullstack = max(self._cosine_sim(self.concept_embeddings["fullstack"], e) for e in chunk_embeddings)
            if max_founding  > 0.4: score += 0.5
            if max_fullstack > 0.4: score += 0.3

        github = candidate.get('redrob_signals', {}).get('github_activity_score', -1)
        if github > 50:
            score += 0.2
        elif github > 0:
            score += 0.1

        # Service company check (company name substring, not keywords in resume text)
        service_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'mindtree']
        company_names = [role.get('company', '').lower() for role in candidate.get('career_history', [])]
        all_service = len(company_names) > 0 and all(any(sc in c for sc in service_companies) for c in company_names)
        if all_service:
            score -= 0.5
        else:
            score += 0.2

        return max(0.0, min(1.0, score))

    def score_evidence(self, candidate):
        """
        Scores based on quantifiable impact metrics in the resume.
        Pure Python token scan — no regex import.
        """
        text_parts = []
        for role in candidate.get('career_history', []):
            text_parts.append(role.get('description', '').lower())
        words = " ".join(text_parts).split()

        scale_vocab = {
            'latency', 'rps', 'qps', 'dau', 'mau', 'users',
            'million', 'billion', 'throughput', 'uptime', 'p99', 'p95',
        }

        metric_count = sum(1 for w in words if any(ch.isdigit() for ch in w))
        scale_count  = sum(1 for w in words if w.strip('.,;:') in scale_vocab)

        return min(1.0, (metric_count + scale_count) / 4.0)
