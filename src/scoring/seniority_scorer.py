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
        
        # Semantic concept for quantifiable impact — replaces the old keyword-based score_evidence
        self.impact_concept = (
            "reduced latency by percentage improved throughput million users requests per second "
            "scaled system billion queries uptime SLA performance optimization quantifiable metrics "
            "increased revenue cost savings efficiency improvement measurable results production impact"
        )
        
        # Semantic concept for service/consulting company culture vs product company culture
        self.product_company_concept = (
            "product company startup technology platform SaaS built product from scratch "
            "user-facing features shipped to production A/B testing product metrics engagement"
        )
        self.service_company_concept = (
            "IT services consulting outsourcing client projects managed services "
            "staff augmentation offshore delivery project management body shopping"
        )

        if self.engine:
            self.concept_embeddings = {
                "leadership": self._normalize(self.engine.encode([self.leadership_concept])[0]),
                "titles":     self._normalize(self.engine.encode([self.titles_concept])[0]),
                "scale":      self._normalize(self.engine.encode([self.scale_concept])[0]),
                "founding":   self._normalize(self.engine.encode([self.founding_concept])[0]),
                "fullstack":  self._normalize(self.engine.encode([self.fullstack_concept])[0]),
                "impact":     self._normalize(self.engine.encode([self.impact_concept])[0]),
                "product_co": self._normalize(self.engine.encode([self.product_company_concept])[0]),
                "service_co": self._normalize(self.engine.encode([self.service_company_concept])[0]),
            }

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

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
        Uses semantic embedding to detect product-company vs service-company culture
        instead of hardcoded company name lists.
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

        # Semantic company culture detection — product vs service company
        # Instead of hardcoded company names, embed career descriptions and
        # check similarity to product-company vs service-company concepts
        if self.engine and chunk_embeddings is not None and len(chunk_embeddings) > 0:
            max_product_sim = max(self._cosine_sim(self.concept_embeddings["product_co"], e) for e in chunk_embeddings)
            max_service_sim = max(self._cosine_sim(self.concept_embeddings["service_co"], e) for e in chunk_embeddings)
            
            if max_product_sim > max_service_sim + 0.1:
                # Career descriptions read like product company work
                score += 0.2
            elif max_service_sim > max_product_sim + 0.1:
                # Career descriptions read like service/consulting work
                score -= 0.3
        
        return max(0.0, min(1.0, score))

    def score_evidence(self, candidate, chunk_embeddings=None):
        """
        Scores based on quantifiable impact metrics in career descriptions.
        Fully semantic — uses embedding similarity against an "impact metrics" concept
        instead of hardcoded keyword lists.
        """
        if self.engine and chunk_embeddings is not None and len(chunk_embeddings) > 0:
            # Semantic approach: compare career chunks against impact concept
            impact_sims = [
                self._cosine_sim(self.concept_embeddings["impact"], e)
                for e in chunk_embeddings
            ]
            max_impact_sim = max(impact_sims)
            avg_impact_sim = sum(impact_sims) / len(impact_sims)
            
            # Score based on how much the career text semantically resembles
            # quantifiable impact language
            score = 0.0
            if max_impact_sim > 0.5:
                score += 0.6
            elif max_impact_sim > 0.4:
                score += 0.4
            elif max_impact_sim > 0.3:
                score += 0.2
            
            # Bonus if multiple chunks contain impact language (consistency)
            high_impact_chunks = sum(1 for s in impact_sims if s > 0.35)
            if high_impact_chunks >= 3:
                score += 0.3
            elif high_impact_chunks >= 2:
                score += 0.2
            elif high_impact_chunks >= 1:
                score += 0.1
            
            return min(1.0, score)
        
        # Fallback for when no embeddings are available
        return 0.0
