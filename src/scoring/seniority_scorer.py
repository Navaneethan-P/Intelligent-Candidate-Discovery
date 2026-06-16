import re

class SeniorityScorer:
    def score_seniority(self, candidate):
        """
        Extracts evidence of scale, architecture, and leadership.
        """
        text_parts = []
        for role in candidate.get('career_history', []):
            text_parts.append(role.get('description', '').lower())
            text_parts.append(role.get('title', '').lower())
        full_text = " ".join(text_parts)
        
        score = 0.0
        
        # Leadership & Ownership
        if re.search(r'\b(led|architected|designed|owned|built from scratch|founded|managed team|spearheaded|drove)\b', full_text):
            score += 0.4
            
        # Titles
        if re.search(r'\b(senior|staff|principal|lead|head|director|founder|architect)\b', full_text):
            score += 0.3
            
        # Scale & Infrastructure
        if re.search(r'\b(scalable|production|deployed|infrastructure)\b', full_text):
            score += 0.3
            
        return min(1.0, score)
        
    def score_founding_fit(self, candidate):
        """
        Extracts evidence of zero-to-one engineering.
        """
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
            
        # Service company penalty check
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
        Scores based on quantifiable metrics in the resume.
        """
        text_parts = []
        for role in candidate.get('career_history', []):
            text_parts.append(role.get('description', '').lower())
        full_text = " ".join(text_parts)
        
        # Numbers followed by %, x, M, K, B, etc.
        metrics = re.findall(r'\b(?:[0-9]+(?:\.[0-9]+)?(?:%|x|m|k|b|tb|gb)|[0-9]+(?:,[0-9]+)+|\$[0-9]+[mkb]?)\b', full_text)
        scale_indicators = re.findall(r'\b(?:latency|requests/sec|rps|qps|dau|mau|users|million|billion)\b', full_text)
        
        total_evidence = len(metrics) + len(scale_indicators)
        score = min(1.0, total_evidence / 4.0)
        return score
