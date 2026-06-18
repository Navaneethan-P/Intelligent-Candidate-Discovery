import google.generativeai as genai
import json
import time
from config import GEMINI_API_KEY

# Use gemini-2.5-flash — much higher free-tier quota than gemini-2.5-pro
LLM_MODEL = 'gemini-2.5-flash'

class LLMReranker:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.quota_exhausted = False  # Circuit breaker: stop calling API if day quota is gone

    def _build_local_reasoning(self, candidate, scores_dict=None):
        """Generates a clean, structured reasoning string from local signals when LLM is unavailable."""
        profile = candidate.get('profile', {})
        title = profile.get('current_title', 'Engineer')
        yoe = profile.get('years_of_experience', 0)
        signals = candidate.get('redrob_signals', {})
        github = signals.get('github_activity_score', -1)
        history = candidate.get('career_history', [])
        companies = [r.get('company', '') for r in history[:3]]
        company_str = ', '.join([c for c in companies if c]) or 'various companies'
        
        parts = [f"{title} with {yoe} years of experience at {company_str}."]
        if github > 50:
            parts.append(f"High GitHub activity score ({github}) signals active open-source contributions.")
        if scores_dict:
            tech = scores_dict.get('technical_fit', 0)
            if tech > 0.7:
                parts.append("Strong semantic alignment with ML/Retrieval system requirements.")
            elif tech > 0.5:
                parts.append("Moderate technical overlap with the role's core ML requirements.")
        return ' '.join(parts)

    def rerank_candidate(self, candidate, raw_jd, scores_dict=None):
        """
        Takes a candidate and the raw JD, asks the LLM to score the candidate from 0 to 100
        and provide a reasoning string. Falls back to local reasoning if quota is exhausted.
        """
        # Circuit breaker: skip LLM calls if we know quota is exhausted
        if self.quota_exhausted:
            return 0, self._build_local_reasoning(candidate, scores_dict)

        profile = candidate.get('profile', {})
        history = candidate.get('career_history', [])
        
        history_str = ""
        for role in history[:4]:  # Keep token usage low
            desc = (role.get('description', '') or '')[:300]  # Truncate descriptions
            history_str += f"- {role.get('title')} at {role.get('company')}\n  {desc}\n"

        prompt = f"""You are an expert technical recruiter. Score this candidate 0-100 and explain why in 1-2 sentences, citing specific evidence.

Return ONLY a valid JSON object: {{"score": <integer>, "reasoning": "<string>"}}

JD SUMMARY: {raw_jd[:500]}

CANDIDATE: {profile.get('current_title', 'Unknown')}, {profile.get('years_of_experience', 0)} yrs exp
{history_str}"""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                time.sleep(2 + attempt * 3)  # Exponential backoff: 2s, 5s
                response = self.model.generate_content(prompt)
                text = response.text.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
                result = json.loads(text)
                return result.get('score', 50), result.get('reasoning', self._build_local_reasoning(candidate, scores_dict))
            except Exception as e:
                err_str = str(e)
                if '429' in err_str and 'PerDay' in err_str:
                    print(f"Daily quota exhausted. Switching to local reasoning for remaining candidates.")
                    self.quota_exhausted = True
                    break
                elif '429' in err_str:
                    print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting...")
                    time.sleep(30)  # Wait 30s on per-minute rate limits
                else:
                    print(f"LLM error: {e}")
                    break

        return 0, self._build_local_reasoning(candidate, scores_dict)
