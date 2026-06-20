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
        response_rate = signals.get('recruiter_response_rate', 0)
        notice = signals.get('notice_period_days', 90)
        history = candidate.get('career_history', [])
        companies = [r.get('company', '') for r in history[:3]]
        company_str = ', '.join([c for c in companies if c]) or 'various companies'
        
        parts = [f"{title} with {yoe} years of experience at {company_str}."]
        
        if scores_dict:
            tech = scores_dict.get('technical_fit', 0)
            if tech > 0.7:
                parts.append("Strong semantic alignment with ML/Retrieval system requirements.")
            elif tech > 0.5:
                parts.append("Moderate technical overlap with the role's core ML requirements.")
            else:
                parts.append("Limited direct technical overlap with core JD requirements.")
            
            senior = scores_dict.get('seniority_fit', 0)
            if senior > 0.7:
                parts.append("Demonstrates architecture ownership and leadership at scale.")
            
            founding = scores_dict.get('founding_fit', 0)
            if founding > 0.6:
                parts.append("Shows strong 0-to-1 startup DNA and versatility.")
            
            evidence = scores_dict.get('evidence_strength', 0)
            if evidence > 0.5:
                parts.append("Career descriptions contain quantifiable impact metrics.")
        
        if github > 50:
            parts.append(f"High GitHub activity score ({github:.0f}) signals active open-source engagement.")
        elif github > 0:
            parts.append(f"GitHub activity score of {github:.0f} indicates some open-source presence.")
        
        if response_rate > 0.6:
            parts.append(f"Strong recruiter engagement (response rate: {response_rate:.0%}).")
        
        if notice <= 30:
            parts.append(f"Short notice period ({notice} days) — immediately available.")
        
        return ' '.join(parts)

    def _estimate_llm_score_from_base(self, base_score, scores_dict):
        """
        When LLM is unavailable, estimate an equivalent LLM score from the base semantic scores.
        This prevents the circuit-breaker from collapsing all non-LLM candidates to near-zero.
        """
        if not scores_dict:
            return max(10, int(base_score * 80))
        
        # Weight the components similar to how the LLM would evaluate
        tech = scores_dict.get('technical_fit', 0)
        senior = scores_dict.get('seniority_fit', 0)
        evidence = scores_dict.get('evidence_strength', 0)
        founding = scores_dict.get('founding_fit', 0)
        
        # The LLM heavily weights technical fit and seniority
        estimated = (tech * 0.45 + senior * 0.25 + evidence * 0.15 + founding * 0.15) * 100
        return max(5, min(95, int(estimated)))

    def rerank_candidate(self, candidate, raw_jd, scores_dict=None, base_score=0.0):
        """
        Takes a candidate and the raw JD, asks the LLM to score the candidate from 0 to 100
        and provide a reasoning string. Falls back to estimated scoring if quota is exhausted.
        """
        # Circuit breaker: skip LLM calls but still produce meaningful scores
        if self.quota_exhausted:
            estimated_score = self._estimate_llm_score_from_base(base_score, scores_dict)
            return estimated_score, self._build_local_reasoning(candidate, scores_dict)

        profile = candidate.get('profile', {})
        history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])
        signals = candidate.get('redrob_signals', {})
        
        # Build skills summary
        skills_str = ", ".join([s.get('name', '') for s in skills[:15]])
        
        # Build education summary
        edu_str = ""
        for edu in education[:2]:
            edu_str += f"- {edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')} ({edu.get('tier', 'unknown')})\n"
        
        history_str = ""
        for role in history[:4]:  # Keep token usage low
            desc = (role.get('description', '') or '')[:300]  # Truncate descriptions
            history_str += f"- {role.get('title')} at {role.get('company')} ({role.get('duration_months', 0)} months)\n  {desc}\n"

        # Build behavioral signals summary
        signals_str = f"""Response rate: {signals.get('recruiter_response_rate', 0):.0%}, Notice: {signals.get('notice_period_days', 90)} days, GitHub: {signals.get('github_activity_score', -1)}, Open to work: {signals.get('open_to_work_flag', False)}, Last active: {signals.get('last_active_date', 'unknown')}, Saved by recruiters (30d): {signals.get('saved_by_recruiters_30d', 0)}"""

        prompt = f"""You are an expert technical recruiter evaluating a candidate for a Senior AI/ML Engineer (Founding Team) role. Score this candidate 0-100 and explain why in 2-3 sentences, citing specific evidence from their profile.

CRITICAL EVALUATION RULES:
- Candidates whose primary career is NOT in engineering/tech/ML/AI (e.g., HR, Sales, Marketing, Accounting) should score 0-10.
- Candidates from only service/consulting companies (TCS, Infosys, Wipro) without product company experience should be penalized.
- Weight PRODUCTION experience with embeddings, retrieval, ranking systems most heavily.
- Behavioral signals matter: low response rate or long inactivity = lower score.

Return ONLY a valid JSON object: {{"score": <integer 0-100>, "reasoning": "<string>"}}

JD SUMMARY: {raw_jd[:600]}

CANDIDATE:
Title: {profile.get('current_title', 'Unknown')}
YoE: {profile.get('years_of_experience', 0)} years
Location: {profile.get('location', 'Unknown')}
Industry: {profile.get('current_industry', 'Unknown')}

Skills: {skills_str}

Education:
{edu_str}

Career History:
{history_str}

Behavioral Signals: {signals_str}"""

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
                    print(f"Daily quota exhausted. Switching to estimated scoring for remaining candidates.")
                    self.quota_exhausted = True
                    break
                elif '429' in err_str:
                    print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting...")
                    time.sleep(30)  # Wait 30s on per-minute rate limits
                else:
                    print(f"LLM error: {e}")
                    break

        # Fallback: use estimated score instead of 0
        estimated_score = self._estimate_llm_score_from_base(base_score, scores_dict)
        return estimated_score, self._build_local_reasoning(candidate, scores_dict)
