import math
from datetime import datetime

class BehavioralScorer:
    def score_hiring_probability(self, candidate, reference_date=datetime(2026, 6, 12)):
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        # Recruiter response rate (0.0 to 1.0)
        # Using a slight exponential penalty for very low response rates
        rr = signals.get('recruiter_response_rate', 0)
        score += (rr ** 1.5) * 0.4
        
        # Notice period
        notice = signals.get('notice_period_days', 90)
        # Decay function: max points at 0 days, decaying to 0 at 90 days
        notice_score = max(0, (90 - notice) / 90.0) * 0.3
        score += notice_score
        
        if signals.get('open_to_work_flag', False):
            score += 0.15
            
        location = candidate.get('profile', {}).get('location', '').lower()
        if any(loc in location for loc in ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr']):
            score += 0.15
        elif signals.get('willing_to_relocate', False):
            score += 0.15
            
        return min(1.0, score)
        
    def score_behavioral_fit(self, candidate, reference_date=datetime(2026, 6, 12)):
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        completeness = signals.get('profile_completeness_score', 0)
        score += (completeness / 100.0) * 0.4
        
        score += signals.get('interview_completion_rate', 0) * 0.3
        
        last_active = signals.get('last_active_date', '2020-01-01')
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (reference_date - last_active_dt).days
            
            # Mathematical decay for inactivity
            if days_inactive < 0: days_inactive = 0
            
            # Exponential decay: e^(-lambda * x)
            # At 30 days, we want score to be ~0.2. lambda = 0.013
            decay_score = 0.3 * math.exp(-0.015 * days_inactive)
            score += decay_score
            
            # Heavy penalty if inactive for > 6 months
            if days_inactive > 180:
                score -= 0.5
        except Exception:
            pass
            
        return max(0.0, min(1.0, score))
