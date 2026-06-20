import math
from datetime import datetime

class BehavioralScorer:
    """
    Hiring probability and behavioral fit scoring.
    These are the legacy split scorers kept for backward compatibility with the 
    weight system, while the new SignalScorer provides the comprehensive 23-signal score.
    """

    def score_hiring_probability(self, candidate, reference_date=datetime(2026, 6, 12)):
        """
        Probability that this candidate can actually be hired:
        - Recruiter response rate (weighted with exponential penalty for very low rates)
        - Notice period (decay function)
        - Open to work flag
        - Location alignment with JD requirements (Pune/Noida preferred)
        - Willingness to relocate
        - Offer acceptance history
        """
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        # Recruiter response rate (0.0 to 1.0)
        rr = signals.get('recruiter_response_rate', 0)
        score += (rr ** 1.5) * 0.30
        
        # Notice period — max points at 0 days, decaying to 0 at 90 days
        notice = signals.get('notice_period_days', 90)
        notice_score = max(0, (90 - notice) / 90.0) * 0.20
        score += notice_score
        
        if signals.get('open_to_work_flag', False):
            score += 0.15
            
        # Location — JD specifies Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR welcome
        location = candidate.get('profile', {}).get('location', '').lower()
        preferred_locs = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'gurgaon', 'gurugram']
        if any(loc in location for loc in preferred_locs):
            score += 0.15
        elif signals.get('willing_to_relocate', False):
            score += 0.10
        
        # Offer acceptance rate (-1 to 1.0)
        oar = signals.get('offer_acceptance_rate', -1)
        if oar >= 0.5:
            score += 0.10
        elif oar >= 0:
            score += 0.05
        
        # Average response time bonus
        avg_resp = signals.get('avg_response_time_hours', 72)
        if avg_resp < 12:
            score += 0.10
        elif avg_resp < 24:
            score += 0.05
            
        return min(1.0, score)
        
    def score_behavioral_fit(self, candidate, reference_date=datetime(2026, 6, 12)):
        """
        Professional engagement and platform behavior:
        - Profile completeness
        - Interview completion rate
        - Recency of platform activity (exponential decay)
        - Verification signals (email, phone, LinkedIn)
        - Applications submitted (actively looking)
        """
        signals = candidate.get('redrob_signals', {})
        score = 0.0
        
        # Profile completeness (0-100)
        completeness = signals.get('profile_completeness_score', 0)
        score += (completeness / 100.0) * 0.25
        
        # Interview completion rate (0-1)
        score += signals.get('interview_completion_rate', 0) * 0.20
        
        # Recency — exponential decay for inactivity
        last_active = signals.get('last_active_date', '2020-01-01')
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = max(0, (reference_date - last_active_dt).days)
            decay_score = 0.20 * math.exp(-0.015 * days_inactive)
            score += decay_score
            
            # Heavy penalty if inactive for > 6 months
            if days_inactive > 180:
                score -= 0.3
        except (ValueError, TypeError):
            pass
        
        # Verification signals
        if signals.get('verified_email', False):
            score += 0.08
        if signals.get('verified_phone', False):
            score += 0.07
        if signals.get('linkedin_connected', False):
            score += 0.05
        
        # Active job seeking
        apps = signals.get('applications_submitted_30d', 0)
        if apps > 5:
            score += 0.10
        elif apps > 0:
            score += 0.05
        
        # Saved by recruiters — external validation
        saved = signals.get('saved_by_recruiters_30d', 0)
        if saved > 10:
            score += 0.10
        elif saved > 3:
            score += 0.05
            
        return max(0.0, min(1.0, score))
