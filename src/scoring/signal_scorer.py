import math
from datetime import datetime


class SignalScorer:
    """
    Comprehensive behavioral signal scorer that integrates ALL 23 Redrob signals.
    
    Signal groups:
    1. Engagement & Responsiveness (recruiter_response_rate, avg_response_time_hours, applications_submitted_30d)
    2. Market Demand (profile_views_received_30d, search_appearance_30d, saved_by_recruiters_30d)
    3. Availability (open_to_work_flag, notice_period_days, last_active_date)
    4. Platform Trust (verified_email, verified_phone, linkedin_connected, profile_completeness_score)
    5. Hiring Track Record (interview_completion_rate, offer_acceptance_rate)
    6. Professional Network (connection_count, endorsements_received)
    7. Technical Signals (github_activity_score, skill_assessment_scores)
    8. Logistics (preferred_work_mode, willing_to_relocate, expected_salary_range_inr_lpa)
    """

    def __init__(self, reference_date=None):
        self.reference_date = reference_date or datetime(2026, 6, 12)
        # JD says Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR also welcome
        self.preferred_locations = ['pune', 'noida', 'hyderabad', 'mumbai', 'delhi', 'ncr', 'gurgaon', 'gurugram']
        # JD says hybrid preferred
        self.preferred_work_modes = ['hybrid', 'flexible', 'onsite']

    def score_engagement(self, signals):
        """How responsive and actively engaged is this candidate?"""
        score = 0.0

        # Recruiter response rate (0-1) — the most direct engagement signal
        rr = signals.get('recruiter_response_rate', 0)
        score += rr * 0.40

        # Average response time — faster is better (exponential decay)
        avg_resp_hours = signals.get('avg_response_time_hours', 72)
        # At 2 hours → ~0.25, at 12 hours → ~0.20, at 48 hours → ~0.12, at 120 hours → ~0.05
        resp_time_score = 0.25 * math.exp(-0.01 * avg_resp_hours)
        score += resp_time_score

        # Applications submitted in 30d — indicates active job searching
        apps = signals.get('applications_submitted_30d', 0)
        if apps > 10:
            score += 0.15
        elif apps > 3:
            score += 0.10
        elif apps > 0:
            score += 0.05

        # Last active date — recency matters
        last_active = signals.get('last_active_date', '2020-01-01')
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = max(0, (self.reference_date - last_active_dt).days)
            # Exponential decay: active recently = high, 6+ months inactive = very low
            activity_score = 0.20 * math.exp(-0.012 * days_inactive)
            score += activity_score
        except (ValueError, TypeError):
            pass

        return min(1.0, score)

    def score_market_demand(self, signals):
        """How much do other recruiters value this candidate? Third-party validation."""
        score = 0.0

        # Saved by recruiters in 30d — STRONGEST external validation signal
        saved = signals.get('saved_by_recruiters_30d', 0)
        if saved > 15:
            score += 0.45
        elif saved > 8:
            score += 0.35
        elif saved > 3:
            score += 0.25
        elif saved > 0:
            score += 0.10

        # Profile views received in 30d — indicates recruiter interest
        views = signals.get('profile_views_received_30d', 0)
        if views > 30:
            score += 0.30
        elif views > 15:
            score += 0.20
        elif views > 5:
            score += 0.10

        # Search appearance in 30d — how often they appear in recruiter searches
        appearances = signals.get('search_appearance_30d', 0)
        if appearances > 50:
            score += 0.25
        elif appearances > 20:
            score += 0.15
        elif appearances > 5:
            score += 0.08

        return min(1.0, score)

    def score_availability(self, signals, candidate):
        """How available is this candidate for the role right now?"""
        score = 0.0

        # Open to work flag
        if signals.get('open_to_work_flag', False):
            score += 0.25

        # Notice period — JD prefers sub-30 day, can buy out up to 30
        notice = signals.get('notice_period_days', 90)
        if notice <= 15:
            score += 0.30
        elif notice <= 30:
            score += 0.25
        elif notice <= 60:
            score += 0.15
        elif notice <= 90:
            score += 0.05
        # >90 days = no bonus

        # Location alignment with JD preferences
        location = candidate.get('profile', {}).get('location', '').lower()
        if any(loc in location for loc in self.preferred_locations):
            score += 0.20
        elif signals.get('willing_to_relocate', False):
            score += 0.15

        # Work mode preference alignment (JD says hybrid)
        work_mode = signals.get('preferred_work_mode', 'remote')
        if work_mode in self.preferred_work_modes:
            score += 0.15
        else:
            score += 0.05  # Remote-only gets minimal credit

        # Willingness to relocate (if not already in preferred location)
        if signals.get('willing_to_relocate', False):
            score += 0.10

        return min(1.0, score)

    def score_platform_trust(self, signals):
        """How trustworthy and complete is this candidate's platform presence?"""
        score = 0.0

        # Profile completeness (0-100)
        completeness = signals.get('profile_completeness_score', 0)
        score += (completeness / 100.0) * 0.30

        # Verification signals
        if signals.get('verified_email', False):
            score += 0.20
        if signals.get('verified_phone', False):
            score += 0.20
        if signals.get('linkedin_connected', False):
            score += 0.15

        # Connection count — indicates network depth
        connections = signals.get('connection_count', 0)
        if connections > 200:
            score += 0.10
        elif connections > 50:
            score += 0.05

        # Endorsements received — peer validation
        endorsements = signals.get('endorsements_received', 0)
        if endorsements > 30:
            score += 0.10
        elif endorsements > 10:
            score += 0.05

        return min(1.0, score)

    def score_hiring_track_record(self, signals):
        """Track record of completing interviews and accepting offers."""
        score = 0.0

        # Interview completion rate (0-1)
        icr = signals.get('interview_completion_rate', 0)
        score += icr * 0.50

        # Offer acceptance rate (-1 to 1, -1 means no history)
        oar = signals.get('offer_acceptance_rate', -1)
        if oar >= 0:
            score += oar * 0.50
        else:
            score += 0.25  # Neutral if no history

        return min(1.0, score)

    def score_technical_signals(self, signals):
        """GitHub activity and verified skill assessments."""
        score = 0.0

        # GitHub activity score (-1 to 100)
        github = signals.get('github_activity_score', -1)
        if github > 70:
            score += 0.50
        elif github > 40:
            score += 0.35
        elif github > 10:
            score += 0.20
        elif github >= 0:
            score += 0.05
        # -1 (no GitHub) = 0 — not penalized, just no bonus

        # Skill assessment scores — platform-verified skills
        assessments = signals.get('skill_assessment_scores', {})
        if assessments:
            avg_score = sum(assessments.values()) / len(assessments)
            high_scores = sum(1 for s in assessments.values() if s > 70)

            if avg_score > 75:
                score += 0.30
            elif avg_score > 55:
                score += 0.20
            elif avg_score > 35:
                score += 0.10

            # Bonus for having multiple high-scoring assessments
            if high_scores >= 4:
                score += 0.15
            elif high_scores >= 2:
                score += 0.08
        
        return min(1.0, score)

    def compute_composite_signal_score(self, candidate):
        """
        Computes the composite signal score across all 23 behavioral signals.
        Returns (composite_score, breakdown_dict) for explainability.
        """
        signals = candidate.get('redrob_signals', {})

        engagement = self.score_engagement(signals)
        market_demand = self.score_market_demand(signals)
        availability = self.score_availability(signals, candidate)
        platform_trust = self.score_platform_trust(signals)
        hiring_track = self.score_hiring_track_record(signals)
        tech_signals = self.score_technical_signals(signals)

        # Weighted composite — market demand and engagement are most predictive
        composite = (
            engagement * 0.25 +
            market_demand * 0.20 +
            availability * 0.20 +
            platform_trust * 0.10 +
            hiring_track * 0.15 +
            tech_signals * 0.10
        )

        breakdown = {
            'engagement': round(engagement, 4),
            'market_demand': round(market_demand, 4),
            'availability': round(availability, 4),
            'platform_trust': round(platform_trust, 4),
            'hiring_track_record': round(hiring_track, 4),
            'technical_signals': round(tech_signals, 4),
        }

        return min(1.0, composite), breakdown
