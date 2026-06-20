class Explainer:
    def generate_reasoning(self, rank, scores_dict, candidate, best_matches):
        """
        Generates factual, rich reasoning citing specific evidence.
        Covers all scoring dimensions with concrete data points.
        """
        tech = scores_dict.get('technical_fit', 0)
        senior = scores_dict.get('seniority_fit', 0)
        founding = scores_dict.get('founding_fit', 0)
        hiring = scores_dict.get('hiring_probability', 0)
        evidence = scores_dict.get('evidence_strength', 0)
        signal = scores_dict.get('signal_score', 0)
        edu = scores_dict.get('education_fit', 0)
        behavioral = scores_dict.get('behavioral_fit', 0)
        conf_score = scores_dict.get('confidence', 0)
        
        profile = candidate.get('profile', {})
        yoe = profile.get('years_of_experience', 0)
        title = profile.get('current_title', 'Engineer')
        location = profile.get('location', '')
        industry = profile.get('current_industry', '')
        signals = candidate.get('redrob_signals', {})
        
        # Confidence band
        if conf_score >= 80:
            conf_str = "High Confidence"
        elif conf_score >= 55:
            conf_str = "Moderate Confidence"
        else:
            conf_str = "Low Confidence"
            
        # Build the core identity
        reason_parts = [f"Ranked #{rank} ({conf_str}): {title} with {yoe} yrs exp"]
        if location:
            reason_parts[0] += f" in {location}"
        reason_parts[0] += "."
        
        # Technical assessment
        strengths = []
        risks = []
        
        if tech > 0.8:
            strengths.append("Exceptional technical alignment with ML/Retrieval/Ranking requirements")
        elif tech > 0.65:
            strengths.append("Strong technical fit for core ML systems")
        elif tech > 0.5:
            strengths.append("Moderate technical overlap with role requirements")
        else:
            risks.append("Limited direct technical match with core JD requirements")
            
        if senior > 0.7:
            strengths.append("proven architecture ownership and leadership at scale")
        elif senior > 0.4:
            strengths.append("some evidence of system design responsibility")
            
        if founding > 0.7:
            strengths.append("strong 0-to-1 startup DNA")
        
        if evidence > 0.6:
            strengths.append("quantifiable impact metrics in career history")
            
        if edu > 0.7:
            education = candidate.get('education', [])
            if education:
                top_edu = education[0]
                strengths.append(f"{top_edu.get('degree', '')} in {top_edu.get('field_of_study', '')} ({top_edu.get('tier', 'unknown')})")
        
        # Behavioral/signal assessment
        github = signals.get('github_activity_score', -1)
        if github > 50:
            strengths.append(f"GitHub activity: {github:.0f}/100")
        
        saved = signals.get('saved_by_recruiters_30d', 0)
        if saved > 5:
            strengths.append(f"bookmarked by {saved} recruiters in 30d")
        
        rr = signals.get('recruiter_response_rate', 0)
        if rr > 0.6:
            strengths.append(f"high recruiter engagement ({rr:.0%} response rate)")
        elif rr < 0.15:
            risks.append(f"very low recruiter response rate ({rr:.0%})")
        
        notice = signals.get('notice_period_days', 90)
        if notice <= 30:
            strengths.append(f"immediately available ({notice}d notice)")
        elif notice > 90:
            risks.append(f"long notice period ({notice}d)")
        
        # Assemble
        if strengths:
            reason_parts.append("Strengths: " + " | ".join(strengths) + ".")
        
        if risks:
            reason_parts.append("Risks: " + " | ".join(risks) + ".")
        
        # Add best matching evidence if present
        if best_matches:
            best_chunk = best_matches.get('role_core') or best_matches.get('infrastructure')
            if best_chunk:
                chunk_text = best_chunk['text']
                if len(chunk_text) > 120:
                    chunk_text = chunk_text[:117] + "..."
                reason_parts.append(f"Best match ({best_chunk['type']}): '{chunk_text}'")
                
        return " ".join(reason_parts)
