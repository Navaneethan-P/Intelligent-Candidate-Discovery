class Explainer:
    def generate_reasoning(self, rank, scores_dict, candidate, best_matches):
        """
        Generates factual reasoning incorporating actual matching chunks if available.
        """
        tech = scores_dict['technical_fit']
        senior = scores_dict['seniority_fit']
        founding = scores_dict['founding_fit']
        hiring = scores_dict['hiring_probability']
        evidence = scores_dict['evidence_strength']
        conf_score = scores_dict['confidence']
        
        yoe = candidate.get('profile', {}).get('years_of_experience', 0)
        title = candidate.get('profile', {}).get('current_title', 'Engineer')
        
        if conf_score >= 80:
            conf_str = "95% Confidence"
        elif conf_score >= 60:
            conf_str = "80% Confidence"
        else:
            conf_str = "50% Confidence"
            
        reason_parts = [f"Ranked #{rank} ({conf_str}): {title} with {yoe} yrs exp."]
        
        metrics = []
        if tech > 0.8:
            metrics.append("Exceptional technical match for ML/Retrieval")
        elif tech > 0.6:
            metrics.append("Strong technical foundation")
            
        if senior > 0.8:
            metrics.append("Proven scale/architecture ownership")
            
        if founding > 0.7:
            metrics.append("Strong startup/0-to-1 DNA")
            
        if evidence > 0.6:
            metrics.append("Quantifiable impact metrics")
            
        if hiring > 0.8:
            metrics.append("Highly responsive and immediately available")
            
        if metrics:
            reason_parts.append(" | ".join(metrics) + ".")
        else:
            reason_parts.append("Solid overall fit.")
            
        # Add factual matching if present
        if best_matches:
            # Get the best match for 'role_core' or 'infrastructure'
            best_chunk = best_matches.get('role_core') or best_matches.get('infrastructure')
            if best_chunk:
                # Truncate text for CSV cleanliness
                chunk_text = best_chunk['text']
                if len(chunk_text) > 100:
                    chunk_text = chunk_text[:97] + "..."
                reason_parts.append(f"Semantic match found in {best_chunk['type']}: '{chunk_text}'")
                
        return " ".join(reason_parts)
