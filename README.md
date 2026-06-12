# Intelligent Candidate Discovery System

## Overview
This repository contains a comprehensive candidate ranking and evaluation system designed to process large-scale applicant datasets and surface high-quality engineering talent based on multi-dimensional scoring. 

Traditional applicant tracking systems rely heavily on keyword matching, which often overlooks capable engineers and disproportionately favors keyword stuffing. To address this, we implemented a structured evaluation framework that computationally models the candidate review process.

## Methodology
The system evaluates candidates across six distinct pillars:

1. Technical Fit: Measures semantic overlap between the candidate's career history and the job description using dense vector embeddings.
2. Seniority Trajectory: Analyzes career progression to identify candidates who have successfully architected and owned systems, rather than just participating in them.
3. Founding-Team Fit: Identifies candidates with experience in fast-paced, zero-to-one startup environments and broad technical ownership.
4. Hiring Probability: Evaluates availability, responsiveness, and geographical alignment based on provided behavioral signals.
5. Behavioral Fit: Assesses profile completeness and recent platform activity.
6. Evidence Strength: Quantifies the presence of measurable engineering impact (e.g., latency reductions, system scale, user metrics) within the candidate's resume.

## Architecture and Tools
- Data Processing: Handles batch ingestion and transformation of large-scale JSONL and DOCX data.
- Semantic Analysis: Utilizes Sentence Transformers (all-MiniLM-L6-v2) to calculate the cosine similarity between the job description and candidate profiles.
- Heuristic Scoring: Employs targeted regex and heuristic patterns to evaluate unquantifiable signals like architectural ownership and startup experience.
- Presentation Layer: Features a clean, analytical dashboard built with Streamlit for reviewing top candidates, their score breakdowns, and the system's underlying reasoning.

## Repository Structure
- `src/`: Contains the core ranking engine and scoring logic.
- `dashboard/`: Contains the Streamlit application for reviewing results.
- `outputs/`: Stores the generated CSV submission and explainability logs.
- `tools/`: Includes the validation script used to verify output formatting against the official specification.
- `docs/`: Contains detailed methodology documentation.

## How to Run

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the extracted challenge dataset into the `challenge_data/India_runs_data_and_ai_challenge` directory. (Note: Raw data is not included in this repository).

3. Execute the ranking pipeline:
   ```bash
   python src/build_ranking.py
   ```

4. Launch the evaluation dashboard:
   ```bash
   python -m streamlit run dashboard/app.py
   ```

## Output Validation
The generated output can be validated using the provided tool to ensure strict compliance with the submission format:
```bash
python tools/validate_submission.py
```
