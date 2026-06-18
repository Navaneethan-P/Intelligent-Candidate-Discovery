# Intelligent Candidate Discovery System

A candidate ranking system for the Redrob AI Hackathon that combines semantic search, scoring layers, and explainability into a simple review workflow.

![AI Recuriter Dashboard](
   dashboard.png
)
## What this project does
- Reads candidate profiles from a large JSONL dataset
- Converts candidate text and the job description into vector embeddings
- Scores candidates using:
  - technical relevance
  - seniority and startup experience
  - founding / leadership potential
  - hiring readiness
  - behavioral signals
- Writes a ranked CSV submission and a JSON explainability log
- Serves a local browser dashboard for reviewing results

## Why it matters
Traditional recruiting systems often depend on keywords. This project uses semantic matching and behavioral scoring so candidate ranking is based on meaning and evidence rather than exact phrases.

## Project layout
- `src/build_ranking.py` — main ranking pipeline
- `src/config.py` — dataset paths, model selection, weights, and output locations
- `src/pipeline/` — data loading, chunk creation, embedding, FAISS retrieval, and explainability
- `src/scoring/` — separate scoring modules for technical, seniority, and behavioral evaluation
- `dashboard/` — HTML dashboard and Python server for viewing results
- `outputs/` — generated `team_submission.csv` and `explainability_logs.json`
- `tools/validate_submission.py` — validate the final submission format

## Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the challenge dataset in `challenge_data/India_runs_data_and_ai_challenge`.

3. Open `src/config.py` and set `DATA_DIR` to the dataset folder if needed.

## Run the ranking pipeline
```bash
python src/build_ranking.py
```

## Run the dashboard
```bash
python dashboard/app.py
```

Then open `http://localhost:5000` in your browser.

## Notes
- `src/config.py` uses `MAX_CANDIDATES = 5000` by default for faster testing. Set it to `None` to process the full dataset.
- If the dashboard is empty, run the ranking pipeline first so `outputs/team_submission.csv` and `outputs/explainability_logs.json` are created.
- Output files:
  - `outputs/team_submission.csv`
  - `outputs/explainability_logs.json`

## Tips
- Keep the dataset path up to date in `src/config.py`.
- Use the dashboard to inspect top-ranked candidates and their scoring reasons.
