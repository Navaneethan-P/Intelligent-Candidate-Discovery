import os
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AI Recruiter Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Outputs are technically in the root dir's outputs, let's go one up then into outputs
ROOT_DIR = os.path.dirname(BASE_DIR)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="Index not found", status_code=404)

@app.get("/api/dashboard")
async def get_dashboard_data():
    csv_path = os.path.join(OUTPUTS_DIR, "team_submission.csv")
    json_path = os.path.join(OUTPUTS_DIR, "explainability_logs.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return JSONResponse(content={
            "table": [],
            "logs": {},
            "summary": {"processed": "5,000+", "top_candidates": 0, "avg_tech": 0, "avg_founding": 0, "avg_hiring": 0}
        })

    try:
        df = pd.read_csv(csv_path)
        with open(json_path, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if logs:
            avg_tech = sum(log["scores"]["technical_fit"] for log in logs.values()) / len(logs)
            avg_founding = sum(log["scores"]["founding_fit"] for log in logs.values()) / len(logs)
            avg_hiring = sum(log["scores"]["hiring_probability"] for log in logs.values()) / len(logs)
        else:
            avg_tech = avg_founding = avg_hiring = 0

        return JSONResponse(content={
            "table": df.to_dict(orient="records"),
            "logs": logs,
            "summary": {
                "processed": "5,000+",
                "top_candidates": len(df),
                "avg_tech": round(avg_tech * 100, 1),
                "avg_founding": round(avg_founding * 100, 1),
                "avg_hiring": round(avg_hiring * 100, 1)
            }
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/health")
async def health_check():
    csv_path = os.path.join(OUTPUTS_DIR, "team_submission.csv")
    json_path = os.path.join(OUTPUTS_DIR, "explainability_logs.json")
    return JSONResponse(content={
        "status": "ok",
        "pipeline": "Agentic Two-Stage Hybrid Retrieval Engine",
        "models": {
            "embedding": "all-MiniLM-L6-v2",
            "llm_reranker": "gemini-2.5-flash",
            "vector_index": "FAISS (IndexFlatIP)"
        },
        "outputs_ready": os.path.exists(csv_path) and os.path.exists(json_path)
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)