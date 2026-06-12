import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="AI Recruiter Dashboard", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

DASH_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DASH_DIR)
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

col_logo, col_title = st.columns([1, 10])
with col_logo:
    try:
        st.image(os.path.join(DASH_DIR, "logo.png"), width=70)
    except Exception:
        pass
        
with col_title:
    st.title("AI Recruiter Dashboard")
    
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(OUTPUTS_DIR, "team_submission.csv"))
        with open(os.path.join(OUTPUTS_DIR, "explainability_logs.json"), "r", encoding="utf-8") as f:
            logs = json.load(f)
        return df, logs
    except FileNotFoundError:
        return None, None

df, logs = load_data()

if df is None or logs is None:
    st.warning("Pipeline is still running. Output files not yet generated. Please wait...")
else:
    st.subheader("Executive Summary")
    
    avg_tech = sum(log['scores']['technical_fit'] for log in logs.values()) / len(logs)
    avg_founding = sum(log['scores']['founding_fit'] for log in logs.values()) / len(logs)
    avg_hiring = sum(log['scores']['hiring_probability'] for log in logs.values()) / len(logs)
    
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Candidates Processed", "50,000+")
    col_b.metric("Top Candidates", len(df))
    col_c.metric("Avg Technical Fit", f"{avg_tech*100:.1f}%")
    col_d.metric("Avg Founding Fit", f"{avg_founding*100:.1f}%")
    col_e.metric("Avg Hiring Prob", f"{avg_hiring*100:.1f}%")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Top 100 Candidates")
        selected_cid = st.selectbox("Select Candidate:", df['candidate_id'].tolist())
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col2:
        if selected_cid:
            candidate_row = df[df['candidate_id'] == selected_cid].iloc[0]
            candidate_log = logs.get(selected_cid, {})
            
            st.subheader(f"Candidate: {selected_cid}")
            st.markdown(f"**Rank:** #{candidate_row['rank']}")
            st.markdown(f"**Final Score:** {candidate_row['score']}")
            
            st.markdown("### Recruiter Reasoning")
            st.info(candidate_row['reasoning'])
            
            if candidate_log:
                st.markdown("### Six Pillar Score Breakdown")
                scores = candidate_log['scores']
                
                sc_tech = scores.get('technical_fit', 0)
                sc_senior = scores.get('seniority_fit', 0)
                sc_founding = scores.get('founding_fit', 0)
                sc_hiring = scores.get('hiring_probability', 0)
                sc_behavior = scores.get('behavioral_fit', 0)
                sc_evidence = scores.get('evidence_strength', 0)
                
                st.progress(sc_tech, text=f"Technical Fit: {sc_tech*100:.0f}")
                st.progress(sc_senior, text=f"Seniority Fit: {sc_senior*100:.0f}")
                st.progress(sc_founding, text=f"Founding-Team Fit: {sc_founding*100:.0f}")
                st.progress(sc_hiring, text=f"Hiring Probability: {sc_hiring*100:.0f}")
                st.progress(sc_behavior, text=f"Behavioral Fit: {sc_behavior*100:.0f}")
                st.progress(sc_evidence, text=f"Evidence Strength: {sc_evidence*100:.0f}")
