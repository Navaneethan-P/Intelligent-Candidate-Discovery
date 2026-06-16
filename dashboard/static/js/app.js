let candidates = [];
let logs = {};

document.addEventListener('DOMContentLoaded', () => {
    // Dynamic initialization call to API interface
    fetch("/api/dashboard")
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error("Pipeline Communication Error:", data.error);
                return;
            }

            candidates = data.table;
            logs = data.logs;

            // Render Core Analytical Metric Arrays in Executive Summary
            document.getElementById("topCandidates").innerText = data.summary.top_candidates;
            document.getElementById("avgTech").innerText = data.summary.avg_tech + "%";
            document.getElementById("avgFounding").innerText = data.summary.avg_founding + "%";
            document.getElementById("avgHiring").innerText = data.summary.avg_hiring + "%";

            populateComponents();
        });
});

function populateComponents() {
    const select = document.getElementById("candidateSelect");
    const tbody = document.querySelector("#candidateTable tbody");
    
    select.innerHTML = "";
    tbody.innerHTML = "";

    candidates.forEach((c) => {
        // Append Dropdown Option
        const option = document.createElement("option");
        option.value = c.candidate_id;
        option.textContent = c.candidate_id;
        select.appendChild(option);

        // Append Data Table Entry Row
        const row = document.createElement("tr");
        row.innerHTML = `
            <td style="font-weight:500; color:#111827;">${c.candidate_id}</td>
            <td>${c.rank}</td>
            <td>${c.score}</td>
            <td title="${c.reasoning}">${c.reasoning}</td>
        `;
        
        // Let table row click select candidate
        row.addEventListener('click', () => {
            select.value = c.candidate_id;
            renderTargetProfile(c.candidate_id);
        });
        
        tbody.appendChild(row);
    });

    // Populate index matching view instantly on load
    if (candidates.length > 0) {
        renderTargetProfile(candidates[0].candidate_id);
    }

    select.addEventListener("change", e => renderTargetProfile(e.target.value));
}

function renderTargetProfile(id) {
    const candidate = candidates.find(c => c.candidate_id === id);
    const log = logs[id];

    if (!candidate) return;

    // Direct binding of baseline layout parameters
    document.getElementById("candidateTitle").innerText = "Candidate: " + id;
    document.getElementById("rank").innerText = candidate.rank;
    document.getElementById("score").innerText = candidate.score;
    document.getElementById("reasoning").innerText = candidate.reasoning;

    if (log && log.scores) {
        const s = log.scores;

        // Map numeric indicators and calculate metrics cleanly
        updateMetricPillar("techBar", "techVal", s.technical_fit);
        updateMetricPillar("seniorBar", "seniorVal", s.seniority_fit);
        updateMetricPillar("foundingBar", "foundingVal", s.founding_fit);
        updateMetricPillar("hiringBar", "hiringVal", s.hiring_probability);
        updateMetricPillar("behaviorBar", "behaviorVal", s.behavioral_fit);
        updateMetricPillar("evidenceBar", "evidenceVal", s.evidence_strength);
    }
}

function updateMetricPillar(barId, valId, value) {
    // Process input scaling factors natively (handles both 0-1 range floats and 0-100 integers gracefully)
    const normalizedPercent = value <= 1 ? Math.round(value * 100) : Math.round(value);
    
    // Explicitly target the value ID span so it isolates the text change without breaking layout alignment
    const scoreValElement = document.getElementById(valId);
    if (scoreValElement) {
        scoreValElement.innerText = normalizedPercent;
    }
    
    // Animate or shift the visual bar width seamlessly
    const barElement = document.getElementById(barId);
    if (barElement) {
        barElement.style.width = normalizedPercent + "%";
    }
}