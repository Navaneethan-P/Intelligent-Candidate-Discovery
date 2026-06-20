let candidates = [];
let logs = {};
let activeRow = null;

document.addEventListener('DOMContentLoaded', () => {
    fetch("/api/dashboard")
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error("Pipeline Communication Error:", data.error);
                return;
            }

            candidates = data.table;
            logs = data.logs;

            // Render Executive Summary
            document.getElementById("processedCount").innerText = data.summary.processed;
            document.getElementById("topCandidates").innerText = data.summary.top_candidates;
            document.getElementById("avgTech").innerText = data.summary.avg_tech + "%";
            document.getElementById("avgSignal").innerText = data.summary.avg_signal + "%";
            document.getElementById("avgHiring").innerText = data.summary.avg_hiring + "%";

            populateTable();

            // Select first candidate
            if (candidates.length > 0) {
                renderTargetProfile(candidates[0].candidate_id);
            }
        })
        .catch(err => {
            console.error("Failed to load dashboard data:", err);
        });

    // Search functionality
    document.getElementById("searchInput").addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        const rows = document.querySelectorAll("#candidateTable tbody tr");
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(query) ? "" : "none";
        });
    });
});

function populateTable() {
    const tbody = document.querySelector("#candidateTable tbody");
    tbody.innerHTML = "";

    candidates.forEach((c) => {
        const row = document.createElement("tr");
        row.dataset.candidateId = c.candidate_id;

        const reasoningTruncated = c.reasoning && c.reasoning.length > 80 
            ? c.reasoning.substring(0, 77) + "..." 
            : (c.reasoning || "—");

        row.innerHTML = `
            <td>${c.rank}</td>
            <td>${c.candidate_id}</td>
            <td>${parseFloat(c.score).toFixed(4)}</td>
            <td title="${(c.reasoning || '').replace(/"/g, '&quot;')}">${reasoningTruncated}</td>
        `;

        row.addEventListener('click', () => {
            renderTargetProfile(c.candidate_id);
            // Highlight active row
            if (activeRow) activeRow.classList.remove('active-row');
            row.classList.add('active-row');
            activeRow = row;
        });

        tbody.appendChild(row);
    });

    // Auto-select first row
    if (tbody.firstChild) {
        tbody.firstChild.classList.add('active-row');
        activeRow = tbody.firstChild;
    }
}

function renderTargetProfile(id) {
    const candidate = candidates.find(c => c.candidate_id === id);
    const log = logs[id];

    if (!candidate) return;

    document.getElementById("candidateTitle").innerText = id;
    document.getElementById("rankBadge").innerText = "Rank #" + candidate.rank;
    document.getElementById("scoreBadge").innerText = "Score: " + parseFloat(candidate.score).toFixed(4);
    document.getElementById("reasoning").innerText = candidate.reasoning || "No reasoning available.";

    if (log && log.scores) {
        const s = log.scores;

        updateMetricPillar("techBar", "techVal", s.technical_fit);
        updateMetricPillar("seniorBar", "seniorVal", s.seniority_fit);
        updateMetricPillar("foundingBar", "foundingVal", s.founding_fit);
        updateMetricPillar("signalBar", "signalVal", s.signal_score || 0);
        updateMetricPillar("educationBar", "educationVal", s.education_fit || 0);
        updateMetricPillar("evidenceBar", "evidenceVal", s.evidence_strength);
        updateMetricPillar("hiringBar", "hiringVal", s.hiring_probability);
        updateMetricPillar("behaviorBar", "behaviorVal", s.behavioral_fit);

        drawRadarChart(s);

        // Signal breakdown
        if (log.signal_breakdown) {
            renderSignalBreakdown(log.signal_breakdown);
        }
    }

    // Highlight the row in the table
    const rows = document.querySelectorAll("#candidateTable tbody tr");
    rows.forEach(row => {
        if (row.dataset.candidateId === id) {
            if (activeRow) activeRow.classList.remove('active-row');
            row.classList.add('active-row');
            activeRow = row;
        }
    });
}

function updateMetricPillar(barId, valId, value) {
    const normalizedPercent = value <= 1 ? Math.round(value * 100) : Math.round(value);
    const el = document.getElementById(valId);
    if (el) el.innerText = normalizedPercent + "%";
    const bar = document.getElementById(barId);
    if (bar) bar.style.width = normalizedPercent + "%";
}

function renderSignalBreakdown(breakdown) {
    const section = document.getElementById("signalBreakdownSection");
    const grid = document.getElementById("signalGrid");
    if (!section || !grid) return;

    section.style.display = "block";
    grid.innerHTML = "";

    const labels = {
        engagement: "Engagement",
        market_demand: "Market Demand",
        availability: "Availability",
        platform_trust: "Platform Trust",
        hiring_track_record: "Hiring Track",
        technical_signals: "Tech Signals",
    };

    for (const [key, label] of Object.entries(labels)) {
        if (breakdown[key] !== undefined) {
            const item = document.createElement("div");
            item.className = "signal-item";
            const pct = Math.round(breakdown[key] * 100);
            item.innerHTML = `
                <p class="signal-label">${label}</p>
                <p class="signal-val">${pct}%</p>
            `;
            grid.appendChild(item);
        }
    }
}

// ---- Radar Chart (Pure Canvas — no external libraries) ----
function drawRadarChart(scores) {
    const canvas = document.getElementById("radarChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const dpr = window.devicePixelRatio || 1;
    const displayW = 380;
    const displayH = 380;
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.width = displayW + "px";
    canvas.style.height = displayH + "px";
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, displayW, displayH);

    const labels = ["Technical", "Seniority", "Founding", "Signal", "Education", "Evidence", "Hiring", "Behavioral"];
    const values = [
        scores.technical_fit || 0,
        scores.seniority_fit || 0,
        scores.founding_fit || 0,
        scores.signal_score || 0,
        scores.education_fit || 0,
        scores.evidence_strength || 0,
        scores.hiring_probability || 0,
        scores.behavioral_fit || 0,
    ];

    const cx = displayW / 2;
    const cy = displayH / 2;
    const maxR = 140;
    const n = labels.length;
    const angleStep = (2 * Math.PI) / n;
    const startAngle = -Math.PI / 2;

    // Draw grid rings
    const rings = [0.25, 0.5, 0.75, 1.0];
    rings.forEach(ring => {
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = startAngle + i * angleStep;
            const x = cx + Math.cos(angle) * maxR * ring;
            const y = cy + Math.sin(angle) * maxR * ring;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Draw axis lines
    for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw data polygon
    ctx.beginPath();
    values.forEach((val, i) => {
        const angle = startAngle + i * angleStep;
        const r = maxR * Math.min(1, val);
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();

    // Fill with gradient
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
    gradient.addColorStop(0, "rgba(59, 130, 246, 0.25)");
    gradient.addColorStop(1, "rgba(139, 92, 246, 0.08)");
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.strokeStyle = "rgba(59, 130, 246, 0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw data points
    values.forEach((val, i) => {
        const angle = startAngle + i * angleStep;
        const r = maxR * Math.min(1, val);
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "#3b82f6";
        ctx.fill();
        ctx.strokeStyle = "#1d4ed8";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });

    // Draw labels
    ctx.font = "500 11px Inter, sans-serif";
    ctx.fillStyle = "#9aa0b0";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    labels.forEach((label, i) => {
        const angle = startAngle + i * angleStep;
        const lr = maxR + 22;
        let x = cx + Math.cos(angle) * lr;
        let y = cy + Math.sin(angle) * lr;

        // Adjust alignment for edge labels
        if (Math.cos(angle) > 0.3) ctx.textAlign = "left";
        else if (Math.cos(angle) < -0.3) ctx.textAlign = "right";
        else ctx.textAlign = "center";

        const pct = Math.round((values[i] || 0) * 100);
        ctx.fillStyle = "#9aa0b0";
        ctx.fillText(label, x, y);
        ctx.fillStyle = "#e8eaed";
        ctx.fillText(pct + "%", x, y + 14);
    });

    ctx.textAlign = "start"; // Reset
}