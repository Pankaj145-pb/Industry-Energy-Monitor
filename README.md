# ⚡ Industrial Energy Monitoring & Intelligence Web Application

An **offline-first, privacy-preserving industrial energy monitoring, anomaly detection, predictive forecasting, and local LLM intelligence system** tailored for heavy industrial manufacturing facilities (such as cement plants on 132 kV transmission feeders).

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Engineering Approach & Methodology](#-engineering-approach--methodology)
3. [Models & Machine Learning Architecture](#-models--machine-learning-architecture)
4. [Project Directory Structure](#-project-directory-structure)
5. [Prerequisites & Installation](#-prerequisites--installation)
6. [Running the Application](#-running-the-application)
7. [Features & Dashboard Capabilities](#-features--dashboard-capabilities)
8. [Sample Queries for the Local AI Agent](#-sample-queries-for-the-local-ai-agent)

---

## 🎯 Project Overview

Heavy manufacturing plants (e.g. cement processing mills, kilns, heavy crushers) consume large amounts of electrical power and operate under strict utility grid compliance contracts. They face:
- **Telemetry noise & timestamp jitter** from edge IoT loggers.
- **Heavy utility tariff penalties** for low power factor ($|PF| < 0.85$), harmonic distortion ($THD > 5\%$), and exceeding contracted maximum demand limits.
- **Strict data privacy requirements**: telemetry cannot be dispatched to cloud APIs.

This application provides a **100% offline, on-premise pipeline** that ingests raw telemetry (over 100 electrical parameters), aligns data to a continuous 8,760-hour annual grid, detects electrical events, generates 24-hour predictive load demand curves, and provides grounded local AI diagnostics using a local LLM (`llama3.2:3b`).

---

## 🧠 Engineering Approach & Methodology

The application follows a clean, modular architecture with separation of concerns:

### 1. Dynamic Schema Discovery & Ingestion (`src/cleaning.py`)
- **Schema-Agnostic Profiling**: Discovers all columns dynamically (tested on 101 channels from `ESP_Demo_Data_1Year.xlsx`) and categorizes them into timestamps, numeric measurement channels, and categorical flags.
- **No Hardcoded Aliases**: Uses true raw sensor names (e.g. `watts_total`) without synthetic duplicate columns (`load_kw`).
- **Strict 8,760-Hour Grid Reindexing**: Enforces an exact hourly timeline ($[00:00 \text{ Jan 1} \rightarrow 23:00 \text{ Dec 31}]$), resampling sub-hourly intervals (15-min / 30-min) to whole hours.
- **Controlled Gap Interpolation**: Uses linear interpolation only for small telemetry dropouts ($\le 3$ hours). Extended blackouts ($> 3$ hours) remain unfilled to preserve genuine plant downtime records.

### 2. Industrial Event Detection Engine (`src/detector.py`)
Scans multiple correlated electrical channels simultaneously and groups contiguous hourly runs into discrete events:
- **Outage Detection**: Flags zero-power conditions ($\le 0.05\text{ kW}$ / $100\text{ W}$) while retaining incoming grid voltage (`vll_avg`) to differentiate between **external utility blackouts** ($V = 0$) and **internal plant breaker trips / shutdowns** ($V \approx 138\text{ kV}$, $P = 0$).
- **Load Trip / Drop Detection**: Flags sudden power drops $> 40\%$ within an hour ($\frac{P_{t-1} - P_t}{P_{t-1}} \times 100 > 40\%$), typical of mill jams or drive trips.
- **Poor Power Factor**: Flags $|PF| < 0.85$ under active load to prevent utility billing surcharges.
- **Harmonics (THD)**: Detects voltage total harmonic distortion exceeding the IEEE 519 limit ($> 5\%$).
- **Severity Rating**: Classifies each event into **Critical**, **High**, **Medium**, or **Low** based on duration and physical magnitude.

### 3. Physics-Aware Rollup & Audit Reporting (`src/reporter.py`)
- Distinguishes **Power** (instantaneous rate in Watts/kW) from **Energy** (accumulated consumption in Wh/kWh integrated over 1-hour time intervals: $\sum P \times 1.0\text{ h}$).
- Exports Daily Summary (CSV), Monthly Summary (CSV), and ReportLab-generated Monthly Executive PDF reports.

---

## 🤖 Models & Machine Learning Architecture

The system utilizes two distinct types of local, offline models:

### A. 24-Hour Predictive Load Demand Model (`src/forecaster.py`)
Predicts the next 24 hours of facility power demand ($P_{t+1} \dots P_{t+24}$) using a dedicated **Random Forest Regressor** to help engineers anticipate peak demand overshoots and plan generator dispatch.
- **Feature Engineering**:
  - *Cyclical Time Features*: Sine and cosine encoding for hour-of-day ($24\text{h}$ period), day-of-week ($7\text{d}$), and month ($12\text{m}$) to preserve periodic continuity ($23:00 \rightarrow 00:00$).
  - *Autoregressive Lags*: $t-1, t-2, t-3$, same-hour yesterday ($t-24$), day-before-yesterday ($t-48$), and same-hour last week ($t-168$).
  - *Rolling Statistics*: Rolling means and standard deviations over 6-hour and 24-hour windows.
- **Model Architecture**:
  - **Random Forest Regressor**: Multi-tree ensemble (`n_estimators=100`, `max_depth=12`, parallelized with `n_jobs=-1`) capturing complex non-linear equipment interactions, shift changes, and thermal kiln cycles.
- **Holdout Validation & Backtesting**:
  - Evaluates performance on an unseen 24-hour holdout window before forecasting.
  - Generates industry-standard evaluation metrics: **MAPE** (Mean Absolute Percentage Error), **MAE** (Mean Absolute Error in MW and Watts), and **RMSE** (Root Mean Squared Error).

### B. Local Offline LLM Agent (`src/rag_agent.py`)
- **Model**: **`llama3.2:3b`** running on a local Ollama daemon (`http://localhost:11434`).
- **Grounded RAG Approach**: Extracts annual metrics (peak, min, average, exact peak timestamp), complete monthly breakdowns, and detected event counts into a structured JSON context.
- **Strict Anti-Hallucination Prompting**: Instructs the model to answer exclusively from verified metrics, report power in both Watts and MW, and explicitly refuse to guess numbers when data is absent.
- **Root-Cause Anomaly Diagnoser**: Analyzes any detected anomaly event and outputs a 3-bullet engineering assessment covering physical causes (e.g. VFD harmonics, capacitor bank trips), severity, and immediate maintenance recommendations.

---

## 📂 Project Directory Structure

```text
Assignment/
├── .gitignore               # Ignores .venv/, data/
├── requirements.txt         # Project dependencies (pandas, streamlit, plotly, scikit-learn, reportlab, ollama)
├── README.md                # Project documentation and architecture guide
├── setup.ps1                # Automated PowerShell environment setup script
├── data/                    # Private telemetry files (Excel / CSV) - Git-ignored
│   └── ESP_Demo_Data_1Year.xlsx  # 1-year 101-channel hourly industrial feeder data
├── src/                     # Core analytical backend
│   ├── __init__.py
│   ├── config.py            # Centralized constants, paths, and thresholds
│   ├── cleaning.py          # Dynamic schema discovery, 8,760h grid reindexing & interpolation
│   ├── detector.py          # Industrial event detection engine & severity scoring
│   ├── forecaster.py        # 24-hour ML load forecasting pipeline & holdout validation
│   ├── rag_agent.py         # Local Ollama llama3.2:3b grounded Q&A and diagnostics
│   └── reporter.py          # Daily/Monthly rollups and ReportLab PDF generator
└── ui/
    └── app.py               # Streamlit multi-tab reactive web dashboard
```

---

## ⚙️ Prerequisites & Installation

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.14)
- **Local Ollama** (Required for the offline LLM advisor):
  1. Download and install from [ollama.com](https://ollama.com/).
  2. Pull the 3B model:
     ```bash
     ollama pull llama3.2:3b
     ```
  3. Ensure the service is running:
     ```bash
     ollama serve
     ```

### 2. Virtual Environment & Dependency Setup

#### **Windows (PowerShell)**:
```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Linux / macOS (Bash)**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run ui/app.py
```
Or use the virtual environment executable directly:
```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🖥️ Features & Dashboard Capabilities

1. **📊 Load Overview Tab**:
   - Ingests Excel (`.xlsx`, `.xls`, `.xlsm`) or `.csv` files (uploaded files are automatically saved to the local `data/` directory).
   - Dynamic channel selector for primary metric (e.g. `watts_total`) and timestamp column.
   - Top KPI cards: Timeline span, raw completeness %, final completeness %, total events flagged, peak and average telemetry.
   - Interactive Plotly chart with raw recorded points, cleaned trace, and red triangle markers for Critical/High events.
2. **⚠️ Anomaly & Event Detection Tab**:
   - Summary metric cards for total events, outages, load trips (>40%), poor PF (<0.85), and high THD (>5%).
   - Multi-select filters for Event Type and Severity.
   - Structured tabular event log with duration, severity tags, and details.
   - Dedicated `📥 Download Event Log CSV` export.
3. **📈 24h Load Forecasting Tab**:
   - Trains a Random Forest Regressor on historical load curves.
   - Visualizes recent actual load alongside the future 24-hour predicted trajectory.
   - Backtest validation metrics: Predicted Peak, Predicted Average, MAPE (%), MAE, and RMSE.
   - Hourly 24-hour prediction schedule table with `📥 Download 24h Forecast CSV`.
4. **🤖 Local Ollama Assistant Tab**:
   - Status indicator for local Ollama connection and `llama3.2:3b` availability.
   - Grounded Q&A interface for facility managers.
   - Interactive Anomaly Root-Cause Diagnoser to generate 3-bullet technical maintenance briefs for any detected event.
5. **📑 Report Generation Tab**:
   - `📥 Download Daily Summary (CSV)`: Day-by-day peak, avg, min load, and total Wh.
   - `📥 Download Monthly Summary (CSV)`: Month-by-month energy and peak table.
   - `📄 Download Monthly Summary (PDF)`: Formal audit PDF report with high-level KPIs and monthly breakdowns.

---

## 💬 Sample Queries for the Local AI Agent

You can test the agent in the **Local Ollama Assistant** tab with queries such as:

| Category | Example Question |
| :--- | :--- |
| **Peak & Averages** | *"What was our peak active power demand in 2025 and when did it occur?"* |
| **Monthly Breakdown** | *"What was the peak load in March?"* |
| **Comparative Query** | *"Compare the peak load between January and June."* |
| **Outage & Trips** | *"How many outages and load trips were detected across the year?"* |
| **Anti-Hallucination** | *"What was our dollar cost of electricity in November?"* *(Refuses to guess; notes tariffs are not in telemetry)* |
