# Industrial Energy Monitoring Web Application

An offline-first, private industrial energy monitoring and analytics system built with Python, Streamlit, Plotly, Scikit-learn/Statsmodels, and local Ollama LLM integration.

---

## Project Structure

```text
Assignment/
├── .gitignore               # Ignores .venv/, data/, models/, outputs/
├── requirements.txt         # Project dependencies
├── README.md                # Quickstart instructions
├── setup.ps1                # PowerShell automated setup script
├── data/                    # Private telemetry data (Excel/CSV files) - Git ignored
├── models/                  # Local model weights and cache - Git ignored
├── outputs/                 # Generated PDF audit reports & CSVs - Git ignored
├── src/                     # Backend core logic
│   ├── __init__.py
│   ├── config.py            # Centralized settings & path constants
│   ├── cleaning.py          # Data ingestion & timestamp reindexing
│   ├── detector.py          # Anomaly & outage detection
│   ├── forecaster.py        # 24-hour predictive load forecasting
│   ├── rag_agent.py         # Local Ollama / pandas intelligence agent
│   └── reporter.py          # ReportLab PDF & CSV exports
└── ui/
    └── app.py               # Streamlit multi-tab dashboard entrypoint
```

---

## Prerequisites

- **Python 3.10+** (Tested on Python 3.14)
- **Local Ollama** (Optional, for offline AI assistance): [Download Ollama](https://ollama.com/)

---

## Quickstart Setup & Execution

### 1. Manual Setup (Terminal)

#### **Windows (PowerShell)**:
```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run ui/app.py
```

#### **Linux / macOS (Bash)**:
```bash
# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run ui/app.py
```

---

### 2. Automated Script Setup (Windows PowerShell)

Run the included automated setup script:
```powershell
.\setup.ps1
```

---

## Features

- **📊 Load Overview:** Ingest factory telemetry (.xlsx, .csv), clean timestamp continuity, and visualize load profiles.
- **⚠️ Anomaly & Outage Detection:** Identify severe drops, total power outages, and statistical peak overshoots.
- **📈 24h Load Forecasting:** Predict next-day load curves to plan generator dispatch and avoid maximum demand penalties.
- **🤖 Local Ollama Assistant:** Query private energy logs completely offline using locally hosted LLMs.
- **📑 Report Generation:** Export compliance and audit summaries directly to formatted PDF reports.

