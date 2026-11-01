"""Main Streamlit entry point with a multi-tab layout for industrial energy monitoring."""
import sys
from pathlib import Path

# Ensure src module is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.config import OUTPUTS_DIR, DATA_DIR
from src.cleaning import DataCleaner
from src.detector import EnergyAnomalyDetector
from src.forecaster import LoadForecaster
from src.rag_agent import LocalEnergyRAGAgent
from src.reporter import EnergyReporter

st.set_page_config(
    page_title="Industrial Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Industrial Energy Monitoring & Intelligence Dashboard")
st.caption("Offline real-time energy telemetry, anomaly detection, predictive load forecasting, and local AI agent.")

# Sidebar Configuration
st.sidebar.header("Data Source & Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Telemetry (Excel / CSV)",
    type=["xlsx", "xls", "csv"],
    help="Upload private factory submeter or main feeder logs."
)

# Tab Layout
tab_overview, tab_anomalies, tab_forecast, tab_rag, tab_reports = st.tabs([
    "📊 Load Overview",
    "⚠️ Anomaly & Outage Detection",
    "📈 24h Load Forecasting",
    "🤖 Local Ollama Assistant",
    "📑 Report Generation"
])

with tab_overview:
    st.subheader("Historical Load Profile")
    if uploaded_file is not None:
        st.success(f"Loaded: {uploaded_file.name}")
        # Placeholder view
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Please upload an energy telemetry file (.xlsx/.csv) in the sidebar to visualize trends.")

with tab_anomalies:
    st.subheader("Spike & Outage Identification")
    st.write("Configured thresholds for immediate alerts on machine idling, sudden dropouts, and peak overshoots.")

with tab_forecast:
    st.subheader("Predictive 24-Hour Load Demand")
    st.write("Anticipate peak factory load demand and potential maximum demand violation windows.")

with tab_rag:
    st.subheader("Local LLM Energy Advisor (Ollama)")
    user_prompt = st.text_input("Ask a question about facility energy usage:", placeholder="e.g. Which shifts showed the highest peak loads?")
    if st.button("Query Local Model"):
        st.info("Connecting to local Ollama instance...")

with tab_reports:
    st.subheader("Export Audit Artifacts")
    st.write("Generate downloadable PDF executive summaries and CSV log extracts.")

