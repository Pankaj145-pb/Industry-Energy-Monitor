"""Main Streamlit entry point with a multi-tab layout for industrial energy monitoring."""
import sys
from pathlib import Path
import io

# Ensure src module is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import OUTPUTS_DIR, DATA_DIR, OUTAGE_THRESHOLD_KW
from src.cleaning import EnergyDataProcessor, get_column_inventory
from src.detector import detect_industrial_events, resolve_channel, POWER_CANDIDATES, VOLTAGE_CANDIDATES, PF_CANDIDATES, THD_CANDIDATES

st.set_page_config(
    page_title="Industrial Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Industrial Energy Monitoring & Intelligence Dashboard")
st.caption("Offline real-time energy telemetry, anomaly & event detection, predictive load forecasting, and local AI agent.")

# Initialize session state for processed dataset and metadata
if "clean_df" not in st.session_state:
    st.session_state["clean_df"] = None
if "audit_summary" not in st.session_state:
    st.session_state["audit_summary"] = None
if "events_df" not in st.session_state:
    st.session_state["events_df"] = None
if "active_filename" not in st.session_state:
    st.session_state["active_filename"] = None
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

# Sidebar Configuration
st.sidebar.header("📁 Data Source & Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload Telemetry (Excel / CSV)",
    type=["xlsx", "xls", "xlsm", "csv"],
    help="Upload private factory submeter or feeder logs."
)

# Optional quick-load from data/ folder if no file is uploaded yet
sample_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.csv"))
selected_sample = None
if not uploaded_file and sample_files:
    sample_options = ["None"] + [f.name for f in sample_files]
    chosen = st.sidebar.selectbox("Or choose a sample dataset from data/:", sample_options)
    if chosen != "None":
        selected_sample = DATA_DIR / chosen

# Load raw dataframe once to inspect all columns
active_source = uploaded_file or selected_sample
raw_df = None

if active_source is not None:
    source_name = uploaded_file.name if uploaded_file else selected_sample.name
    # Cache raw data in session state if file changed
    if st.session_state.get("active_filename") != source_name or st.session_state.get("raw_df") is None:
        processor = EnergyDataProcessor()
        if hasattr(active_source, "read"):
            if source_name.endswith(".csv"):
                raw_df = pd.read_csv(active_source)
            else:
                raw_df = pd.read_excel(active_source)
        else:
            raw_df = processor.load_excel(active_source)
        st.session_state["raw_df"] = raw_df
        st.session_state["active_filename"] = source_name
    else:
        raw_df = st.session_state["raw_df"]

# Dynamic Column Selectors & Cleaning Settings in Sidebar
if raw_df is not None:
    col_inventory = get_column_inventory(raw_df)

    st.sidebar.header("🔍 Dynamic Column Mapping")
    st.sidebar.caption(f"Discovered **{col_inventory['total_columns_count']}** columns ({len(col_inventory['numeric_columns'])} numeric channels).")

    all_cols = col_inventory["all_columns"]
    numeric_cols = col_inventory["numeric_columns"] if col_inventory["numeric_columns"] else all_cols

    # --- Timestamp shortlist (not all 101) ---
    ts_candidates = []
    for c in all_cols:
        name = str(c).strip().lower()
        if name in ("ts", "time", "date", "datetime", "timestamp") or "time" in name or "date" in name:
            ts_candidates.append(c)
            continue
        try:
            sample = raw_df[c].dropna().head(20)
            if not sample.empty and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8:
                ts_candidates.append(c)
        except Exception:
            pass
    if not ts_candidates:
        ts_candidates = all_cols
    if col_inventory["timestamp_column"] in ts_candidates:
        ts_candidates = [col_inventory["timestamp_column"]] + [c for c in ts_candidates if c != col_inventory["timestamp_column"]]
    selected_ts_col = st.sidebar.selectbox("Timestamp Column", options=ts_candidates, index=0)

    # --- Primary load shortlist (not all 90) ---
    LOAD_LIKE = ("watt", "power", "load", "kw", "mw", "demand")
    EXCLUDE = ("received", "delivered", "wh_", "vah_", "varh_", "temp", "humidity", "thd", "harmonic", "angle", "unbal", "pf_", "frequency", "_ok", "pending", "pct", "phase", "why", "reset")
    load_candidates = [c for c in numeric_cols
                       if any(k in c.lower() for k in LOAD_LIKE)
                       and not any(e in c.lower() for e in EXCLUDE)]
    if not load_candidates:
        load_candidates = numeric_cols
    if "watts_total" in load_candidates:
        load_candidates = ["watts_total"] + [c for c in load_candidates if c != "watts_total"]
    elif col_inventory["primary_metric_column"] in load_candidates:
        load_candidates = [col_inventory["primary_metric_column"]] + [c for c in load_candidates if c != col_inventory["primary_metric_column"]]
    selected_metric_col = st.sidebar.selectbox("Primary Load / Active Power", options=load_candidates, index=0)
    if st.sidebar.checkbox("Show all numeric columns for primary", value=False):
        selected_metric_col = st.sidebar.selectbox("Primary Load (full list)", options=numeric_cols, index=numeric_cols.index(selected_metric_col) if selected_metric_col in numeric_cols else 0)

    # --- Auto-detect voltage / PF / THD ---
    detected_v = resolve_channel(raw_df, None, VOLTAGE_CANDIDATES, fallback_contains="volt") or \
                 resolve_channel(raw_df, None, None, fallback_contains="vll")
    detected_pf = resolve_channel(raw_df, None, PF_CANDIDATES, fallback_contains="pf")
    detected_thd = resolve_channel(raw_df, None, THD_CANDIDATES, fallback_contains="thd")

    with st.sidebar.expander("⚡ Electrical Diagnostic Channels", expanded=False):
        v_options = ["(Auto / None)"] + [c for c in numeric_cols if c != selected_metric_col]
        v_idx = v_options.index(detected_v) if detected_v in v_options else 0
        selected_voltage_col = st.selectbox("Voltage Channel", options=v_options, index=v_idx, key="v_chan")
        selected_v = None if selected_voltage_col == "(Auto / None)" else selected_voltage_col

        pf_options = ["(Auto / None)"] + [c for c in numeric_cols if c != selected_metric_col]
        pf_idx = pf_options.index(detected_pf) if detected_pf in pf_options else 0
        selected_pf_col = st.selectbox("Power Factor Channel", options=pf_options, index=pf_idx, key="pf_chan")
        selected_pf = None if selected_pf_col == "(Auto / None)" else selected_pf_col

        thd_options = ["(Auto / None)"] + [c for c in numeric_cols if c != selected_metric_col]
        thd_idx = thd_options.index(detected_thd) if detected_thd in thd_options else 0
        selected_thd_col = st.selectbox("Harmonics (THD) Channel", options=thd_options, index=thd_idx, key="thd_chan")
        selected_thd = None if selected_thd_col == "(Auto / None)" else selected_thd_col

    extra_channels_to_aggregate = []
    for cand in [selected_v or detected_v, selected_pf or detected_pf, selected_thd or detected_thd]:
        if cand and cand in raw_df.columns and cand != selected_metric_col and cand not in extra_channels_to_aggregate:
            extra_channels_to_aggregate.append(cand)

    with st.sidebar.expander("⚙️ Detection Thresholds & Grid Settings", expanded=False):
        max_gap_interp = st.slider("Max Gap Interpolation (Hours)", 1, 12, 3)
        outage_threshold_w = st.number_input("Outage Power Threshold (W)", min_value=0.0, max_value=500000.0, value=50000.0, step=1000.0,
            help="Normal plant ~12,000,000 W. Below this = outage.")
        outage_threshold_kw = outage_threshold_w / 1000.0
        st.caption(f"= {outage_threshold_kw:.1f} kW / {outage_threshold_w/1e6:.3f} MW")
        drop_threshold_pct = st.slider("Load Trip Drop Threshold (%)", 20, 90, 40, 5)
        pf_threshold_val = st.slider("Poor PF Threshold", 0.50, 0.95, 0.85, 0.01)
        thd_threshold_val = st.slider("High THD Threshold (%)", 2.0, 15.0, 5.0, 0.5)

    processor = EnergyDataProcessor(max_interp_gap=max_gap_interp, outage_threshold_kw=outage_threshold_kw)

    with st.spinner("Processing telemetry, reindexing continuous grid, and detecting events..."):
        clean_df, summary = processor.process_and_reindex_grid(
            raw_df, timestamp_col=selected_ts_col, value_col=selected_metric_col,
            additional_cols=extra_channels_to_aggregate
        )

        def _clean(name):
            if not name:
                return None
            return f"{name}_clean" if f"{name}_clean" in clean_df.columns else (name if name in clean_df.columns else None)

        events_df = detect_industrial_events(
            clean_df,
            power_col=_clean(selected_metric_col),
            voltage_col=_clean(selected_v or detected_v),
            pf_col=_clean(selected_pf or detected_pf),
            thd_col=_clean(selected_thd or detected_thd),
            power_threshold_zero=outage_threshold_w,
            drop_threshold_pct=float(drop_threshold_pct),
            pf_threshold=float(pf_threshold_val),
            thd_threshold_pct=float(thd_threshold_val)
        )

        st.session_state["clean_df"] = clean_df
        st.session_state["audit_summary"] = summary
        st.session_state["events_df"] = events_df

# Tab Layout
tab_overview, tab_anomalies, tab_forecast, tab_rag, tab_reports = st.tabs([
    "📊 Load Overview",
    "⚠️ Anomaly & Event Detection",
    "📈 24h Load Forecasting",
    "🤖 Local Ollama Assistant",
    "📑 Report Generation"
])

clean_df = st.session_state.get("clean_df")
summary = st.session_state.get("audit_summary")
events_df = st.session_state.get("events_df")

with tab_overview:
    if clean_df is not None and summary is not None:
        st.markdown(f"### 📋 Telemetry & Grid Quality Audit (`{summary['selected_metric_column']}`)")

        # KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Timeline Span", f"{summary['expected_grid_hours']:,}h", f"{summary['start_timestamp'][:10]} to {summary['end_timestamp'][:10]}")
        col2.metric("Raw Completeness", f"{summary['raw_completeness_percentage']:.2f}%", f"{summary['raw_missing_hours']}h missing")
        col3.metric("Final Completeness", f"{summary['final_completeness_percentage']:.2f}%", f"+{summary['interpolated_gap_hours']}h recovered", delta_color="normal")
        col4.metric("Total Events Flagged", f"{len(events_df):,}" if events_df is not None else "0", f"{len(events_df[events_df['severity'].isin(['High', 'Critical'])])} Critical/High" if events_df is not None and not events_df.empty else "None")
        col5.metric("Peak Telemetry", f"{summary['peak_load_kw']:,.1f}", f"Avg: {summary['average_load_kw']:,.1f}")

        st.divider()

        # Detailed expandable data audit metrics
        with st.expander("🔍 Complete Dataset & Channel Inventory", expanded=False):
            audit_cols = st.columns(3)
            with audit_cols[0]:
                st.write("**Dataset Inventory:**")
                st.write(f"- Total Columns Discovered: `{summary['total_columns_count']}`")
                st.write(f"- Numeric Channels: `{len(summary['numeric_columns'])}`")
                st.write(f"- Categorical/Metadata: `{len(summary['categorical_columns'])}`")
            with audit_cols[1]:
                st.write("**Data Completeness:**")
                st.write(f"- Expected Grid Duration: `{summary['expected_grid_hours']:,}` hours")
                st.write(f"- Recorded Hours: `{summary['raw_recorded_hours']:,}` hours")
                st.write(f"- Interpolated Gaps: `{summary['interpolated_gap_hours']}` hours")
                st.write(f"- Remaining Missing: `{summary['remaining_missing_hours']}` hours")
            with audit_cols[2]:
                st.write("**Active Metric Profile:**")
                st.write(f"- Selected Channel: `{summary['selected_metric_column']}`")
                st.write(rf"- Outage Hours ($\le$ {summary['outage_threshold_kw']}): `{summary['total_outage_hours']}` hours")
                st.write(f"- Metric Range: `{summary['min_load_kw']:,.2f}` to `{summary['peak_load_kw']:,.2f}`")

        # Interactive Plotly Time Series Chart
        st.subheader(f"📈 Reindexed Profile: {summary['selected_metric_column']}")
        
        plot_df = clean_df.reset_index()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["load_kw_clean"],
            mode="lines",
            name=f"{summary['selected_metric_column']} (Cleaned)",
            line=dict(color="#1f77b4", width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["load_kw"],
            mode="markers",
            marker=dict(size=2, color="#2ca02c", opacity=0.4),
            name="Raw Recorded Points"
        ))

        # Overlay detected critical events onto chart
        if events_df is not None and not events_df.empty:
            crit_events = events_df[events_df["severity"].isin(["Critical", "High"])]
            if not crit_events.empty:
                event_points = plot_df[plot_df["timestamp"].isin(crit_events["start_time"])]
                fig.add_trace(go.Scatter(
                    x=event_points["timestamp"],
                    y=event_points["load_kw_clean"],
                    mode="markers",
                    marker=dict(size=8, color="#d62728", symbol="triangle-up"),
                    name="Critical / High Events"
                ))

        fig.update_layout(
            title=f"Telemetry Profile — {summary['selected_metric_column']}",
            xaxis_title="Timestamp",
            yaxis_title=summary['selected_metric_column'],
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data Preview & Quick Export
        st.subheader("📄 Telemetry Data Preview")
        preview_col1, preview_col2 = st.columns([2, 1])
        
        with preview_col1:
            st.dataframe(plot_df.head(24), use_container_width=True)
            
        with preview_col2:
            st.write("**Export Cleaned Grid:**")
            csv_buffer = io.StringIO()
            clean_df.to_csv(csv_buffer)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_buffer.getvalue(),
                file_name=f"cleaned_{summary['selected_metric_column']}_grid.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info("👈 Please upload an Excel (.xlsx / .csv) file in the sidebar to automatically clean, detect events, and visualize trends.")

with tab_anomalies:
    st.subheader("⚠️ Industrial Event Detection Engine Log")
    if events_df is not None and not events_df.empty:
        # Event Summary KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Events", len(events_df))
        kpi2.metric("Outages", len(events_df[events_df["event_type"] == "Outage"]))
        kpi3.metric("Load Trips (>40%)", len(events_df[events_df["event_type"] == "Load Trip / Drop"]))
        kpi4.metric("Poor PF (<0.85)", len(events_df[events_df["event_type"] == "Poor Power Factor"]))
        kpi5.metric("High THD (>5%)", len(events_df[events_df["event_type"] == "High Harmonics (THD)"]))

        st.divider()

        # Severity & Type Filtering
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            type_filter = st.multiselect(
                "Filter Event Type",
                options=list(events_df["event_type"].unique()),
                default=list(events_df["event_type"].unique())
            )
        with f_col2:
            sev_filter = st.multiselect(
                "Filter Severity",
                options=["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium", "Low"]
            )
        with f_col3:
            st.caption("Structured Event Log grouped by contiguous consecutive hourly duration.")

        filtered_events = events_df[
            (events_df["event_type"].isin(type_filter)) &
            (events_df["severity"].isin(sev_filter))
        ]

        # Color-coded severity styling function
        def style_severity(val):
            color_map = {
                "Critical": "background-color: #ffcccc; color: #900; font-weight: bold",
                "High": "background-color: #ffe6cc; color: #c60; font-weight: bold",
                "Medium": "background-color: #fff2cc; color: #8a6d3b",
                "Low": "background-color: #e6f2ff; color: #31708f"
            }
            return color_map.get(val, "")

        # Format details into readable string for table view
        display_events = filtered_events.copy()
        display_events["start_time"] = display_events["start_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_events["end_time"] = display_events["end_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_events["event_summary"] = display_events["details"].apply(
            lambda d: ", ".join([f"{k}: {v}" for k, v in d.items() if k not in ["mean_value", "min_value", "max_value"]])
        )

        st.dataframe(
            display_events[["start_time", "end_time", "duration_hours", "event_type", "severity", "event_summary"]],
            use_container_width=True
        )

        # Download event log
        event_csv = io.StringIO()
        filtered_events.to_csv(event_csv, index=False)
        st.download_button(
            label="📥 Download Event Log CSV",
            data=event_csv.getvalue(),
            file_name="industrial_events_log.csv",
            mime="text/csv"
        )
    elif clean_df is not None:
        st.success("✅ No industrial anomalies (Outages, Trips, Poor PF, High THD) detected under current threshold settings.")
    else:
        st.info("Upload energy telemetry to review detected industrial events.")

with tab_forecast:
    st.subheader("📈 24-Hour Predictive Load Demand")
    if clean_df is not None:
        st.write("24-hour predictive pipeline ready for training on the cleaned continuous grid.")
    else:
        st.info("Upload energy telemetry to enable predictive forecasting.")

with tab_rag:
    st.subheader("🤖 Local LLM Energy Advisor (Ollama)")
    if summary is not None:
        st.success(f"Context loaded for metric '{summary['selected_metric_column']}' ({summary['final_completeness_percentage']}% completeness).")
    user_prompt = st.text_input("Ask a question about facility energy usage:", placeholder="e.g. What was our maximum continuous outage duration?")
    if st.button("Query Local Model"):
        if summary is not None:
            st.info("Synthesizing context and querying local Ollama instance...")
        else:
            st.warning("Please upload energy data first to provide context for the local AI.")

with tab_reports:
    st.subheader("📑 Export Audit Artifacts")
    if summary is not None:
        st.write("Generate downloadable PDF executive summaries and CSV extracts based on the cleaned grid.")
    else:
        st.info("Upload energy telemetry to generate executive audit reports.")
