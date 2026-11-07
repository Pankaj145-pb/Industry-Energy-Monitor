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
from src.reporter import EnergyReporter
from src.rag_agent import ask_energy_agent, explain_detected_event, check_ollama_status
from src.forecaster import LoadForecaster

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
if active_source is None:
    for k in ["clean_df", "audit_summary", "events_df", "raw_df", "active_filename"]:
        if k in st.session_state:
            st.session_state[k] = None
    st.info("👆 Upload a file or pick a sample dataset to begin.")
    st.stop()
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
        col5.metric("Peak Telemetry", f"{summary.get('peak_metric_value', summary.get('peak_load_kw', 0)):,.1f}", f"Avg: {summary.get('average_metric_value', summary.get('average_load_kw', 0)):,.1f}")

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
                st.write(f"- Metric Range: `{summary.get('min_metric_value', summary.get('min_load_kw', 0)):,.2f}` to `{summary.get('peak_metric_value', summary.get('peak_load_kw', 0)):,.2f}`")

        # Interactive Plotly Time Series Chart
        st.subheader(f"📈 Reindexed Profile: {summary['selected_metric_column']}")
        
        plot_df = clean_df.reset_index()
        fig = go.Figure()

        metric_col = summary['selected_metric_column']
        metric_clean_col = f"{metric_col}_clean"

        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df[metric_clean_col],
            mode="lines",
            name=f"{metric_col} (Cleaned)",
            line=dict(color="#1f77b4", width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df[metric_col],
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
                    y=event_points[metric_clean_col],
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
    if clean_df is not None and summary is not None:
        metric_col = summary["selected_metric_column"]
        clean_col = f"{metric_col}_clean" if f"{metric_col}_clean" in clean_df.columns else metric_col

        st.caption(f"Predicting next 24 hours of load trajectory for `{metric_col}` using cyclical hour/day feature engineering and machine learning.")

        f_control1, f_control2, f_control3 = st.columns([1, 1, 1])
        with f_control1:
            model_choice = st.selectbox("Model Architecture", options=["Ridge Regression", "Random Forest Regressor"], index=0)
        with f_control2:
            model_key = "ridge" if model_choice == "Ridge Regression" else "rf"
            run_forecast_btn = st.button("🚀 Generate 24-Hour Forecast", type="primary")
        with f_control3:
            st.caption("Includes 24-hour holdout validation (MAE, RMSE, MAPE).")

        # Session state cache for forecast results
        forecast_cache_key = f"fc_{metric_col}_{model_key}"
        if run_forecast_btn or (forecast_cache_key in st.session_state):
            if run_forecast_btn:
                with st.spinner(f"Training {model_choice} and extrapolating future 24-hour demand..."):
                    forecaster = LoadForecaster(horizon_hours=24, model_type=model_key)
                    fc_df, recent_df, fc_metrics = forecaster.fit_and_predict_next_24h(clean_df[clean_col], target_name=metric_col)
                    st.session_state[forecast_cache_key] = (fc_df, recent_df, fc_metrics)
            
            fc_df, recent_df, fc_metrics = st.session_state[forecast_cache_key]

            # 1. Performance Metric Cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Predicted Peak", f"{fc_metrics['forecast_peak_mw']:.2f} MW", f"{fc_metrics['forecast_peak_time'][-8:]}")
            m2.metric("Predicted Average", f"{fc_metrics['forecast_avg_mw']:.2f} MW")
            m3.metric("Backtest MAPE", f"{fc_metrics['mape_pct']:.2f}%", "Validation Error")
            m4.metric("Backtest MAE", f"{fc_metrics['mae_mw']:.3f} MW", f"{fc_metrics['mae']:,.0f} W")
            m5.metric("Backtest RMSE", f"{fc_metrics['rmse_mw']:.3f} MW")

            st.divider()

            # 2. Interactive Forecast Plotly Chart
            fig_fc = go.Figure()

            # Recent actual load (last 48-72h)
            fig_fc.add_trace(go.Scatter(
                x=recent_df.index,
                y=recent_df["actual_load_mw"],
                mode="lines",
                name="Recent Actual Load (MW)",
                line=dict(color="#1f77b4", width=2)
            ))

            # Future 24-hour forecast
            fig_fc.add_trace(go.Scatter(
                x=fc_df.index,
                y=fc_df["predicted_load_mw"],
                mode="lines+markers",
                name="Predicted Next 24h (MW)",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                marker=dict(size=5, color="#ff7f0e")
            ))

            # Vertical separator line indicating forecast horizon start
            fig_fc.add_vline(
                x=pd.to_datetime(fc_metrics["last_historical_timestamp"]),
                line_width=1.5,
                line_dash="dot",
                line_color="#7f7f7f",
                annotation_text="Forecast Start",
                annotation_position="top left"
            )

            fig_fc.update_layout(
                title=f"24-Hour Load Demand Forecast — {metric_col} ({fc_metrics['model_type']})",
                xaxis_title="Timeline",
                yaxis_title="Load Demand (MW)",
                template="plotly_white",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
                height=450
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # 3. Forecast Table & CSV Export
            st.markdown("#### 📋 Hourly Forecast Schedule")
            table_col1 = st.columns(2)

            with table_col1[0]:
                display_fc = fc_df.reset_index()
                display_fc["timestamp"] = display_fc["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                display_fc.rename(columns={
                    "timestamp": "Target Hour",
                    "predicted_load": f"Predicted Load ({metric_col})",
                    "predicted_load_mw": "Predicted Load (MW)"
                }, inplace=True)
                st.dataframe(display_fc, use_container_width=True, height=280)

    else:
        st.info("Upload energy telemetry to enable predictive load forecasting.")

with tab_rag:
    st.subheader("🤖 Local LLM Energy Advisor (Ollama — llama3.2:3b)")
    
    # Check Ollama connection status
    ollama_stat = check_ollama_status()
    if ollama_stat["online"]:
        st.success(f"🟢 Local Ollama Online | Target Model: `llama3.2:3b` {'(Ready)' if ollama_stat['target_model_available'] else '(Not Found)'}")
    else:
        st.warning("⚠️ Local Ollama is offline. Ensure `ollama serve` is running at `http://localhost:11434`.")

    if clean_df is not None and summary is not None:
        st.caption(f"Context grounded on `{summary['selected_metric_column']}` ({summary['final_completeness_percentage']}% completeness, {summary['expected_grid_hours']:,}h).")
        
        user_prompt = st.text_input(
            "Ask a question about facility energy usage:",
            placeholder="e.g. What was our peak active power demand and when did it occur?"
        )
        if st.button("Query Local Model", type="primary"):
            if user_prompt.strip():
                with st.spinner("Querying local llama3.2:3b model with grounded context..."):
                    response_text = ask_energy_agent(user_prompt, clean_df, events_df=events_df)
                    st.markdown("### 💬 Energy Advisor Response")
                    st.info(response_text)
            else:
                st.warning("Please enter a question to ask the local energy advisor.")

        # Event Root-Cause Explanation Assistant
        if events_df is not None and not events_df.empty:
            st.divider()
            st.subheader("🔍 AI Event Root-Cause Diagnoser")
            st.caption("Select any detected industrial anomaly to generate an AI technical breakdown:")
            
            event_options = [
                f"{i}: {row['event_type']} | {row['start_time']} | Severity: {row['severity']}"
                for i, row in events_df.iterrows()
            ]
            selected_event_idx_str = st.selectbox("Select Anomaly Event to Diagnose", options=event_options)
            if st.button("Generate Root-Cause Explanation"):
                selected_idx = int(selected_event_idx_str.split(":")[0])
                target_event = events_df.iloc[selected_idx]
                with st.spinner("Analyzing electrical waveform patterns with llama3.2:3b..."):
                    explanation_text = explain_detected_event(target_event, clean_df)
                    st.markdown("#### 🛠️ AI Engineering Diagnostic")
                    st.markdown(explanation_text)
    else:
        st.info("Upload energy telemetry to ground the local AI agent with verified plant metrics.")

with tab_reports:
    st.subheader("📑 Export Audit Artifacts")
    if clean_df is not None and summary is not None:
        metric_col = summary["selected_metric_column"]
        daily_df = EnergyReporter.generate_daily_summary(clean_df, metric_col)
        monthly_df = EnergyReporter.generate_monthly_summary(clean_df, metric_col)

        rep_col1, rep_col2, rep_col3 = st.columns(3)

        with rep_col1:
            st.markdown("#### 📅 Daily Summary")
            st.caption("Aggregated daily peak, average, and min load.")
            daily_csv = io.StringIO()
            daily_df.to_csv(daily_csv)
            st.download_button(
                label="📥 Download Daily Summary (CSV)",
                data=daily_csv.getvalue(),
                file_name=f"daily_energy_summary_{metric_col}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with rep_col2:
            st.markdown("#### 📊 Monthly Summary")
            st.caption("Monthly energy consumption and peak profile.")
            monthly_csv = io.StringIO()
            monthly_df.to_csv(monthly_csv)
            st.download_button(
                label="📥 Download Monthly Summary (CSV)",
                data=monthly_csv.getvalue(),
                file_name=f"monthly_energy_summary_{metric_col}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            monthly_pdf_bytes = EnergyReporter.generate_monthly_pdf_report(summary, monthly_df, metric_col)
            st.download_button(
                label="📄 Download Monthly Summary (PDF)",
                data=monthly_pdf_bytes,
                file_name=f"monthly_energy_summary_{metric_col}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with rep_col3:
            st.markdown("#### ⚠️ Industrial Events")
            st.caption("All detected outages, trips, PF, and THD events.")
            if events_df is not None and not events_df.empty:
                events_export_csv = io.StringIO()
                events_df.to_csv(events_export_csv, index=False)
                st.download_button(
                    label="📥 Download Events (CSV)",
                    data=events_export_csv.getvalue(),
                    file_name="detected_industrial_events.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No industrial events recorded to export.")
    else:
        st.info("Upload energy telemetry to generate executive audit reports.")
