"""Local Ollama-powered industrial energy intelligence agent.

Uses the local `ollama` Python library and `llama3.2:3b` to provide offline,
strictly grounded Q&A and root-cause event explanations for the cement plant's telemetry.
"""
from typing import Dict, Any, Optional, Union, List
import json
import re
import pandas as pd
import ollama
from src.config import OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from src.detector import detect_industrial_events, resolve_channel, POWER_CANDIDATES


MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def get_ollama_client(base_url: str = OLLAMA_BASE_URL) -> ollama.Client:
    """Initializes and returns an Ollama Client pointing to the local host."""
    return ollama.Client(host=base_url)


def check_ollama_status(model: str = DEFAULT_OLLAMA_MODEL, client: Optional[ollama.Client] = None) -> Dict[str, Any]:
    """Verifies that the local Ollama instance is reachable and the required model is loaded."""
    cl = client or get_ollama_client()
    try:
        models_resp = cl.list()
        model_names = []
        if hasattr(models_resp, "models"):
            for m in models_resp.models:
                name = getattr(m, "model", None) or getattr(m, "name", "")
                model_names.append(name)
        elif isinstance(models_resp, dict):
            for m in models_resp.get("models", []):
                model_names.append(m.get("model") or m.get("name", ""))

        target_found = any(model in name for name in model_names)
        return {
            "online": True,
            "target_model_available": target_found,
            "available_models": model_names,
            "error": None
        }
    except Exception as e:
        return {
            "online": False,
            "target_model_available": False,
            "available_models": [],
            "error": str(e)
        }


def extract_monthly_breakdown(df: pd.DataFrame, power_col: str) -> Dict[str, Any]:
    """Extracts monthly peak, average, and min load metrics from the DataFrame."""
    monthly_stats = {}
    if not isinstance(df.index, pd.DatetimeIndex) or power_col not in df.columns:
        return monthly_stats

    # Group by year-month and month name
    grouped = df.groupby([df.index.year, df.index.month])
    for (year, month_num), group in grouped:
        series = pd.to_numeric(group[power_col], errors="coerce").dropna()
        if series.empty:
            continue
        month_name = pd.Timestamp(year=year, month=month_num, day=1).strftime("%B")
        peak_idx = series.idxmax()
        peak_val = float(series.loc[peak_idx])
        peak_ts = peak_idx.strftime("%Y-%m-%d %H:%M:%S") if isinstance(peak_idx, pd.Timestamp) else str(peak_idx)

        monthly_stats[month_name] = {
            "month_number": int(month_num),
            "year": int(year),
            "peak_load_watts": round(peak_val, 2),
            "peak_load_mw": round(peak_val / 1e6, 2),
            "peak_timestamp": peak_ts,
            "average_load_watts": round(float(series.mean()), 2),
            "average_load_mw": round(float(series.mean()) / 1e6, 2),
            "min_load_watts": round(float(series.min()), 2),
            "min_load_mw": round(float(series.min()) / 1e6, 2),
            "total_records_hours": len(series)
        }
    return monthly_stats


def query_specific_monthly_stats(df: pd.DataFrame, power_col: str, query: str) -> Optional[Dict[str, Any]]:
    """If user asks about a specific month or day, extracts exact targeted slice metrics."""
    q_lower = query.lower()
    matched_month = None
    for idx, m in enumerate(MONTH_NAMES, start=1):
        if re.search(r'\b' + m + r'\b', q_lower):
            matched_month = (m.capitalize(), idx)
            break

    if not matched_month or not isinstance(df.index, pd.DatetimeIndex) or power_col not in df.columns:
        return None

    m_name, m_num = matched_month
    sub_df = df[df.index.month == m_num]
    if sub_df.empty:
        return None

    series = pd.to_numeric(sub_df[power_col], errors="coerce").dropna()
    if series.empty:
        return None

    peak_idx = series.idxmax()
    peak_val = float(series.loc[peak_idx])
    peak_ts = peak_idx.strftime("%Y-%m-%d %H:%M:%S") if isinstance(peak_idx, pd.Timestamp) else str(peak_idx)

    return {
        "queried_month": m_name,
        "peak_active_power_watts": round(peak_val, 2),
        "peak_active_power_mw": round(peak_val / 1e6, 2),
        "peak_timestamp": peak_ts,
        "average_active_power_watts": round(float(series.mean()), 2),
        "average_active_power_mw": round(float(series.mean()) / 1e6, 2),
        "min_active_power_watts": round(float(series.min()), 2),
        "min_active_power_mw": round(float(series.min()) / 1e6, 2),
        "total_hours": len(series)
    }


def generate_data_context(
    df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    power_col: Optional[str] = None,
    user_query: Optional[str] = None
) -> Dict[str, Any]:
    """Accepts the cleaned Pandas DataFrame (containing the cement plant's 2025 hourly energy data)

    and extracts annual and monthly grounding metrics to prevent LLM hallucinations:
      - Total hourly records (expected 8,760).
      - Peak active power (watts_total) and its exact timestamp.
      - Minimum and average active power.
      - Full monthly breakdown (peak, avg, min per month).
      - Targeted query-specific slice (if user mentions March, July, etc.).
      - Summary count of detected outages or events if available.
    """
    if df.empty:
        return {
            "total_records": 0,
            "peak_active_power": 0.0,
            "peak_timestamp": "N/A",
            "min_active_power": 0.0,
            "average_active_power": 0.0,
            "power_unit": "Watts",
            "events_summary": {}
        }

    # Resolve active power column
    resolved_power = resolve_channel(df, power_col, POWER_CANDIDATES, fallback_contains="watt") or \
                     resolve_channel(df, None, None, fallback_contains="power")

    if not resolved_power or resolved_power not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        resolved_power = numeric_cols[0] if numeric_cols else df.columns[0]

    series = pd.to_numeric(df[resolved_power], errors="coerce").dropna()

    total_records = len(df)
    min_power = float(series.min()) if not series.empty else 0.0
    avg_power = float(series.mean()) if not series.empty else 0.0

    if not series.empty:
        peak_idx = series.idxmax()
        peak_power = float(series.loc[peak_idx])
        if isinstance(peak_idx, pd.Timestamp):
            peak_timestamp_str = peak_idx.strftime("%Y-%m-%d %H:%M:%S")
        elif "timestamp" in [c.lower() for c in df.columns]:
            ts_actual = [c for c in df.columns if c.lower() == "timestamp"][0]
            val = df.loc[peak_idx, ts_actual]
            peak_timestamp_str = pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(val) else str(peak_idx)
        else:
            peak_timestamp_str = str(peak_idx)
    else:
        peak_power = 0.0
        peak_timestamp_str = "N/A"

    # Monthly breakdown extraction
    monthly_data = extract_monthly_breakdown(df, resolved_power)

    # Targeted query match (e.g. user asks specifically about "March")
    targeted_query_context = None
    if user_query:
        targeted_query_context = query_specific_monthly_stats(df, resolved_power, user_query)

    # Event count summary
    if events_df is None or events_df.empty:
        events_df = detect_industrial_events(df, power_col=resolved_power)

    events_summary = {}
    if events_df is not None and not events_df.empty:
        events_summary["total_events"] = int(len(events_df))
        for event_type, count in events_df["event_type"].value_counts().items():
            events_summary[str(event_type)] = int(count)
        if "severity" in events_df.columns:
            events_summary["severity_breakdown"] = {
                str(k): int(v) for k, v in events_df["severity"].value_counts().items()
            }
    else:
        events_summary["total_events"] = 0

    context = {
        "total_records": int(total_records),
        "primary_metric_channel": str(resolved_power),
        "annual_peak_active_power": round(peak_power, 2),
        "annual_peak_active_power_mw": round(peak_power / 1e6, 2),
        "annual_peak_timestamp": peak_timestamp_str,
        "annual_min_active_power": round(min_power, 2),
        "annual_min_active_power_mw": round(min_power / 1e6, 2),
        "annual_average_active_power": round(avg_power, 2),
        "annual_average_active_power_mw": round(avg_power / 1e6, 2),
        "events_summary": events_summary,
        "monthly_summary": monthly_data
    }

    if targeted_query_context:
        context["specific_queried_period"] = targeted_query_context

    return context


def ask_energy_agent(
    user_query: str,
    df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    client: Optional[ollama.Client] = None
) -> str:
    """Builds a rigorous system prompt instructing llama3.2:3b to act as an expert industrial

    energy analyst, grounded strictly on the verified metrics to prevent hallucination.
    """
    cl = client or get_ollama_client()
    context = generate_data_context(df, events_df=events_df, user_query=user_query)

    system_prompt = (
        "You are an expert Industrial Energy Analyst and High-Voltage Power Systems Engineer "
        "monitoring a cement manufacturing plant connected to a 132 kV utility grid.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. You must answer the user's question strictly using the verified Grounding Context below.\n"
        "2. If the user asks about a specific month (e.g., March, July), refer to 'specific_queried_period' or 'monthly_summary'.\n"
        "3. Do NOT hallucinate, guess, extrapolate, or invent numerical values or timestamps.\n"
        "4. If the provided data genuinely does not contain sufficient information to answer the question, "
        "explicitly state: 'The verified energy telemetry data does not contain that information.'\n"
        "5. Always present active power in both Watts and Megawatts (MW = Watts / 1,000,000) for clarity.\n"
        "6. Keep responses concise, direct, professional, and actionable for plant managers.\n\n"
        f"--- VERIFIED GROUNDING CONTEXT ---\n"
        f"{json.dumps(context, indent=2)}\n"
        "----------------------------------"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = cl.chat(model=model, messages=messages)
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "No response content received.")
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        return str(response)
    except ConnectionError:
        return (
            "⚠️ Local Ollama connection failed: Unable to connect to http://localhost:11434. "
            "Please ensure Ollama is running (`ollama serve`)."
        )
    except Exception as e:
        return f"⚠️ Error querying local Ollama model '{model}': {str(e)}"


def explain_detected_event(
    event_row: Union[pd.Series, Dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    client: Optional[ollama.Client] = None
) -> str:
    """Takes a specific detected anomaly row (start time, end time, event type, severity)

    and prompts llama3.2:3b to generate a short, plain-English root-cause explanation
    tailored for an industrial cement plant feeder.
    """
    cl = client or get_ollama_client()

    if isinstance(event_row, pd.Series):
        event_dict = event_row.to_dict()
    else:
        event_dict = dict(event_row)

    for k, v in event_dict.items():
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            event_dict[k] = str(v)

    prompt = (
        "You are an industrial electrical engineer investigating an anomaly at a cement manufacturing plant "
        "(132 kV grid connection, major loads include raw mills, kiln drives, and cement ball mills).\n\n"
        f"ANOMALY EVENT DETAILS:\n"
        f"{json.dumps(event_dict, indent=2)}\n\n"
        "Task: In 2-3 clear, technical bullet points:\n"
        "1. Explain what physical event likely occurred at the facility (e.g. main breaker trip, ball mill stoppage, capacitor bank de-energization, harmonic distortion from VFDs).\n"
        "2. Note the electrical severity impact.\n"
        "3. Provide one immediate preventive recommendation for the electrical maintenance team.\n"
        "Do not invent external facts; ground explanations in typical heavy industrial cement operations."
    )

    messages = [
        {"role": "system", "content": "You are a senior industrial electrical reliability engineer. Provide direct, factual assessments."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = cl.chat(model=model, messages=messages)
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "No explanation generated.")
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        return str(response)
    except ConnectionError:
        return (
            "⚠️ Local Ollama connection failed: Unable to connect to http://localhost:11434. "
            "Please ensure Ollama is running (`ollama serve`)."
        )
    except Exception as e:
        return f"⚠️ Error generating event explanation with local Ollama: {str(e)}"
