"""Local Ollama-powered industrial energy intelligence agent.

Uses the local `ollama` Python library and `llama3.2:3b` to provide offline,
strictly grounded Q&A and root-cause event explanations for the cement plant's telemetry.
"""
from typing import Dict, Any, Optional, Union, List
import json
import re
import numpy as np
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


def find_matching_events(
    events_df: pd.DataFrame,
    query: str,
    df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """Scans detected industrial events to find records matching a queried date or event type,

    enriching them with concurrent telemetry (operating load MW and voltage kV).
    """
    if events_df is None or events_df.empty:
        return []

    q_lower = query.lower()
    matched_month_num = None
    for idx, m in enumerate(MONTH_NAMES, start=1):
        if re.search(r'\b' + m + r'\b', q_lower):
            matched_month_num = idx
            break

    day_match = re.search(r'\b([0-2]?[0-9]|3[01])(?:st|nd|rd|th)?\b', q_lower)
    matched_day = int(day_match.group(1)) if day_match else None

    event_type_keywords = {
        "power factor": ["power factor", "pf"],
        "outage": ["outage", "blackout", "shutdown"],
        "trip": ["trip", "drop", "load trip"],
        "harmonic": ["harmonic", "thd"]
    }

    has_date_query = bool(matched_month_num)

    matched = []
    for _, row in events_df.iterrows():
        start_ts = pd.to_datetime(row["start_time"])
        end_ts = pd.to_datetime(row["end_time"])
        event_type = str(row.get("event_type", ""))
        severity = str(row.get("severity", ""))

        is_match = False
        if matched_month_num:
            month_match = (start_ts.month == matched_month_num or end_ts.month == matched_month_num)
            if matched_day:
                day_match = (start_ts.day == matched_day or end_ts.day == matched_day)
                if month_match and day_match:
                    is_match = True
            elif month_match:
                is_match = True

        if not is_match and not has_date_query:
            for k, synonyms in event_type_keywords.items():
                if any(syn in q_lower for syn in synonyms) and any(syn in event_type.lower() for syn in synonyms):
                    is_match = True

        if is_match:
            item = {
                "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_hours": int(row.get("duration_hours", 1)),
                "event_type": event_type,
                "severity": severity,
                "details": row.get("details", {})
            }
            if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
                try:
                    slice_sub = df.loc[start_ts:end_ts]
                    if not slice_sub.empty:
                        watts_col = [c for c in slice_sub.columns if "watt" in c.lower() or "power" in c.lower()]
                        if watts_col:
                            mean_w = float(pd.to_numeric(slice_sub[watts_col[0]], errors="coerce").mean())
                            if not np.isnan(mean_w):
                                item["operating_load_watts"] = round(mean_w, 2)
                                item["operating_load_mw"] = round(mean_w / 1e6, 3)
                        volt_col = [c for c in slice_sub.columns if "vll" in c.lower() or "volt" in c.lower()]
                        if volt_col:
                            mean_v = float(pd.to_numeric(slice_sub[volt_col[0]], errors="coerce").mean())
                            if not np.isnan(mean_v):
                                item["operating_voltage_kv"] = round(mean_v / 1e3, 1)
                except Exception:
                    pass
            matched.append(item)

    return matched[:10]


def format_telemetry_for_llm(context: Dict[str, Any]) -> str:
    """Serializes the telemetry context dictionary into clean, flat, human-readable text

    with pre-calculated megawatts and explicit zero/coverage indicators to prevent hallucinations.
    """
    lines = []

    # 1. Facility & Channel Overview
    channel = context.get("primary_metric_channel", "Unknown")
    total_rec = context.get("total_records", 0)
    lines.append("=== FACILITY & TELEMETRY OVERVIEW ===")
    lines.append("Grid Connection: 132 kV Utility Feeder (Cement Manufacturing Facility)")
    lines.append(f"Primary Monitored Channel: {channel}")
    lines.append(f"Total Monitored Timeline Records: {total_rec:,} hours")
    lines.append("")

    # 2. Annual Baseline Statistics (pre-calculated MW)
    peak_w = context.get("annual_peak_active_power", 0.0)
    peak_mw = context.get("annual_peak_active_power_mw", round(peak_w / 1e6, 3))
    peak_ts = context.get("annual_peak_timestamp", "N/A")

    avg_w = context.get("annual_average_active_power", 0.0)
    avg_mw = context.get("annual_average_active_power_mw", round(avg_w / 1e6, 3))

    min_w = context.get("annual_min_active_power", 0.0)
    min_mw = context.get("annual_min_active_power_mw", round(min_w / 1e6, 3))

    lines.append("=== ANNUAL POWER METRICS ===")
    lines.append(f"Annual Peak Active Power: {peak_w:,.2f} W ({peak_mw:.3f} MW) recorded at {peak_ts}")
    lines.append(f"Annual Average Active Power: {avg_w:,.2f} W ({avg_mw:.3f} MW)")
    lines.append(f"Annual Minimum Active Power: {min_w:,.2f} W ({min_mw:.3f} MW)")
    lines.append("")

    # 3. Overall Detected Events Summary
    ev_summary = context.get("events_summary", {})
    total_ev = ev_summary.get("total_events", 0)
    lines.append("=== DETECTED INDUSTRIAL EVENTS SUMMARY ===")
    lines.append(f"Total Detected Events in Entire Dataset: {total_ev}")
    if total_ev > 0:
        for k, v in ev_summary.items():
            if k not in ("total_events", "severity_breakdown"):
                lines.append(f"- {k}: {v} event(s)")
        sev = ev_summary.get("severity_breakdown", {})
        if sev:
            sev_str = ", ".join(f"{sk}: {sv}" for sk, sv in sev.items())
            lines.append(f"- Severity Breakdown: {sev_str}")
    else:
        lines.append("- No industrial events detected (0 events).")
    lines.append("")

    # 4. Specific Queried Date Events (Critical for single-date queries)
    lines.append("=== SPECIFIC QUERIED DATE EVENTS ===")
    queried_ev = context.get("queried_events")
    if queried_ev is not None and len(queried_ev) > 0:
        lines.append(f"Specific Queried Date Events: {len(queried_ev)} event(s) recorded:")
        for i, ev in enumerate(queried_ev, 1):
            etype = ev.get("event_type", "Event")
            sev = ev.get("severity", "Unknown")
            start = ev.get("start_time", "N/A")
            end = ev.get("end_time", "N/A")
            dur = ev.get("duration_hours", 1)
            lines.append(f"  [{i}] {etype} (Severity: {sev})")
            lines.append(f"      Start Time: {start} | End Time: {end} | Duration: {dur} hour(s)")

            if "operating_load_watts" in ev or "operating_load_mw" in ev:
                w_val = ev.get("operating_load_watts", 0.0)
                mw_val = ev.get("operating_load_mw", round(w_val / 1e6, 3))
                lines.append(f"      Operating Active Power: {w_val:,.2f} W ({mw_val:.3f} MW)")
            if "operating_voltage_kv" in ev:
                lines.append(f"      Operating Voltage: {ev['operating_voltage_kv']} kV")

            det = ev.get("details", {})
            det_parts = [f"{dk.replace('_', ' ').title()}: {dv}" for dk, dv in det.items() if dk != "duration_hours"]
            if det_parts:
                lines.append(f"      Measurements: {', '.join(det_parts)}")
    else:
        lines.append("Specific Queried Date Events: None recorded (0 events).")
    lines.append("")

    # 5. Targeted Queried Month (if specific month was queried)
    spec_period = context.get("specific_queried_period")
    if spec_period:
        m_name = spec_period.get("queried_month", "Unknown Month")
        p_w = spec_period.get("peak_active_power_watts", 0.0)
        p_mw = spec_period.get("peak_active_power_mw", round(p_w / 1e6, 3))
        p_ts = spec_period.get("peak_timestamp", "N/A")
        a_w = spec_period.get("average_active_power_watts", 0.0)
        a_mw = spec_period.get("average_active_power_mw", round(a_w / 1e6, 3))
        m_w = spec_period.get("min_active_power_watts", 0.0)
        m_mw = spec_period.get("min_active_power_mw", round(m_w / 1e6, 3))
        cov_hrs = spec_period.get("total_hours", 0)

        lines.append(f"=== SPECIFIC QUERIED MONTH ({m_name.upper()}) ===")
        lines.append(f"Month: {m_name}")
        lines.append(f"Peak Active Power: {p_w:,.2f} W ({p_mw:.3f} MW) recorded at {p_ts}")
        lines.append(f"Average Active Power: {a_w:,.2f} W ({a_mw:.3f} MW)")
        lines.append(f"Minimum Active Power: {m_w:,.2f} W ({m_mw:.3f} MW)")
        lines.append(f"Dataset coverage: {cov_hrs} hours (total monitored hours in month, NOT outage duration)")
        lines.append("")

    # 6. Monthly Summaries (Full Year)
    monthly = context.get("monthly_summary", {})
    lines.append("=== MONTHLY DATASET COVERAGE & PEAK LOAD SUMMARY ===")
    if monthly:
        for m_name, m_info in monthly.items():
            p_w = m_info.get("peak_load_watts", 0.0)
            p_mw = m_info.get("peak_load_mw", round(p_w / 1e6, 3))
            p_ts = m_info.get("peak_timestamp", "N/A")
            cov = m_info.get("total_records_hours", 0)
            lines.append(
                f"- {m_name}: Peak {p_w:,.2f} W ({p_mw:.3f} MW) at {p_ts} | "
                f"Dataset coverage: {cov} hours (total monitored hours in month, NOT outage duration)"
            )
    else:
        lines.append("- No monthly breakdown available.")

    return "\n".join(lines)


def format_json_as_bullets(obj: Any, indent: int = 0) -> str:
    """Recursively formats a JSON dictionary or list into clean Markdown bullet points."""
    lines = []
    prefix = " " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_title = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}* **{key_title}**:")
                sub = format_json_as_bullets(v, indent + 2)
                if sub:
                    lines.append(sub)
            else:
                lines.append(f"{prefix}* **{key_title}**: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(format_json_as_bullets(item, indent))
            else:
                lines.append(f"{prefix}* {item}")
    else:
        lines.append(f"{prefix}* {obj}")
    return "\n".join([line for line in lines if line.strip()])


def clean_response_text(text: str) -> str:
    """Ensures LLM response is strictly clean human-readable prose and Markdown,

    converting any leaked JSON blocks or raw dictionaries into structured bullet points.
    """
    if not text:
        return ""

    # 1. Catch and convert markdown code blocks containing JSON
    def _replace_codeblock(match):
        raw_snippet = match.group(1).strip()
        try:
            parsed = json.loads(raw_snippet)
            if isinstance(parsed, (dict, list)):
                return "\n" + format_json_as_bullets(parsed) + "\n"
        except Exception:
            pass
        return ""

    cleaned = re.sub(r"```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```", _replace_codeblock, text)

    # 2. Catch and convert inline/standalone raw JSON objects
    def _replace_inline_json(match):
        raw_snippet = match.group(0).strip()
        try:
            parsed = json.loads(raw_snippet)
            if isinstance(parsed, (dict, list)) and len(parsed) > 0:
                return "\n" + format_json_as_bullets(parsed) + "\n"
        except Exception:
            pass
        return raw_snippet

    cleaned = re.sub(
        r"\{[^{}]*(?:\"start_time\"|\"event_type\"|\"duration_hours\"|\"severity\"|\"details\")[^{}]*\}",
        _replace_inline_json,
        cleaned
    )

    # 3. Strip any stray raw JSON bracket artifacts
    cleaned = re.sub(r'^\s*[\{\}]\s*$', '', cleaned, flags=re.MULTILINE)

    # 4. Clean up multiple empty lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 5. Check if model leaked internal prompt metadata or couldn't answer
    lower_c = cleaned.lower()
    if (
        "grounding context" in lower_c
        or "specific_queried_period" in lower_c
        or "events_summary" in lower_c
        or "telemetry metrics" in lower_c
        or "provided data does not contain" in lower_c
        or "unable to answer" in lower_c
        or "not enough data" in lower_c
        or "not sufficient information" in lower_c
        or "none recorded" in lower_c
        or "not contain information" in lower_c
        or "not contain that information" in lower_c
    ) and (
        "not contain" in lower_c
        or "unable to answer" in lower_c
        or "does not include" in lower_c
        or "not enough data" in lower_c
        or "not specify" in lower_c
        or "none recorded" in lower_c
    ):
        return "There is not enough data to answer the question."

    return cleaned


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
            "events_summary": {},
            "queried_events": []
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

    # Match specific queried events if user query references date or event type
    queried_events = []
    if events_df is not None and not events_df.empty and user_query:
        queried_events = find_matching_events(events_df, user_query, df=df)

    context = {
        "total_records": int(total_records),
        "primary_metric_channel": str(resolved_power),
        "annual_peak_active_power": round(peak_power, 2),
        "annual_peak_active_power_mw": round(peak_power / 1e6, 3),
        "annual_peak_timestamp": peak_timestamp_str,
        "annual_min_active_power": round(min_power, 2),
        "annual_min_active_power_mw": round(min_power / 1e6, 3),
        "annual_average_active_power": round(avg_power, 2),
        "annual_average_active_power_mw": round(avg_power / 1e6, 3),
        "events_summary": events_summary,
        "queried_events": queried_events,
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
    serialized_telemetry = format_telemetry_for_llm(context)

    system_prompt = (
        "You are an expert Industrial Energy Analyst and High-Voltage Power Systems Engineer "
        "monitoring a cement manufacturing plant connected to a 132 kV utility grid.\n\n"
        "STRICT GROUNDING & OPERATING RULES:\n"
        "1. STRICT FALLBACK RULE: If the user asks about an event on a specific date (e.g., a specific day or date), and 'Specific Queried Date Events' states 'None recorded (0 events)' or does not contain that specific event/date, you MUST output ONLY this exact sentence and nothing else:\n"
        "   \"There is not enough data to answer the question.\"\n"
        "   Do NOT use monthly summaries, dataset coverage hours, or annual peaks to guess, approximate, or fabricate events for missing specific dates.\n"
        "2. SEPARATION OF RECORDED FACTS AND ENGINEERING HYPOTHESES:\n"
        "   - First, state the verified recorded facts (exact start time, end time, duration, and measured electrical values) directly from 'Specific Queried Date Events'.\n"
        "   - ONLY if the user explicitly asks 'Why' AND abnormal electrical metrics (such as power factor drop, voltage sag, or harmonic surge) are explicitly listed in 'Specific Queried Date Events', you may provide physical hypotheses under a separate header named '### Possible Engineering Causes' (e.g., APFC capacitor bank trips, heavy inductive raw/ball mill starts, or grid disturbances).\n"
        "   - If the user did not ask 'Why', or if abnormal readings are not recorded, do not invent physical causes.\n"
        "3. DATASET COVERAGE DISTINCTION: 'Dataset coverage' represents the number of recorded telemetry hours (e.g., 744 hours in May). It is NEVER an outage duration, downtime, or event length.\n"
        "4. PRE-CALCULATED UNITS: Use the exact pre-calculated Watts and Megawatts (MW) as given in the text. Do not perform manual arithmetic or calculate current/amperes.\n"
        "5. FORMATTING: Output strictly in natural English prose and Markdown bullets. NEVER output raw JSON, curly braces { }, quotation marks around keys, or code blocks.\n\n"
        "--- VERIFIED PLANT TELEMETRY ---\n"
        f"{serialized_telemetry}\n"
        "--------------------------------"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = cl.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0}
        )
        content = ""
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "No response content received.")
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content
        else:
            content = str(response)

        return clean_response_text(content)
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

    # Format event dict into flat text representation
    lines = ["=== ANOMALY EVENT DETAILS ==="]
    for k, v in event_dict.items():
        key_title = str(k).replace("_", " ").title()
        lines.append(f"{key_title}: {v}")
    flat_event_text = "\n".join(lines)

    prompt = (
        "You are an industrial electrical engineer investigating an anomaly at a cement manufacturing plant "
        "(132 kV grid connection, major loads include raw mills, kiln drives, and cement ball mills).\n\n"
        f"{flat_event_text}\n\n"
        "Task: In 2-3 clear, technical bullet points:\n"
        "1. Explain what physical event likely occurred at the facility (e.g. main breaker trip, ball mill stoppage, capacitor bank de-energization, harmonic distortion from VFDs).\n"
        "2. Note the electrical severity impact.\n"
        "3. Provide one immediate preventive recommendation for the electrical maintenance team.\n"
        "CRITICAL FORMAT RULE: Output ONLY natural human-readable text with bullet points. Do NOT output raw JSON, curly braces { }, or code blocks.\n"
        "Do not invent external facts; ground explanations in typical heavy industrial cement operations."
    )

    messages = [
        {"role": "system", "content": "You are a senior industrial electrical reliability engineer. Provide direct, factual assessments without JSON."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = cl.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0}
        )
        content = ""
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "No explanation generated.")
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content
        else:
            content = str(response)

        return clean_response_text(content)
    except ConnectionError:
        return (
            "⚠️ Local Ollama connection failed: Unable to connect to http://localhost:11434. "
            "Please ensure Ollama is running (`ollama serve`)."
        )
    except Exception as e:
        return f"⚠️ Error generating event explanation with local Ollama: {str(e)}"
