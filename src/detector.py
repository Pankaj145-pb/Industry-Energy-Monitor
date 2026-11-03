"""Industrial Event Detection Engine.

Scans cleaned telemetry DataFrames to identify:
  1. Power Outages: Zero active power and zero voltage.
  2. Load Trips / Severe Drops: >40% drop in active power within one hour.
  3. Poor Power Factor: PF < 0.85 (under running load).
  4. High Total Harmonic Distortion: THD > 5%.

Groups consecutive hourly anomalies into a structured event log with:
start_time, end_time, duration_hours, event_type, severity, and event_metrics.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Column candidate helpers
POWER_CANDIDATES = [
    "watts_total_clean", "watts_total", "load_kw_clean", "load_kw",
    "active_power", "power", "kw", "kwh", "demand",
]
VOLTAGE_CANDIDATES = [
    "vll_avg", "vln_avg", "voltage", "volts", "v_avg", "vll"
]
PF_CANDIDATES = [
    "true_pf_avg", "pf_avg", "power_factor", "pf", "true_pf"
]
THD_CANDIDATES = [
    "v_r_thd_pct", "v_thd_pct", "v_thd", "thd_pct", "thd", "i_r_thd_pct"
]


def resolve_channel(
    df: pd.DataFrame,
    preferred_col: Optional[str] = None,
    candidate_list: Optional[List[str]] = None,
    fallback_contains: Optional[str] = None
) -> Optional[str]:
    """Dynamically resolves a column from the DataFrame using explicit preference,

    candidate patterns, or case-insensitive keyword matches.
    """
    if preferred_col and preferred_col in df.columns:
        return preferred_col

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    # 1. Exact candidate match
    if candidate_list:
        for cand in candidate_list:
            cand_lower = cand.lower()
            if cand_lower in lower_map:
                return lower_map[cand_lower]

    # 2. Substring match
    if fallback_contains:
        kw = fallback_contains.lower()
        for c_lower, orig in lower_map.items():
            if kw in c_lower:
                return orig

    return None


def calculate_severity(
    event_type: str,
    duration_hours: int,
    metrics: Dict[str, Any]
) -> str:
    """Assigns severity rating: 'Low', 'Medium', 'High', or 'Critical'

    based on the nature of the electrical event, duration, and magnitude.
    """
    if event_type == "Outage":
        if duration_hours >= 6:
            return "Critical"
        elif duration_hours >= 2:
            return "High"
        else:
            return "Medium"

    elif event_type == "Load Trip / Drop":
        max_drop = metrics.get("max_drop_pct", 0.0)
        if max_drop >= 75.0 or duration_hours >= 3:
            return "Critical"
        elif max_drop >= 55.0 or duration_hours >= 2:
            return "High"
        else:
            return "Medium"

    elif event_type == "Poor Power Factor":
        min_pf = metrics.get("min_pf", 1.0)
        if min_pf < 0.65 or duration_hours >= 8:
            return "Critical"
        elif min_pf < 0.75 or duration_hours >= 4:
            return "High"
        elif min_pf < 0.80 or duration_hours >= 2:
            return "Medium"
        else:
            return "Low"

    elif event_type == "High Harmonics (THD)":
        max_thd = metrics.get("max_thd_pct", 0.0)
        if max_thd > 12.0 or duration_hours >= 12:
            return "Critical"
        elif max_thd > 8.0 or duration_hours >= 6:
            return "High"
        elif max_thd > 6.0 or duration_hours >= 3:
            return "Medium"
        else:
            return "Low"

    return "Medium"


def group_consecutive_hourly_events(
    condition_mask: pd.Series,
    timestamps: pd.Series,
    values: pd.Series,
    event_type: str,
    additional_series: Optional[Dict[str, pd.Series]] = None
) -> List[Dict[str, Any]]:
    """Groups contiguous runs of True condition flags into discrete industrial events."""
    events = []
    if condition_mask.empty or not condition_mask.any():
        return events

    # Label contiguous True groups
    group_ids = (~condition_mask).cumsum()[condition_mask]

    for _, group_indices in group_ids.groupby(group_ids).groups.items():
        sub_ts = timestamps.loc[group_indices]
        sub_vals = values.loc[group_indices]

        start_time = sub_ts.iloc[0]
        end_time = sub_ts.iloc[-1]
        duration_hours = len(group_indices)

        metrics: Dict[str, Any] = {
            "duration_hours": int(duration_hours),
            "mean_value": float(np.round(sub_vals.mean(), 2)) if not sub_vals.empty else 0.0,
            "min_value": float(np.round(sub_vals.min(), 2)) if not sub_vals.empty else 0.0,
            "max_value": float(np.round(sub_vals.max(), 2)) if not sub_vals.empty else 0.0,
        }

        # Event-specific metrics
        if event_type == "Outage":
            metrics["max_outage_load"] = metrics["max_value"]
        elif event_type == "Load Trip / Drop":
            # Track largest percentage drop
            metrics["max_drop_pct"] = float(np.round(sub_vals.max(), 2))
        elif event_type == "Poor Power Factor":
            metrics["min_pf"] = metrics["min_value"]
        elif event_type == "High Harmonics (THD)":
            metrics["max_thd_pct"] = metrics["max_value"]

        # Incorporate additional series if provided
        if additional_series:
            for name, s in additional_series.items():
                sub_s = s.loc[group_indices].dropna()
                if not sub_s.empty:
                    metrics[f"mean_{name}"] = float(np.round(sub_s.mean(), 2))

        severity = calculate_severity(event_type, duration_hours, metrics)

        events.append({
            "start_time": pd.to_datetime(start_time),
            "end_time": pd.to_datetime(end_time),
            "duration_hours": int(duration_hours),
            "event_type": event_type,
            "severity": severity,
            "details": metrics
        })

    return events


def detect_industrial_events(
    df: pd.DataFrame,
    power_col: Optional[str] = None,
    voltage_col: Optional[str] = None,
    pf_col: Optional[str] = None,
    thd_col: Optional[str] = None,
    power_threshold_zero: float = 0.05,
    voltage_threshold_zero: float = 1.0,
    drop_threshold_pct: float = 40.0,
    pf_threshold: float = 0.85,
    thd_threshold_pct: float = 5.0
) -> pd.DataFrame:
    """Scans the cleaned DataFrame to detect:

      1. Outages: zero active power and zero voltage (or near-zero threshold).
      2. Load trips/drops: >40% drop in active power in an hour.
      3. Poor power factor: PF < 0.85 (evaluated when power > zero).
      4. High harmonics: THD > 5%.

    Returns:
      pd.DataFrame: Structured event log sorted chronologically with columns:
        ['start_time', 'end_time', 'duration_hours', 'event_type', 'severity', 'details']
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "start_time", "end_time", "duration_hours", "event_type", "severity", "details"
        ])

    work_df = df.copy()

    # Determine Timestamp series
    if isinstance(work_df.index, pd.DatetimeIndex):
        timestamps = work_df.index.to_series()
    elif "timestamp" in [c.lower() for c in work_df.columns]:
        ts_actual = [c for c in work_df.columns if c.lower() == "timestamp"][0]
        timestamps = pd.to_datetime(work_df[ts_actual])
    else:
        # Fallback to integer sequence
        timestamps = pd.Series(pd.date_range("2025-01-01", periods=len(work_df), freq="1h"), index=work_df.index)

    # Resolve measurement channels
    p_col = resolve_channel(work_df, power_col, POWER_CANDIDATES, fallback_contains="watt") or \
            resolve_channel(work_df, None, None, fallback_contains="power")
    v_col = resolve_channel(work_df, voltage_col, VOLTAGE_CANDIDATES, fallback_contains="volt") or \
            resolve_channel(work_df, None, None, fallback_contains="vll") or \
            resolve_channel(work_df, None, None, fallback_contains="vln")
    pf_channel = resolve_channel(work_df, pf_col, PF_CANDIDATES, fallback_contains="pf")
    thd_channel = resolve_channel(work_df, thd_col, THD_CANDIDATES, fallback_contains="thd")

    all_events: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # 1. OUTAGE DETECTION
    # ---------------------------------------------------------
    if p_col is not None:
        power_series = pd.to_numeric(work_df[p_col], errors="coerce").fillna(0.0)
        power_zero = power_series <= power_threshold_zero

        # If voltage is tracked and drops to zero, it's a utility blackout;
        # if voltage remains high while power is 0, it's a plant shutdown/breaker trip.
        # Both represent zero-power outages for the facility.
        has_voltage = v_col is not None and v_col in work_df.columns
        if has_voltage:
            voltage_series = pd.to_numeric(work_df[v_col], errors="coerce").fillna(0.0)
            voltage_zero = voltage_series <= voltage_threshold_zero
            # Flag outage when power is 0 (distinguishing grid blackout vs plant trip in details)
            outage_mask = power_zero
        else:
            voltage_series = None
            outage_mask = power_zero

        outage_events = group_consecutive_hourly_events(
            condition_mask=outage_mask,
            timestamps=timestamps,
            values=power_series,
            event_type="Outage",
            additional_series={"voltage": voltage_series} if has_voltage else None
        )
        all_events.extend(outage_events)

    # ---------------------------------------------------------
    # 2. LOAD TRIPS / DROPS (>40% drop in an hour)
    # ---------------------------------------------------------
    if p_col is not None:
        power_series = pd.to_numeric(work_df[p_col], errors="coerce")
        # Hourly percentage drop: (P[t-1] - P[t]) / P[t-1] * 100
        prev_power = power_series.shift(1)
        
        # Only evaluate drop when previous power was substantial (above baseline threshold)
        significant_previous = prev_power > (power_threshold_zero * 10)
        pct_drop = ((prev_power - power_series) / prev_power.replace(0, np.nan)) * 100.0
        pct_drop = pct_drop.fillna(0.0)

        # Flag load trip (drop > 40%)
        trip_mask = significant_previous & (pct_drop >= drop_threshold_pct)

        trip_events = group_consecutive_hourly_events(
            condition_mask=trip_mask,
            timestamps=timestamps,
            values=pct_drop,
            event_type="Load Trip / Drop",
            additional_series={"load_drop_from": prev_power, "load_drop_to": power_series}
        )
        all_events.extend(trip_events)

    # ---------------------------------------------------------
    # 3. POOR POWER FACTOR (PF < 0.85)
    # ---------------------------------------------------------
    if pf_channel is not None:
        # Industrial meters often report signed PF for lagging/leading; take absolute value
        pf_series = pd.to_numeric(work_df[pf_channel], errors="coerce").abs()
        
        # In power engineering, low PF is evaluated under active operational load
        if p_col is not None:
            active_load_running = pd.to_numeric(work_df[p_col], errors="coerce") > (power_threshold_zero * 2)
            poor_pf_mask = active_load_running & (pf_series < pf_threshold) & (pf_series > 0.0)
        else:
            poor_pf_mask = (pf_series < pf_threshold) & (pf_series > 0.0)

        pf_events = group_consecutive_hourly_events(
            condition_mask=poor_pf_mask,
            timestamps=timestamps,
            values=pf_series,
            event_type="Poor Power Factor"
        )
        all_events.extend(pf_events)

    # ---------------------------------------------------------
    # 4. HIGH HARMONICS (THD > 5%)
    # ---------------------------------------------------------
    if thd_channel is not None:
        thd_series = pd.to_numeric(work_df[thd_channel], errors="coerce")
        high_thd_mask = thd_series > thd_threshold_pct

        thd_events = group_consecutive_hourly_events(
            condition_mask=high_thd_mask,
            timestamps=timestamps,
            values=thd_series,
            event_type="High Harmonics (THD)"
        )
        all_events.extend(thd_events)

    if not all_events:
        return pd.DataFrame(columns=[
            "start_time", "end_time", "duration_hours", "event_type", "severity", "details"
        ])

    event_df = pd.DataFrame(all_events)
    # Sort chronologically by start_time
    event_df = event_df.sort_values(by=["start_time", "severity"]).reset_index(drop=True)
    return event_df


class EnergyAnomalyDetector:
    """Wrapper class providing legacy helper methods and event detection interfaces."""

    def __init__(
        self,
        power_threshold_zero: float = 0.05,
        drop_threshold_pct: float = 40.0,
        pf_threshold: float = 0.85,
        thd_threshold_pct: float = 5.0
    ):
        self.power_threshold_zero = power_threshold_zero
        self.drop_threshold_pct = drop_threshold_pct
        self.pf_threshold = pf_threshold
        self.thd_threshold_pct = thd_threshold_pct

    def detect_events(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Runs the industrial event detection engine on the telemetry dataset."""
        return detect_industrial_events(
            df=df,
            power_threshold_zero=kwargs.get("power_threshold_zero", self.power_threshold_zero),
            drop_threshold_pct=kwargs.get("drop_threshold_pct", self.drop_threshold_pct),
            pf_threshold=kwargs.get("pf_threshold", self.pf_threshold),
            thd_threshold_pct=kwargs.get("thd_threshold_pct", self.thd_threshold_pct),
            power_col=kwargs.get("power_col"),
            voltage_col=kwargs.get("voltage_col"),
            pf_col=kwargs.get("pf_col"),
            thd_col=kwargs.get("thd_col")
        )

