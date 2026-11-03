"""Data ingestion, dynamic column detection, multi-column aggregation, grid reindexing, gap interpolation, and data quality profiling."""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import argparse
import json
import numpy as np
import pandas as pd
import re


def detect_timestamp_column(df: pd.DataFrame, explicit_col: Optional[str] = None) -> str:
    """Dynamically detects the best timestamp column from a dataframe.

    Checks explicit names, datetime dtypes, parsing sample values, and keyword heuristics.
    """
    if explicit_col and explicit_col in df.columns:
        return explicit_col

    # 1. Check for existing datetime dtypes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)

    # 2. Check column names with timestamp-like keywords
    ts_keywords = [
        "timestamp", "datetime", "date_time", "read_time", "reading_time",
        "utc_time", "local_time", "time", "date", "ts", "period"
    ]
    for kw in ts_keywords:
        for col in df.columns:
            cleaned = str(col).strip().lower().replace(" ", "_")
            if cleaned == kw or cleaned.startswith(f"{kw}_") or cleaned.endswith(f"_{kw}"):
                # Validate parseability on a small sample
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        return str(col)

    # 3. Dynamic test: test string/object columns to see if they parse to datetime
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    return str(col)

    # Fallback to the first column
    return str(df.columns[0])

DEFAULT_LOAD_KEYWORDS = [
    "watts_total", "active_power", "load_kw",
    "kw", "kwh", "power", "load", "demand", "energy"
]

# catches L1_power_w, total_kwh, consumption_mwh, etc. without listing them
UNIT_PATTERN = re.compile(
    r'(^|_)(w|kw|mw|gw|wh|kwh|mwh|watt|watts|volt_amp|va|power|load|demand|consumption|energy)(_|$|s_)'
)

def _normalize(col: str) -> str:
    return col.strip().lower().replace(" ", "_").replace("-", "_")

def get_column_inventory(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    primary_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Dynamically profiles ALL columns, categorizing them into
    timestamp, numeric (measurements/telemetry), and categorical/metadata fields.
    """
    ts_col = timestamp_col or detect_timestamp_column(df)
    all_columns = list(df.columns)

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for col in all_columns:
        if col == ts_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(str(col))
        else:
            sample = df[col].dropna().head(20)
            if not sample.empty:
                numeric_converted = pd.to_numeric(sample, errors="coerce")
                if numeric_converted.notna().mean() >= 0.8:
                    numeric_columns.append(str(col))
                else:
                    categorical_columns.append(str(col))
            else:
                categorical_columns.append(str(col))

    # 1. Explicit keywords (caller can override per plant/dataset)
    keywords = primary_keywords or DEFAULT_LOAD_KEYWORDS
    ranked_primary = None

    for kw in keywords:
        kw_norm = _normalize(kw)
        for col in numeric_columns:
            cleaned = _normalize(col)
            if kw_norm == cleaned or kw_norm in cleaned.split("_"):
                ranked_primary = col
                break
        if ranked_primary:
            break

    # 2. Unit-aware dynamic match (no hard-coded name needed)
    if not ranked_primary:
        for col in numeric_columns:
            if UNIT_PATTERN.search(_normalize(col)):
                ranked_primary = col
                break

    # 3. Statistical fallback: most-varying numeric (load/energy varies most)
    if not ranked_primary and numeric_columns:
        try:
            stds = pd.to_numeric(df[numeric_columns].apply(pd.to_numeric, errors="coerce").std(), errors="coerce")
            ranked_primary = stds.dropna().sort_values(ascending=False).index[0]
        except Exception:
            ranked_primary = numeric_columns[0]

    return {
        "timestamp_column": ts_col,
        "primary_metric_column": ranked_primary or (all_columns[0] if all_columns else ""),
        "primary_keywords_used": keywords,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "all_columns": [str(c) for c in all_columns],
        "total_columns_count": len(all_columns),
    }

def calculate_max_continuous_consecutive(condition_series: pd.Series) -> int:
    """Calculates the maximum continuous run of True values in a boolean Series."""
    if condition_series.empty or not condition_series.any():
        return 0
    blocks = (~condition_series).cumsum()[condition_series]
    if blocks.empty:
        return 0
    return int(blocks.value_counts().max())


class EnergyDataProcessor:
    """Ingests Excel sheets, dynamically catalogs all columns, aligns timestamps to a

    continuous hourly (e.g. 8,760h) grid, interpolates minor missing gaps, and computes quality KPIs.
    """

    def __init__(
        self,
        target_hours: int = 8760,
        max_interp_gap: int = 3,
        outage_threshold_kw: float = 0.05
    ):
        self.target_hours = target_hours
        self.max_interp_gap = max_interp_gap
        self.outage_threshold_kw = outage_threshold_kw

    def load_excel(
        self,
        file_path: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = 0
    ) -> pd.DataFrame:
        """Loads raw telemetry data from an Excel workbook or CSV."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(path, sheet_name=sheet_name)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return df

    def process_and_reindex_grid(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        value_col: Optional[str] = None,
        additional_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cleans, dynamically discovers all columns, aggregates numeric channels to 1H,

        reindexes to a continuous grid, interpolates minor gaps, and computes quality KPIs.
        """
        inventory = get_column_inventory(df, timestamp_col=timestamp_col)
        ts_col = inventory["timestamp_column"]
        val_col = value_col if (value_col and value_col in df.columns) else inventory["primary_metric_column"]

        # Parse timestamps and drop invalid rows
        work_df = df.copy()
        work_df[ts_col] = pd.to_datetime(work_df[ts_col], errors="coerce")
        work_df = work_df.dropna(subset=[ts_col])

        if work_df.empty:
            raise ValueError(f"No valid datetime records found in column '{ts_col}'.")

        # Select which numeric columns to process: primary + additional requested (or all numeric if requested)
        cols_to_process = [val_col]
        if additional_cols:
            for c in additional_cols:
                if c in df.columns and c not in cols_to_process:
                    cols_to_process.append(c)

        for col in cols_to_process:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

        # Determine start and end boundaries (floored/ceiled to whole hour)
        start_ts = work_df[ts_col].min().floor("h")
        end_ts = work_df[ts_col].max().ceil("h")

        # Create continuous hourly grid
        grid_continuous = pd.date_range(start=start_ts, end=end_ts, freq="1h", name="timestamp")
        total_hours = len(grid_continuous)

        # Aggregate raw telemetry to hourly mean for all numeric columns being processed
        work_df = work_df.sort_values(by=ts_col)
        hourly_df = (
            work_df.set_index(ts_col)[cols_to_process]
            .resample("1h")
            .mean()
        )

        # Reindex to continuous grid
        reindexed_df = hourly_df.reindex(grid_continuous)
        
        # Primary load channel calculations
        raw_series = reindexed_df[val_col].copy()
        raw_missing_mask = raw_series.isna()
        raw_missing_count = int(raw_missing_mask.sum())

        max_continuous_missing_hours = calculate_max_continuous_consecutive(raw_missing_mask)

        # Interpolate minor missing gaps on all processed numeric channels
        clean_cols = {}
        for col in cols_to_process:
            clean_cols[f"{col}_clean"] = reindexed_df[col].interpolate(
                method="linear",
                limit=self.max_interp_gap,
                limit_direction="both"
            )
        
        for clean_name, clean_series in clean_cols.items():
            reindexed_df[clean_name] = clean_series

        interpolated_series = reindexed_df[f"{val_col}_clean"]
        post_interp_missing_mask = interpolated_series.isna()
        post_interp_missing_count = int(post_interp_missing_mask.sum())
        interpolated_points_count = raw_missing_count - post_interp_missing_count

        # Completeness metrics
        raw_completeness_pct = round(((total_hours - raw_missing_count) / total_hours) * 100, 2)
        final_completeness_pct = round(((total_hours - post_interp_missing_count) / total_hours) * 100, 2)

        # Outages detection
        outage_mask = (reindexed_df[f"{val_col}_clean"] <= self.outage_threshold_kw) & (~post_interp_missing_mask)
        total_outage_hours = int(outage_mask.sum())
        max_continuous_outage_hours = calculate_max_continuous_consecutive(outage_mask)

        # Summary dictionary
        summary = {
            "start_timestamp": str(start_ts),
            "end_timestamp": str(end_ts),
            "detected_timestamp_column": ts_col,
            "selected_metric_column": val_col,
            "all_columns": inventory["all_columns"],
            "numeric_columns": inventory["numeric_columns"],
            "categorical_columns": inventory["categorical_columns"],
            "total_columns_count": inventory["total_columns_count"],
            "expected_grid_hours": int(total_hours),
            "raw_recorded_hours": int(total_hours - raw_missing_count),
            "raw_missing_hours": raw_missing_count,
            "raw_completeness_percentage": raw_completeness_pct,
            "interpolated_gap_hours": int(interpolated_points_count),
            "remaining_missing_hours": post_interp_missing_count,
            "final_completeness_percentage": final_completeness_pct,
            "max_continuous_missing_gap_hours": int(max_continuous_missing_hours),
            "total_outage_hours": int(total_outage_hours),
            "max_continuous_outage_hours": int(max_continuous_outage_hours),
            "outage_threshold_kw": float(self.outage_threshold_kw),
            "peak_metric_value": float(reindexed_df[f"{val_col}_clean"].max()) if not reindexed_df[f"{val_col}_clean"].dropna().empty else 0.0,
            "average_metric_value": float(reindexed_df[f"{val_col}_clean"].mean()) if not reindexed_df[f"{val_col}_clean"].dropna().empty else 0.0,
            "min_metric_value": float(reindexed_df[f"{val_col}_clean"].min()) if not reindexed_df[f"{val_col}_clean"].dropna().empty else 0.0,
        }

        return reindexed_df, summary