"""24-Hour Industrial Load Forecasting Engine.

Implements feature engineering (cyclical hour/day transforms, lag features, rolling statistics)
and training via Random Forest Regressor to predict the next 24 hours of facility load demand.
Includes holdout validation metrics (MAE, RMSE, MAPE) and future 24h trajectory extrapolation.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def build_forecasting_features(
    series: pd.Series,
    lags: Tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    rolling_windows: Tuple[int, ...] = (6, 24)
) -> pd.DataFrame:
    """Builds lag, rolling, calendar, and cyclical features for time series forecasting.

    `series` should be an hourly continuous Series with a DatetimeIndex.
    """
    df = pd.DataFrame({"target": series.astype(float)})

    # Calendar features
    hour = df.index.hour
    dayofweek = df.index.dayofweek
    month = df.index.month

    # Cyclical transformations (preserves periodic continuity: 23h -> 00h)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Shift / Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = df["target"].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df["target"].shift(1).rolling(window=w).std()

    return df


class LoadForecaster:
    """Trains a 24-hour predictive model using Random Forest Regressor on historical telemetry."""

    def __init__(
        self,
        horizon_hours: int = 24,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42
    ):
        self.horizon_hours = horizon_hours
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_cols = []
        self.metrics = {}
        self.last_timestamp = None
        self.target_name = "load"

    def evaluate_holdout(
        self,
        df_feat: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluates backtesting performance on the last 24 hours of historical data."""
        clean = df_feat.dropna()
        if len(clean) <= self.horizon_hours * 2:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        train_set = clean.iloc[:-self.horizon_hours]
        test_set = clean.iloc[-self.horizon_hours:]

        x_train = train_set[self.feature_cols]
        y_train = train_set["target"]
        x_test = test_set[self.feature_cols]
        y_test = test_set["target"]

        eval_model = RandomForestRegressor(
            n_estimators=60,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        eval_model.fit(x_train, y_train)
        preds = eval_model.predict(x_test)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        non_zero = y_test != 0
        if non_zero.any():
            mape = float(np.mean(np.abs((y_test[non_zero] - preds[non_zero]) / y_test[non_zero])) * 100.0)
        else:
            mape = 0.0

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2)
        }

    def fit_and_predict_next_24h(
        self,
        series: pd.Series,
        target_name: str = "Active Power"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Trains on the historical series, generates 24 future hourly steps,

        and returns (forecast_df, backtest_df, metrics_dict).
        """
        self.target_name = target_name
        series_clean = series.dropna().astype(float)
        if len(series_clean) < 168:
            raise ValueError("Insufficient history for forecasting (minimum 168 hours / 1 week required).")

        df_feat = build_forecasting_features(series_clean)
        self.feature_cols = [c for c in df_feat.columns if c != "target"]

        # Backtest metrics on last 24h
        self.metrics = self.evaluate_holdout(df_feat)

        # Train on full clean historical data
        train_df = df_feat.dropna()
        x_full = train_df[self.feature_cols]
        y_full = train_df["target"]
        self.model.fit(x_full, y_full)

        # Generate future 24-hour timestamps
        last_ts = series_clean.index[-1]
        self.last_timestamp = last_ts
        future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=self.horizon_hours, freq="1h")

        # Recursive multi-step prediction
        history = series_clean.copy()
        future_preds = []

        for next_ts in future_index:
            temp_series = pd.concat([history, pd.Series([np.nan], index=[next_ts])])
            temp_features = build_forecasting_features(temp_series)
            next_feat_row = temp_features.loc[[next_ts], self.feature_cols]

            pred_val = float(self.model.predict(next_feat_row)[0])
            pred_val = max(0.0, pred_val)  # Power cannot be negative
            future_preds.append(pred_val)

            history.loc[next_ts] = pred_val

        # 1. Future Forecast DataFrame
        forecast_df = pd.DataFrame(
            {
                "predicted_load": np.round(future_preds, 2),
                "predicted_load_mw": np.round(np.array(future_preds) / 1e6, 3),
            },
            index=future_index
        )
        forecast_df.index.name = "timestamp"

        # 2. Backtest Comparison DataFrame (last 72 hours actual)
        recent_actuals = series_clean.tail(72).to_frame(name="actual_load")
        recent_actuals["actual_load_mw"] = np.round(recent_actuals["actual_load"] / 1e6, 3)

        summary_metrics = {
            "model_type": "RANDOM FOREST REGRESSOR",
            "mae": self.metrics["mae"],
            "mae_mw": round(self.metrics["mae"] / 1e6, 3),
            "rmse": self.metrics["rmse"],
            "rmse_mw": round(self.metrics["rmse"] / 1e6, 3),
            "mape_pct": self.metrics["mape"],
            "forecast_peak": float(forecast_df["predicted_load"].max()),
            "forecast_peak_mw": float(forecast_df["predicted_load_mw"].max()),
            "forecast_peak_time": str(forecast_df["predicted_load"].idxmax()),
            "forecast_avg": float(forecast_df["predicted_load"].mean()),
            "forecast_avg_mw": float(forecast_df["predicted_load_mw"].mean()),
            "forecast_min": float(forecast_df["predicted_load"].min()),
            "forecast_min_mw": float(forecast_df["predicted_load_mw"].min()),
            "last_historical_timestamp": str(last_ts),
            "forecast_start": str(future_index[0]),
            "forecast_end": str(future_index[-1])
        }

        return forecast_df, recent_actuals, summary_metrics

    def predict_and_compare_date_profiles(
        self,
        series: pd.Series,
        baseline_date: Any,
        target_date: Any,
        growth_rate_pct: float = 0.0,
        target_name: str = "Active Power"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compares a 24-hour historical baseline day (e.g. Nov 15, 2025)

        with a predicted future 24-hour profile (e.g. Nov 15, 2026),
        factoring in calendar shifts (day of week) and an optional YoY growth rate.
        """
        self.target_name = target_name
        series_clean = series.dropna().astype(float)
        if len(series_clean) < 168:
            raise ValueError("Insufficient history for forecasting (minimum 168 hours / 1 week required).")

        # Ensure model is trained
        if not self.feature_cols or not hasattr(self.model, "estimators_"):
            df_feat = build_forecasting_features(series_clean)
            self.feature_cols = [c for c in df_feat.columns if c != "target"]
            train_df = df_feat.dropna()
            self.model.fit(train_df[self.feature_cols], train_df["target"])

        # Normalize baseline and target dates
        base_dt = pd.to_datetime(baseline_date).date()
        target_dt = pd.to_datetime(target_date).date()

        base_hours = pd.date_range(start=f"{base_dt} 00:00:00", periods=24, freq="1h")
        target_hours = pd.date_range(start=f"{target_dt} 00:00:00", periods=24, freq="1h")

        # Extract baseline historical actuals
        base_series = series_clean.reindex(base_hours).interpolate().bfill().ffill()
        growth_factor = 1.0 + (growth_rate_pct / 100.0)
        base_vals = base_series.values * growth_factor

        # Multi-step predictive inference for target day
        preds = []
        last_pred = base_vals[-1]
        last_pred_2 = base_vals[-2]
        last_pred_3 = base_vals[-3]

        for h in range(24):
            t_hour = target_hours[h]
            dow = getattr(t_hour, "day_of_week", getattr(t_hour, "dayofweek", 0))

            feat = {
                "hour_sin": np.sin(2 * np.pi * t_hour.hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * t_hour.hour / 24.0),
                "day_sin": np.sin(2 * np.pi * dow / 7.0),
                "day_cos": np.cos(2 * np.pi * dow / 7.0),
                "month_sin": np.sin(2 * np.pi * t_hour.month / 12.0),
                "month_cos": np.cos(2 * np.pi * t_hour.month / 12.0),
                "lag_1": last_pred,
                "lag_2": last_pred_2,
                "lag_3": last_pred_3,
                "lag_24": base_vals[h],
                "lag_48": base_vals[h],
                "lag_168": base_vals[h],
                "rolling_mean_6": float(np.mean(preds[-6:] if len(preds) >= 6 else base_vals[:6])),
                "rolling_std_6": float(np.std(preds[-6:] if len(preds) >= 6 else base_vals[:6])),
                "rolling_mean_24": float(np.mean(base_vals)),
                "rolling_std_24": float(np.std(base_vals)),
            }
            feat_df = pd.DataFrame([feat])[self.feature_cols]
            pred_val = float(self.model.predict(feat_df)[0])
            pred_val = max(0.0, pred_val)
            preds.append(pred_val)

            last_pred_3 = last_pred_2
            last_pred_2 = last_pred
            last_pred = pred_val

        preds = np.array(preds)
        base_actual_mw = np.round(base_series.values / 1e6, 3)
        predicted_mw = np.round(preds / 1e6, 3)
        delta_mw = np.round(predicted_mw - base_actual_mw, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_pct = np.where(base_actual_mw > 0, np.round((delta_mw / base_actual_mw) * 100.0, 1), 0.0)

        hour_labels = [f"{h:02d}:00" for h in range(24)]

        comp_df = pd.DataFrame({
            "hour": range(24),
            "hour_label": hour_labels,
            "baseline_date": str(base_dt),
            "target_date": str(target_dt),
            "baseline_load_w": np.round(base_series.values, 2),
            "baseline_load_mw": base_actual_mw,
            "predicted_load_w": np.round(preds, 2),
            "predicted_load_mw": predicted_mw,
            "delta_mw": delta_mw,
            "delta_pct": delta_pct
        })

        base_peak_mw = float(np.max(base_actual_mw))
        base_peak_hr = hour_labels[int(np.argmax(base_actual_mw))]
        pred_peak_mw = float(np.max(predicted_mw))
        pred_peak_hr = hour_labels[int(np.argmax(predicted_mw))]

        base_total_mwh = float(np.sum(base_actual_mw))
        pred_total_mwh = float(np.sum(predicted_mw))

        peak_delta_mw = round(pred_peak_mw - base_peak_mw, 3)
        peak_delta_pct = round((peak_delta_mw / base_peak_mw * 100.0), 2) if base_peak_mw > 0 else 0.0
        energy_delta_mwh = round(pred_total_mwh - base_total_mwh, 3)
        energy_delta_pct = round((energy_delta_mwh / base_total_mwh * 100.0), 2) if base_total_mwh > 0 else 0.0

        metrics = {
            "baseline_date": str(base_dt),
            "target_date": str(target_dt),
            "growth_rate_pct": float(growth_rate_pct),
            "baseline_peak_mw": base_peak_mw,
            "baseline_peak_hour": base_peak_hr,
            "predicted_peak_mw": pred_peak_mw,
            "predicted_peak_hour": pred_peak_hr,
            "peak_delta_mw": peak_delta_mw,
            "peak_delta_pct": peak_delta_pct,
            "baseline_avg_mw": round(float(np.mean(base_actual_mw)), 3),
            "predicted_avg_mw": round(float(np.mean(predicted_mw)), 3),
            "delta_avg_mw": round(float(np.mean(predicted_mw) - np.mean(base_actual_mw)), 3),
            "baseline_total_mwh": round(base_total_mwh, 2),
            "predicted_total_mwh": round(pred_total_mwh, 2),
            "energy_delta_mwh": energy_delta_mwh,
            "energy_delta_pct": energy_delta_pct,
        }

        return comp_df, metrics
