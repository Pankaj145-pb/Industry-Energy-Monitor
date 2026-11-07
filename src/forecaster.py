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
