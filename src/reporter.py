"""PDF and CSV export utilities for industrial energy reports."""
from typing import Dict, Any, Union, Optional
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class EnergyReporter:
    """Exports structured energy reports: Daily/Monthly summaries and PDF audit reports."""

    @staticmethod
    def generate_daily_summary(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        """Aggregates hourly telemetry to daily energy metrics:

        peak load, mean load, minimum load, total energy consumption.
        """
        clean_col = f"{metric_col}_clean" if f"{metric_col}_clean" in df.columns else metric_col
        daily_df = df.resample("1D").agg(
            peak_power=(clean_col, "max"),
            avg_power=(clean_col, "mean"),
            min_power=(clean_col, "min"),
            total_energy_wh=(clean_col, lambda s: s.sum() * 1.0)  # 1-hour interval Wh
        )
        daily_df["peak_power"] = daily_df["peak_power"].round(2)
        daily_df["avg_power"] = daily_df["avg_power"].round(2)
        daily_df["min_power"] = daily_df["min_power"].round(2)
        daily_df["total_energy_wh"] = daily_df["total_energy_wh"].round(2)
        return daily_df

    @staticmethod
    def generate_monthly_summary(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        """Aggregates hourly telemetry to monthly energy metrics."""
        clean_col = f"{metric_col}_clean" if f"{metric_col}_clean" in df.columns else metric_col
        monthly_df = df.resample("1ME").agg(
            peak_power=(clean_col, "max"),
            avg_power=(clean_col, "mean"),
            min_power=(clean_col, "min"),
            total_energy_wh=(clean_col, lambda s: s.sum() * 1.0)
        )
        monthly_df["peak_power"] = monthly_df["peak_power"].round(2)
        monthly_df["avg_power"] = monthly_df["avg_power"].round(2)
        monthly_df["min_power"] = monthly_df["min_power"].round(2)
        monthly_df["total_energy_wh"] = monthly_df["total_energy_wh"].round(2)
        return monthly_df

    @staticmethod
    def generate_monthly_pdf_report(
        summary_dict: Dict[str, Any],
        monthly_df: pd.DataFrame,
        metric_col: str
    ) -> bytes:
        """Generates a formatted Monthly Summary PDF audit report in-memory."""
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Industrial Energy Audit: Monthly Executive Summary")

        c.setFont("Helvetica", 10)
        c.drawString(50, height - 68, f"Target Metric: {metric_col} | Timeline: {summary_dict.get('start_timestamp', '')[:10]} to {summary_dict.get('end_timestamp', '')[:10]}")
        c.line(50, height - 74, width - 50, height - 74)

        # Overview KPI block
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 100, "High-Level Facility KPIs:")
        c.setFont("Helvetica", 10)
        y = height - 120
        c.drawString(60, y, f"- Peak Telemetry: {summary_dict.get('peak_metric_value', summary_dict.get('peak_load_kw', 0)):,.2f}")
        y -= 16
        c.drawString(60, y, f"- Average Telemetry: {summary_dict.get('average_metric_value', summary_dict.get('average_load_kw', 0)):,.2f}")
        y -= 16
        c.drawString(60, y, f"- Total Grid Hours: {summary_dict.get('expected_grid_hours', 0):,} h")
        y -= 16
        c.drawString(60, y, f"- Final Completeness: {summary_dict.get('final_completeness_percentage', 0):.2f}%")
        y -= 16
        c.drawString(60, y, f"- Total Outage Hours: {summary_dict.get('total_outage_hours', 0)} h")

        # Monthly Table
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Monthly Breakdown:")
        y -= 20

        c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y, "Month")
        c.drawString(140, y, "Peak Value")
        c.drawString(250, y, "Avg Value")
        c.drawString(360, y, "Min Value")
        c.drawString(470, y, "Total Wh")
        y -= 5
        c.line(50, y, width - 50, y)
        y -= 14

        c.setFont("Helvetica", 9)
        for idx, row in monthly_df.iterrows():
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 9)
            month_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)[:7]
            c.drawString(50, y, month_str)
            c.drawString(140, y, f"{row['peak_power']:,.1f}")
            c.drawString(250, y, f"{row['avg_power']:,.1f}")
            c.drawString(360, y, f"{row['min_power']:,.1f}")
            c.drawString(470, y, f"{row['total_energy_wh']:,.0f}")
            y -= 16

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

