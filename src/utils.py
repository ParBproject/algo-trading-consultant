"""
src/utils.py
============
Shared utilities: logging setup, config loading, plotting, and PDF reporting.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure loguru to write to console and rotating file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        f"{log_dir}/trading_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="30 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config/example.yaml") -> dict[str, Any]:
    """Load YAML config file and expand environment variable references."""
    with open(path) as f:
        raw = f.read()

    # Expand ${VAR} references
    for key, val in os.environ.items():
        raw = raw.replace(f"${{{key}}}", val)

    return yaml.safe_load(raw)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    figsize: tuple = (12, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot strategy equity curve with optional benchmark overlay.

    Args:
        equity:     Strategy equity curve (dollar values).
        benchmark:  Optional benchmark equity (e.g., buy-and-hold).
        title:      Chart title.
        figsize:    Matplotlib figure size.
        save_path:  If provided, save figure to this path (PNG).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax_eq, ax_dd = axes

    # Normalise to 100
    norm_eq = equity / equity.iloc[0] * 100
    ax_eq.plot(norm_eq.index, norm_eq.values, color="#2196F3", lw=1.5, label="Strategy")

    if benchmark is not None:
        # Rebase on the strategy window. Dividing by the benchmark's own
        # first row first leaves earlier history in the level, so the line
        # does not start at 100 on the date the strategy starts.
        aligned = benchmark.reindex(norm_eq.index).ffill()
        valid = aligned.dropna()
        if len(valid) and float(valid.iloc[0]) != 0:
            norm_bm = aligned / float(valid.iloc[0]) * 100
            ax_eq.plot(norm_bm.index, norm_bm.values, color="#9E9E9E",
                       lw=1.0, linestyle="--", label="Benchmark")

    ax_eq.set_title(title, fontsize=14, fontweight="bold")
    ax_eq.set_ylabel("Portfolio Value (rebased 100)")
    ax_eq.legend()
    ax_eq.grid(alpha=0.3)

    # Drawdown subplot
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                        color="#F44336", alpha=0.4, label="Drawdown %")
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.3)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Equity chart saved: {save_path}")
    return fig


def plot_signals(
    df: pd.DataFrame,
    price_col: str = "Close",
    signal_col: str = "Signal",
    title: str = "Price & Signals",
    figsize: tuple = (14, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot price chart with buy/sell signal markers.

    Green triangles (▲) = long entry.
    Red triangles (▼)   = short entry / long exit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[price_col], color="#37474F", lw=1.2, label=price_col)

    longs = df[df[signal_col] == 1]
    shorts = df[df[signal_col] == -1]
    exits = df[df[signal_col] == 0]

    ax.scatter(longs.index, longs[price_col], marker="^", color="#4CAF50",
               s=80, zorder=5, label="Long Entry")
    ax.scatter(shorts.index, shorts[price_col], marker="v", color="#F44336",
               s=80, zorder=5, label="Short Entry")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_optimization_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe",
    title: str = "Parameter Optimization Heatmap",
    figsize: tuple = (10, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Pivot optimization results into a 2D heatmap.

    Args:
        results_df: DataFrame from StrategyOptimizer.run() with columns
                    for param_x, param_y, and metric.
        param_x:    Column for x-axis parameter.
        param_y:    Column for y-axis parameter.
        metric:     Column for the color/value axis.
    """
    import seaborn as sns

    pivot = results_df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── PDF report generator ─────────────────────────────────────────────────────

def generate_report(
    metrics,
    strategy_name: str,
    ticker: str,
    chart_path: str | None = None,
    output_path: str = "reports/backtest_report.pdf",
) -> str:
    """
    Generate a client-ready PDF backtest report via ReportLab.

    Args:
        metrics:       PerformanceMetrics dataclass instance.
        strategy_name: Human-readable strategy name.
        ticker:        Asset traded.
        chart_path:    Optional path to equity chart image to embed.
        output_path:   Destination PDF path.

    Returns:
        Path to generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage)
    except ImportError:
        raise ImportError("Install reportlab: pip install reportlab")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(
        f"<b>Backtest Report</b>: {strategy_name} on {ticker}",
        styles["Title"]
    ))
    story.append(Paragraph(
        "⚠ Educational use only. Past performance does not guarantee future results. "
        "Trading involves substantial risk of loss.",
        ParagraphStyle("disclaimer", parent=styles["Normal"],
                       textColor=colors.red, fontSize=8, spaceAfter=12)
    ))
    story.append(Spacer(1, 0.2*inch))

    # Metrics table
    data = [["Metric", "Value"]]
    metric_map = {
        "Total Return": f"{metrics.total_return:.2%}",
        "Annual Return": f"{metrics.annual_return:.2%}",
        "Ann. Volatility": f"{metrics.volatility_annual:.2%}",
        "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
        "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
        "Calmar Ratio": f"{metrics.calmar_ratio:.2f}",
        "Max Drawdown": f"{metrics.max_drawdown:.2%}",
        "# Trades": str(metrics.num_trades),
        "Win Rate": f"{metrics.win_rate:.2%}",
        "Profit Factor": f"{metrics.profit_factor:.2f}",
        "Avg Trade Return": f"{metrics.avg_trade_return:.4%}",
    }
    data.extend(metric_map.items())

    tbl = Table(data, colWidths=[3*inch, 2*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)

    # Equity chart
    if chart_path and Path(chart_path).exists():
        story.append(Spacer(1, 0.3*inch))
        story.append(RLImage(chart_path, width=6*inch, height=3*inch))

    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")
    return output_path
