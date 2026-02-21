"""Streamlit performance dashboard for the Kalshi BTC paper trader."""

import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = "data/trader.db"

st.set_page_config(page_title="Kalshi BTC Trader", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    """Return a shared SQLite connection (read-only)."""
    db = Path(DB_PATH)
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_trades() -> pd.DataFrame:
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_candles(hours: int = 24) -> pd.DataFrame:
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM candles WHERE open_time >= ? ORDER BY open_time",
            conn,
            params=(cutoff,),
        )
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute KPI metrics from the trades DataFrame."""
    resolved = df[df["resolved"] == 1].copy() if not df.empty else pd.DataFrame()
    total = len(df)
    total_resolved = len(resolved)

    if resolved.empty:
        return {
            "total_pnl": 0.0, "pnl_delta": 0.0, "win_rate": 0.0,
            "sharpe": 0.0, "profit_factor": 0.0, "total_trades": total,
            "max_drawdown": 0.0,
        }

    pnl_series = resolved.sort_values("timestamp")["pnl"]
    total_pnl = pnl_series.sum()

    # Yesterday's P&L for delta
    now = datetime.now(timezone.utc)
    yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_ts = int(yesterday_start.timestamp() * 1000)
    today_ts = int(today_start.timestamp() * 1000)
    yesterday_pnl = resolved[
        (resolved["timestamp"] >= yesterday_ts) & (resolved["timestamp"] < today_ts)
    ]["pnl"].sum()
    today_pnl = resolved[resolved["timestamp"] >= today_ts]["pnl"].sum()
    pnl_delta = today_pnl - yesterday_pnl

    wins = (pnl_series > 0).sum()
    win_rate = wins / total_resolved * 100 if total_resolved > 0 else 0.0

    # Sharpe (annualised, assuming ~96 trades/day for 15-min windows)
    if pnl_series.std() > 0:
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(96 * 365)
    else:
        sharpe = 0.0

    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    return {
        "total_pnl": total_pnl,
        "pnl_delta": pnl_delta,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "total_trades": total,
        "max_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    resolved = df[df["resolved"] == 1].sort_values("timestamp").copy()
    if resolved.empty:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve", height=350)
        return fig

    resolved["cum_pnl"] = resolved["pnl"].cumsum()
    running_max = resolved["cum_pnl"].cummax()
    resolved["drawdown"] = resolved["cum_pnl"] - running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=resolved["time"], y=resolved["cum_pnl"],
        mode="lines", name="Cumulative P&L",
        line=dict(color="#00cc96", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=resolved["time"], y=resolved["drawdown"],
        fill="tozeroy", name="Drawdown",
        line=dict(color="#ef553b", width=1),
        fillcolor="rgba(239,85,59,0.2)",
    ))
    fig.update_layout(
        title="Equity Curve", height=350,
        xaxis_title="Time", yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def btc_price_chart(candles: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not candles.empty:
        fig.add_trace(go.Candlestick(
            x=candles["time"],
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            name="BTC",
        ))

    # Overlay trade markers
    if not trades.empty:
        resolved = trades[trades["resolved"] == 1].copy()
        if not resolved.empty:
            wins = resolved[resolved["pnl"] > 0]
            losses = resolved[resolved["pnl"] <= 0]
            for subset, color, label in [(wins, "green", "Win"), (losses, "red", "Loss")]:
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset["time"], y=[candles["close"].iloc[-1]] * len(subset),
                        mode="markers", name=label,
                        marker=dict(color=color, size=10, symbol="triangle-up"),
                    ))

    fig.update_layout(
        title="BTC Price (Last 24h)", height=400,
        xaxis_title="Time", yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def vol_scatter(df: pd.DataFrame) -> go.Figure:
    vol_data = df.dropna(subset=["predicted_vol", "implied_vol"])
    fig = go.Figure()
    if not vol_data.empty:
        fig.add_trace(go.Scatter(
            x=vol_data["implied_vol"], y=vol_data["predicted_vol"],
            mode="markers", name="Trades",
            marker=dict(
                color=vol_data["pnl"].apply(lambda p: "green" if p and p > 0 else "red"),
                size=8, opacity=0.7,
            ),
        ))
        vol_min = min(vol_data["implied_vol"].min(), vol_data["predicted_vol"].min())
        vol_max = max(vol_data["implied_vol"].max(), vol_data["predicted_vol"].max())
        fig.add_trace(go.Scatter(
            x=[vol_min, vol_max], y=[vol_min, vol_max],
            mode="lines", name="x=y",
            line=dict(dash="dash", color="gray"),
        ))
    fig.update_layout(
        title="Predicted vs Implied Vol", height=350,
        xaxis_title="Implied Vol", yaxis_title="Predicted Vol",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def edge_histogram(df: pd.DataFrame) -> go.Figure:
    resolved = df[df["resolved"] == 1].copy()
    fig = go.Figure()
    if not resolved.empty:
        wins = resolved[resolved["pnl"] > 0]
        losses = resolved[resolved["pnl"] <= 0]
        for subset, color, label in [(wins, "green", "Win"), (losses, "red", "Loss")]:
            if not subset.empty:
                fig.add_trace(go.Histogram(
                    x=subset["edge"], name=label,
                    marker_color=color, opacity=0.6,
                ))
        fig.update_layout(barmode="overlay")
    fig.update_layout(
        title="Edge Distribution", height=350,
        xaxis_title="Edge", yaxis_title="Count",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("Kalshi BTC Trader Dashboard")
with col_status:
    db_exists = Path(DB_PATH).exists()
    if db_exists:
        st.success("DB Connected")
    else:
        st.error("No DB found")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

trades = load_trades()
candles = load_candles()

if trades.empty:
    st.info("Waiting for trades... Start the paper trader and check back soon.")
else:
    metrics = compute_metrics(trades)

    # -- KPI cards --
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Total P&L", f"${metrics['total_pnl']:.2f}", f"${metrics['pnl_delta']:+.2f}")
    k2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    k3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    pf_display = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] != float("inf") else "Inf"
    k4.metric("Profit Factor", pf_display)
    k5.metric("Total Trades", metrics["total_trades"])
    k6.metric("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
    avg_conf = trades["fill_confidence"].mean() if "fill_confidence" in trades.columns and trades["fill_confidence"].notna().any() else 0
    k7.metric("Avg Fill Conf.", f"{avg_conf:.0f}%")

    # -- Equity curve --
    st.plotly_chart(equity_curve_chart(trades), use_container_width=True)

    # -- BTC price chart --
    st.plotly_chart(btc_price_chart(candles, trades), use_container_width=True)

    # -- Trade log table --
    st.subheader("Trade Log")
    display_df = trades.copy()
    display_cols = ["time", "side", "price", "size", "edge", "fill_confidence", "predicted_vol", "implied_vol", "pnl"]
    available_cols = [c for c in display_cols if c in display_df.columns]
    if "predicted_vol" in display_df.columns and "implied_vol" in display_df.columns:
        display_df["vol_ratio"] = display_df.apply(
            lambda r: r["predicted_vol"] / r["implied_vol"]
            if r["implied_vol"] and r["implied_vol"] > 0 and r["predicted_vol"] else None,
            axis=1,
        )
        available_cols.append("vol_ratio")

    # Add result column
    display_df["result"] = display_df["pnl"].apply(
        lambda p: "WIN" if p is not None and p > 0 else ("LOSS" if p is not None and p < 0 else "PENDING")
    )
    available_cols.append("result")

    st.dataframe(
        display_df[available_cols],
        use_container_width=True,
        height=400,
        column_config={
            "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
            "edge": st.column_config.NumberColumn("Edge", format="%.4f"),
            "fill_confidence": st.column_config.ProgressColumn("Fill Conf.", format="%.0f%%", min_value=0, max_value=100),
            "predicted_vol": st.column_config.NumberColumn("Pred Vol", format="%.5f"),
            "implied_vol": st.column_config.NumberColumn("Impl Vol", format="%.5f"),
            "vol_ratio": st.column_config.NumberColumn("Vol Ratio", format="%.3f"),
        },
    )

    # -- Fill confidence analysis --
    if "fill_confidence" in trades.columns and trades["fill_confidence"].notna().any():
        st.subheader("Fill Confidence (Latency Risk)")
        fc1, fc2 = st.columns(2)
        with fc1:
            fig_conf = go.Figure()
            conf_data = trades["fill_confidence"].dropna()
            colors = conf_data.apply(
                lambda c: "green" if c >= 70 else ("orange" if c >= 40 else "red")
            )
            fig_conf.add_trace(go.Bar(
                x=list(range(len(conf_data))),
                y=conf_data.values,
                marker_color=colors.values,
                name="Fill Confidence",
            ))
            fig_conf.add_hline(y=70, line_dash="dash", line_color="green",
                               annotation_text="High (70%+)")
            fig_conf.add_hline(y=40, line_dash="dash", line_color="orange",
                               annotation_text="Medium (40%+)")
            fig_conf.update_layout(
                title="Fill Confidence per Trade",
                xaxis_title="Trade #", yaxis_title="Confidence %",
                yaxis_range=[0, 100], height=350,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        with fc2:
            high = (conf_data >= 70).sum()
            medium = ((conf_data >= 40) & (conf_data < 70)).sum()
            low = (conf_data < 40).sum()
            total_conf = len(conf_data)
            st.metric("High Confidence (70%+)", f"{high}/{total_conf}")
            st.metric("Medium Confidence (40-70%)", f"{medium}/{total_conf}")
            st.metric("Low Confidence (<40%)", f"{low}/{total_conf}")
            st.caption(
                "**Factors:** Orderbook depth vs order size (35%), "
                "data staleness (30%), bid-ask spread (20%), "
                "price impact/levels consumed (15%)"
            )

    # -- Vol model analysis --
    st.subheader("Vol Model Analysis")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(vol_scatter(trades), use_container_width=True)
    with vcol2:
        st.plotly_chart(edge_histogram(trades), use_container_width=True)

    # -- Per-side breakdown --
    st.subheader("Per-Side Breakdown")
    scol1, scol2 = st.columns(2)
    for col, side in [(scol1, "yes"), (scol2, "no")]:
        with col:
            side_df = trades[trades["side"] == side]
            side_resolved = side_df[side_df["resolved"] == 1]
            count = len(side_df)
            wins = (side_resolved["pnl"] > 0).sum() if not side_resolved.empty else 0
            wr = wins / len(side_resolved) * 100 if len(side_resolved) > 0 else 0
            total_pnl = side_resolved["pnl"].sum() if not side_resolved.empty else 0
            st.metric(f"{side.upper()} Trades", count)
            st.metric(f"{side.upper()} Win Rate", f"{wr:.1f}%")
            st.metric(f"{side.upper()} P&L", f"${total_pnl:.2f}")


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.rerun()
