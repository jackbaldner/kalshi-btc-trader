"""Streamlit performance dashboard for the Kalshi BTC trader."""

import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/trader.db"
MODE = os.getenv("MODE", "paper")

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
    open_trades = total - total_resolved

    if resolved.empty:
        return {
            "total_pnl": 0.0, "pnl_delta": 0.0, "win_rate": 0.0,
            "sharpe": 0.0, "profit_factor": 0.0, "total_trades": total,
            "max_drawdown": 0.0, "open_trades": open_trades,
            "today_pnl": 0.0, "today_trades": 0,
        }

    pnl_series = resolved.sort_values("timestamp")["pnl"]
    total_pnl = pnl_series.sum()

    now = datetime.now(timezone.utc)
    yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_ts = int(yesterday_start.timestamp() * 1000)
    today_ts = int(today_start.timestamp() * 1000)
    yesterday_pnl = resolved[
        (resolved["timestamp"] >= yesterday_ts) & (resolved["timestamp"] < today_ts)
    ]["pnl"].sum()
    today_pnl = resolved[resolved["timestamp"] >= today_ts]["pnl"].sum()
    today_trades = len(df[df["timestamp"] >= today_ts])
    pnl_delta = today_pnl - yesterday_pnl

    wins = (pnl_series > 0).sum()
    win_rate = wins / total_resolved * 100 if total_resolved > 0 else 0.0

    if pnl_series.std() > 0:
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(96 * 365)
    else:
        sharpe = 0.0

    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

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
        "open_trades": open_trades,
        "today_pnl": today_pnl,
        "today_trades": today_trades,
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


def daily_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of P&L by day."""
    resolved = df[df["resolved"] == 1].copy()
    fig = go.Figure()
    if resolved.empty:
        fig.update_layout(title="Daily P&L", height=300)
        return fig

    resolved["date"] = resolved["time"].dt.date
    daily = resolved.groupby("date")["pnl"].sum().reset_index()
    colors = daily["pnl"].apply(lambda p: "#00cc96" if p >= 0 else "#ef553b")

    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["pnl"],
        marker_color=colors,
        name="Daily P&L",
    ))
    fig.update_layout(
        title="Daily P&L", height=300,
        xaxis_title="Date", yaxis_title="P&L ($)",
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

    st.divider()
    st.subheader("Risk Limits")
    st.caption("Max Position: $50")
    st.caption("Max Exposure: $150")
    st.caption("Max Daily Loss: $50")
    st.caption("Edge Threshold: 4%")
    st.caption("Kelly Fraction: 0.5")
    st.divider()
    st.subheader("Trade Timing")
    st.caption("Evaluates 15 min before market close")
    st.caption("Matches vol model 15-min training window")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_mode, col_status = st.columns([3, 1, 1])
with col_title:
    st.title("Kalshi BTC Trader")
with col_mode:
    if MODE == "live":
        st.error("LIVE TRADING", icon="🔴")
    else:
        st.info("PAPER MODE", icon="📝")
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


def load_fees(trade_tickers: set) -> dict:
    """Fetch fee data from Kalshi API, filtered to only bot-traded tickers."""
    try:
        from kalshi_trader.config import load_config
        from kalshi_trader.data.kalshi_client import KalshiClient
        cfg = load_config("config.yaml")
        if cfg["mode"] != "live":
            return {"total_fees": 0.0, "by_ticker": {}}
        client = KalshiClient(cfg)
        fills = client.get_fills(limit=200)
        total_fees = 0.0
        by_ticker = {}
        for f in fills:
            t = f.get("ticker", "")
            if trade_tickers and t not in trade_tickers:
                continue
            fee = f.get("fee_cost", "0")
            fee_val = float(fee) if fee else 0.0
            total_fees += fee_val
            by_ticker[t] = by_ticker.get(t, 0.0) + fee_val
        client.close()
        return {"total_fees": total_fees, "by_ticker": by_ticker}
    except Exception:
        return {"total_fees": 0.0, "by_ticker": {}}


bot_tickers = set(trades["ticker"].unique()) if not trades.empty and "ticker" in trades.columns else set()
fee_data = load_fees(bot_tickers)

# ---------------------------------------------------------------------------
# Live Analysis (from log file)
# ---------------------------------------------------------------------------

def load_latest_analysis() -> dict:
    """Parse the live log for the most recent bracket evaluation."""
    log_path = Path("logs/live.log")
    if not log_path.exists():
        return {}

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
    except Exception:
        return {}

    import re

    # Find the most recent evaluation block
    brackets = []
    btc_price = None
    current_bracket = None
    last_eval_time = None
    event_ticker = None
    event_close = None

    for line in reversed(lines):
        if "Evaluating" in line and "brackets" in line:
            last_eval_time = line[:23]  # timestamp
            break
        if "Vol: raw=" in line and "bracket:" in line:
            bp = re.search(r"bracket_prob=([\d.]+)", line)
            mkt = re.search(r"mkt=([\d.]+)", line)
            edge = re.search(r"edge=([-\d.]+)", line)
            bkt = re.search(r"bracket: \[([^\]]+)\]", line)
            pred = re.search(r"raw=([\d.]+)", line)
            impl = re.search(r"impl=([\d.]+)", line)
            if bp and mkt and edge and bkt:
                brackets.insert(0, {
                    "bracket": bkt.group(1),
                    "our_prob": float(bp.group(1)),
                    "market_price": float(mkt.group(1)),
                    "edge": float(edge.group(1)),
                    "pred_vol": float(pred.group(1)) if pred else None,
                    "impl_vol": float(impl.group(1)) if impl else None,
                })
        if "BTC price: $" in line and btc_price is None:
            m = re.search(r"BTC price: \$([\d,.]+)", line)
            if m:
                btc_price = m.group(1)
        if "Kalshi bracket:" in line and current_bracket is None:
            m = re.search(r"Kalshi bracket: (\[[^\]]+\])", line)
            if m:
                current_bracket = m.group(1)
        if "Tracking event:" in line and event_ticker is None:
            m = re.search(r"Tracking event: (\S+)", line)
            if m:
                event_ticker = m.group(1)
            m2 = re.search(r"closes (\S+)\)", line)
            if m2:
                event_close = m2.group(1)

    # Find next eval time from logs
    next_eval_in = None
    for line in reversed(lines):
        if "Next eval in " in line:
            m = re.search(r"Next eval in (\S+)", line)
            if m:
                next_eval_in = m.group(1)
            break
        if "Eval window active" in line:
            next_eval_in = "NOW"
            break

    return {
        "brackets": brackets,
        "btc_price": btc_price,
        "current_bracket": current_bracket,
        "last_eval_time": last_eval_time,
        "event_ticker": event_ticker,
        "event_close": event_close,
        "next_eval_in": next_eval_in,
    }


analysis = load_latest_analysis()

st.subheader("Live Analysis")

# Event info row
event_close_str = analysis.get("event_close", "")
if event_close_str:
    try:
        close_dt = datetime.fromisoformat(event_close_str.replace("Z", "+00:00"))
        remaining = close_dt - datetime.now(timezone.utc)
        hours_left = remaining.total_seconds() / 3600
        close_display = close_dt.strftime("%b %d, %Y %I:%M %p UTC")
        if hours_left > 0:
            time_left = f"{hours_left:.1f}h remaining"
        else:
            time_left = "CLOSED"
    except Exception:
        close_display = event_close_str
        time_left = ""
else:
    close_display = "---"
    time_left = ""

a1, a2, a3, a4, a5 = st.columns(5)
with a1:
    st.metric("BTC Price", f"${analysis.get('btc_price', '---')}")
with a2:
    st.metric("Current Bracket", analysis.get("current_bracket", "---"))
with a3:
    st.metric("Market Closes", close_display)
    if time_left:
        st.caption(time_left)
with a4:
    # Next eval = market close - 15 min
    if event_close_str:
        try:
            close_dt_eval = datetime.fromisoformat(event_close_str.replace("Z", "+00:00"))
            eval_dt = close_dt_eval - timedelta(minutes=15)
            now_utc = datetime.now(timezone.utc)
            secs_to_eval = (eval_dt - now_utc).total_seconds()
            if secs_to_eval > 0:
                h, rem = divmod(int(secs_to_eval), 3600)
                m, s = divmod(rem, 60)
                next_eval_display = f"{h}h {m:02d}m {s:02d}s"
            elif secs_to_eval > -30:
                next_eval_display = "NOW"
            else:
                next_eval_display = "Done"
            st.metric("Next Eval", next_eval_display)
            st.caption(f"at {eval_dt.strftime('%I:%M %p UTC')}")
        except Exception:
            st.metric("Next Eval", "---")
    else:
        st.metric("Next Eval", "---")
with a5:
    eval_time = analysis.get("last_eval_time", "---")
    st.metric("Last Evaluation", eval_time[11:19] if eval_time and len(eval_time) >= 19 else "---")
    if analysis.get("event_ticker"):
        st.caption(f"Event: {analysis['event_ticker']}")

if analysis.get("brackets"):
    eval_df = pd.DataFrame(analysis["brackets"])
    eval_df["edge_pct"] = eval_df["edge"].apply(lambda e: f"{e:+.1%}")
    eval_df["signal"] = eval_df["edge"].apply(
        lambda e: "BUY" if e >= 0.04 else ("SELL" if e <= -0.04 else "---")
    )

    st.dataframe(
        eval_df,
        width="stretch",
        column_config={
            "bracket": "Bracket",
            "our_prob": st.column_config.NumberColumn("Our Prob", format="%.1%%"),
            "market_price": st.column_config.NumberColumn("Market", format="%.1%%"),
            "edge": st.column_config.NumberColumn("Edge", format="%.4f"),
            "edge_pct": "Edge %",
            "pred_vol": st.column_config.NumberColumn("Pred Vol", format="%.5f"),
            "impl_vol": st.column_config.NumberColumn("Impl Vol", format="%.5f"),
            "signal": "Signal",
        },
    )
else:
    st.caption("No bracket evaluations yet — next eval fires 15 min before market close.")

st.divider()

if trades.empty:
    st.info("Waiting for trades... Start the trader and check back soon.")
else:
    metrics = compute_metrics(trades)

    # -- KPI cards --
    total_fees = fee_data.get("total_fees", 0.0)
    net_pnl = metrics["total_pnl"] - total_fees

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    k1.metric("Net P&L", f"${net_pnl:.2f}", f"${metrics['pnl_delta']:+.2f}")
    k2.metric("Today P&L", f"${metrics['today_pnl']:.2f}")
    k3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    k4.metric("Fees Paid", f"${total_fees:.2f}")
    pf_display = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] != float("inf") else "Inf"
    k5.metric("Profit Factor", pf_display)
    k6.metric("Trades", f"{metrics['total_trades']} ({metrics['today_trades']} today)")
    k7.metric("Open", metrics["open_trades"])
    k8.metric("Max DD", f"${metrics['max_drawdown']:.2f}")

    # -- Equity curve + Daily P&L --
    eq_col, daily_col = st.columns(2)
    with eq_col:
        st.plotly_chart(equity_curve_chart(trades), width="stretch")
    with daily_col:
        st.plotly_chart(daily_pnl_chart(trades), width="stretch")

    # -- BTC price chart --
    st.plotly_chart(btc_price_chart(candles, trades), width="stretch")

    # -- Trade log table --
    st.subheader("Trade Log")
    display_df = trades.copy()
    # Format time as readable string
    if "time" in display_df.columns:
        display_df["time"] = display_df["time"].dt.strftime("%b %d %I:%M:%S %p")

    # Look up settlement time for each ticker from Kalshi API
    @st.cache_data(ttl=300)
    def get_settle_times(tickers):
        settle_map = {}
        try:
            from kalshi_trader.config import load_config
            from kalshi_trader.data.kalshi_client import KalshiClient
            cfg = load_config("config.yaml")
            if cfg["mode"] != "live":
                return settle_map
            client = KalshiClient(cfg)
            for t in tickers:
                try:
                    m = client.get_market(t)
                    close_str = m.get("close_time", "")
                    if close_str:
                        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        settle_map[t] = close_dt.strftime("%b %d %I:%M %p EST")
                except Exception:
                    pass
            client.close()
        except Exception:
            pass
        return settle_map

    settle_times = get_settle_times(tuple(display_df["ticker"].unique()))
    display_df["settles"] = display_df["ticker"].map(settle_times).fillna("---")

    display_cols = ["time", "ticker", "side", "price", "size", "settles", "edge", "fill_confidence", "predicted_vol", "implied_vol", "pnl"]
    available_cols = [c for c in display_cols if c in display_df.columns]
    if "predicted_vol" in display_df.columns and "implied_vol" in display_df.columns:
        display_df["vol_ratio"] = display_df.apply(
            lambda r: r["predicted_vol"] / r["implied_vol"]
            if r["implied_vol"] and r["implied_vol"] > 0 and r["predicted_vol"] else None,
            axis=1,
        )
        available_cols.append("vol_ratio")

    display_df["cost"] = display_df["price"] * display_df["size"] / 100
    available_cols.insert(available_cols.index("edge"), "cost")

    # Add fees from Kalshi API
    fee_by_ticker = fee_data.get("by_ticker", {})
    display_df["fees"] = display_df["ticker"].map(fee_by_ticker).fillna(0.0)
    available_cols.insert(available_cols.index("pnl"), "fees")

    display_df["result"] = display_df["pnl"].apply(
        lambda p: "WIN" if p is not None and p > 0 else ("LOSS" if p is not None and p < 0 else "PENDING")
    )
    available_cols.append("result")

    st.dataframe(
        display_df[available_cols],
        width="stretch",
        height=400,
        column_config={
            "time": "Placed",
            "settles": "Settles",
            "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
            "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
            "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
            "edge": st.column_config.NumberColumn("Edge", format="%.4f"),
            "fill_confidence": st.column_config.ProgressColumn("Fill Conf.", format="%.0f%%", min_value=0, max_value=100),
            "predicted_vol": st.column_config.NumberColumn("Pred Vol", format="%.5f"),
            "implied_vol": st.column_config.NumberColumn("Impl Vol", format="%.5f"),
            "vol_ratio": st.column_config.NumberColumn("Vol Ratio", format="%.3f"),
        },
    )

    # -- Per-bracket breakdown --
    st.subheader("Per-Bracket Performance")
    resolved = trades[trades["resolved"] == 1].copy()
    if not resolved.empty:
        bracket_stats = resolved.groupby("ticker").agg(
            trades=("pnl", "count"),
            wins=("pnl", lambda x: (x > 0).sum()),
            total_pnl=("pnl", "sum"),
            avg_edge=("edge", "mean"),
            avg_cost=("price", lambda x: x.mean()),
        ).reset_index()
        bracket_stats["win_rate"] = (bracket_stats["wins"] / bracket_stats["trades"] * 100).round(1)
        bracket_stats = bracket_stats.sort_values("total_pnl", ascending=False)

        st.dataframe(
            bracket_stats[["ticker", "trades", "win_rate", "total_pnl", "avg_edge", "avg_cost"]],
            width="stretch",
            column_config={
                "total_pnl": st.column_config.NumberColumn("Total P&L", format="$%.2f"),
                "avg_edge": st.column_config.NumberColumn("Avg Edge", format="%.4f"),
                "avg_cost": st.column_config.NumberColumn("Avg Price (c)", format="%.0f"),
                "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
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
            st.plotly_chart(fig_conf, width="stretch")
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
        st.plotly_chart(vol_scatter(trades), width="stretch")
    with vcol2:
        st.plotly_chart(edge_histogram(trades), width="stretch")

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
