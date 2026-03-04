import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analysis.technical import compute_rsi, compute_sma
from ui.translations import T


def render_traffic_light(action: str, confidence: float):
    """Render the main signal as a large traffic-light indicator."""
    if action == "BULL":
        color = "#00C853"
        icon = "🟢"
        text = T["signal_bull"]
    elif action == "BEAR":
        color = "#FF1744"
        icon = "🔴"
        text = T["signal_bear"]
    else:
        color = "#757575"
        icon = "⚪"
        text = T["signal_none"]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 6px solid {color};
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if action != "NONE":
        st.metric(T["confidence_label"], f"{confidence:.0%}")


def render_confidence_bar(confidence: float):
    """Render a horizontal confidence bar."""
    pct = int(confidence * 100)
    if pct >= 70:
        color = "#00C853"
    elif pct >= 40:
        color = "#FFD600"
    else:
        color = "#FF1744"

    st.markdown(
        f"""
        <div style="background: #1A1D23; border-radius: 8px; height: 24px; overflow: hidden; margin: 8px 0;">
            <div style="
                background: {color};
                width: {pct}%;
                height: 100%;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: bold;
                color: #000;
            ">{pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_chart(df: pd.DataFrame, asset_name: str):
    """Render candlestick chart with SMA overlay and RSI subplot."""
    if df.empty or len(df) < 50:
        st.warning(T["insufficient_data"])
        return

    display_df = df.tail(120).copy()
    rsi_full = compute_rsi(df["Close"])
    sma_full = compute_sma(df["Close"])

    rsi_display = rsi_full.tail(120)
    sma_display = sma_full.tail(120)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{asset_name} — Prisdiagram", "RSI (14)"],
    )

    fig.add_trace(
        go.Candlestick(
            x=display_df.index,
            open=display_df["Open"],
            high=display_df["High"],
            low=display_df["Low"],
            close=display_df["Close"],
            name="OHLC",
            increasing_line_color="#00C853",
            decreasing_line_color="#FF1744",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sma_display.index,
            y=sma_display,
            name="200 SMA",
            line=dict(color="#2196F3", width=2),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=rsi_display.index,
            y=rsi_display,
            name="RSI",
            line=dict(color="#AB47BC", width=2),
        ),
        row=2, col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="#FF1744", opacity=0.3, row=2, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="#FF1744", opacity=0.5, row=2, col=1)
    fig.add_hline(y=45, line_dash="dot", line_color="#00C853", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00C853", opacity=0.3, row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_headline_table(headline_details: list):
    """Render color-coded headline sentiment table."""
    if not headline_details:
        return

    rows = []
    for h in headline_details:
        if isinstance(h, dict):
            sentiment = h.get("sentiment", "neutral")
            headline = h.get("headline", "")
            relevance = h.get("relevance", 0)
            reasoning = h.get("reasoning", "")
        else:
            sentiment = h.sentiment
            headline = h.headline
            relevance = h.relevance
            reasoning = h.reasoning

        if sentiment == "positive":
            indicator = "🟢"
        elif sentiment == "negative":
            indicator = "🔴"
        else:
            indicator = "⚪"

        rows.append({
            "": indicator,
            "Rubrik": headline[:80],
            "Relevans": f"{relevance:.0%}",
            "Bedömning": reasoning[:60],
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


def render_warning_box(warnings: list[str], title: str | None = None):
    """Render a styled warning box."""
    if not warnings:
        return

    if title:
        st.markdown(f"**{title}**")

    for warning in warnings:
        st.warning(warning)


def render_info_metric(label: str, value, delta: str | None = None):
    """Render a single metric."""
    if value is not None:
        st.metric(label, value, delta=delta)
