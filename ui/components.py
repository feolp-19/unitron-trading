import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analysis.technical import compute_rsi, compute_sma
from ui.translations import T


def render_traffic_light(action: str, confidence: float):
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


def render_price_chart(
    df: pd.DataFrame,
    asset_name: str,
    supports: list[float] | None = None,
    resistances: list[float] | None = None,
):
    """Render candlestick chart with SMA 20/50/200, RSI, and S/R lines."""
    if df.empty or len(df) < 50:
        st.warning(T["insufficient_data"])
        return

    display_df = df.tail(120).copy()
    close_full = df["Close"]

    rsi_display = compute_rsi(close_full).tail(120)
    sma_20_display = compute_sma(close_full, 20).tail(120)
    sma_50_display = compute_sma(close_full, 50).tail(120)
    sma_200_display = compute_sma(close_full).tail(120)

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
            x=sma_20_display.index, y=sma_20_display,
            name="SMA 20", line=dict(color="#FFD600", width=1, dash="dot"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sma_50_display.index, y=sma_50_display,
            name="SMA 50", line=dict(color="#FF9100", width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sma_200_display.index, y=sma_200_display,
            name="SMA 200", line=dict(color="#2196F3", width=2),
        ),
        row=1, col=1,
    )

    # Support & Resistance lines
    if supports:
        for i, s in enumerate(supports[:3]):
            fig.add_hline(
                y=s, line_dash="dash", line_color="#00C853", opacity=0.6,
                annotation_text=f"S{i+1}: {s:,.2f}",
                annotation_position="bottom right",
                annotation_font_color="#00C853",
                row=1, col=1,
            )
    if resistances:
        for i, r in enumerate(resistances[:3]):
            fig.add_hline(
                y=r, line_dash="dash", line_color="#FF1744", opacity=0.6,
                annotation_text=f"R{i+1}: {r:,.2f}",
                annotation_position="top right",
                annotation_font_color="#FF1744",
                row=1, col=1,
            )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=rsi_display.index, y=rsi_display,
            name="RSI", line=dict(color="#AB47BC", width=2),
        ),
        row=2, col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="#FF1744", opacity=0.3, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00C853", opacity=0.3, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#666", opacity=0.3, row=2, col=1)

    fig.update_layout(
        height=650,
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
    if not warnings:
        return

    if title:
        st.markdown(f"**{title}**")

    for warning in warnings:
        st.warning(warning)


def render_trading_plan(trading_plan, action: str):
    """Render the full Handelsplan (Trading Plan) as a styled container."""
    if trading_plan is None:
        return

    if action == "BULL":
        color = "#00C853"
        direction = "BULL"
    else:
        color = "#FF1744"
        direction = "BEAR"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}10, {color}05);
            border: 1px solid {color}33;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <h4 style="color: {color}; margin-bottom: 16px;">
                EXIT-STRATEGI ({direction})
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ingångspris", f"{trading_plan.entry_price:,.2f}")
    with col2:
        st.metric("Stop-Loss", f"{trading_plan.stop_loss:,.2f}",
                   delta=f"-{trading_plan.risk_amount:,.2f}" if action == "BULL" else f"+{trading_plan.risk_amount:,.2f}")
    with col3:
        st.metric("Målkurs", f"{trading_plan.take_profit:,.2f}",
                   delta=f"+{trading_plan.reward_amount:,.2f}" if action == "BULL" else f"-{trading_plan.reward_amount:,.2f}")
    with col4:
        st.metric("Risk/Reward", trading_plan.risk_reward_ratio)

    st.markdown(f"**Stop-Loss ({trading_plan.stop_loss_method}):** {trading_plan.stop_loss_reasoning}")
    st.markdown(f"**Målkurs ({trading_plan.take_profit_method}):** {trading_plan.take_profit_reasoning}")
    st.markdown(f"**Trailing Stop:** {trading_plan.trailing_stop_reasoning}")
