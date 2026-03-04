from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker: str, days: int = 300) -> pd.DataFrame:
    """Fetch daily OHLC data from yfinance. Returns empty DataFrame on failure."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()

        # Flatten multi-level columns if present (yfinance >= 0.2.40)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()


def get_current_price(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_52_week_range(df: pd.DataFrame) -> tuple[float, float] | None:
    if len(df) < 20:
        return None
    last_252 = df["Close"].tail(252)
    return float(last_252.min()), float(last_252.max())
