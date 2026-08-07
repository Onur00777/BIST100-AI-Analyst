"""
Market data helpers for BIST stocks via yfinance.

Fetches recent price history and computes key technical indicators
(RSI-14, EMA-20, EMA-50) used by the dashboard and AI analyst.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yfinance as yf

# Try pandas-ta first; fall back to manual calculations if unavailable
try:
    import pandas_ta as ta

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


# Popular BIST 100 tickers for dropdown convenience
BIST100_TICKERS = [
    "THYAO", "GARAN", "AKBNK", "ISCTR", "YKBNK", "SAHOL", "KCHOL",
    "EREGL", "SISE", "TUPRS", "BIMAS", "ASELS", "TCELL", "KOZAL",
    "PGSUS", "FROTO", "TOASO", "SASA", "PETKM", "HEKTS", "ENKAI",
    "TAVHL", "DOHOL", "EKGYO", "VESTL", "MGROS", "ARCLK", "KRDMD",
    "HALKB", "VAKBN", "TTKOM", "ULKER", "CCOLA", "SOKM", "GUBRF",
]


def normalize_ticker(ticker: str) -> str:
    """Ensure BIST tickers have the Yahoo Finance '.IS' suffix."""
    ticker = ticker.upper().strip()
    if not ticker.endswith(".IS"):
        ticker = f"{ticker}.IS"
    return ticker


def _compute_rsi(close: pd.Series, length: int = 14) -> Optional[float]:
    """Compute RSI manually if pandas-ta is unavailable."""
    if len(close) < length + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def _compute_ema(close: pd.Series, length: int) -> Optional[float]:
    """Compute EMA manually if pandas-ta is unavailable."""
    if len(close) < length:
        return None
    ema = close.ewm(span=length, adjust=False).mean()
    val = ema.iloc[-1]
    return float(val) if pd.notna(val) else None


def get_stock_summary(ticker: str, lookback_days: int = 60) -> dict[str, Any]:
    """
    Fetch ~60 trading days of OHLCV data and return a summary dict.

    Returns keys:
        ticker, yahoo_ticker, current_price, previous_close,
        daily_change_pct, rsi_14, ema_20, ema_50,
        high_60d, low_60d, volume, success, error

    On failure (invalid ticker / no internet), success=False and
    error contains a human-readable message; numeric fields are None.
    """
    yahoo_ticker = normalize_ticker(ticker)
    bare = yahoo_ticker.replace(".IS", "")

    result: dict[str, Any] = {
        "ticker": bare,
        "yahoo_ticker": yahoo_ticker,
        "current_price": None,
        "previous_close": None,
        "daily_change_pct": None,
        "rsi_14": None,
        "ema_20": None,
        "ema_50": None,
        "high_60d": None,
        "low_60d": None,
        "volume": None,
        "success": False,
        "error": None,
    }

    try:
        # Fetch a bit more than lookback to ensure enough bars after weekends/holidays
        hist = yf.download(
            yahoo_ticker,
            period=f"{lookback_days + 30}d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if hist is None or hist.empty:
            result["error"] = f"'{yahoo_ticker}' için veri bulunamadı. Ticker geçersiz olabilir."
            return result

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist.dropna(subset=["Close"]).tail(lookback_days)
        if len(hist) < 2:
            result["error"] = f"'{yahoo_ticker}' için yeterli fiyat verisi yok."
            return result

        close = hist["Close"].astype(float)
        current = float(close.iloc[-1])
        previous = float(close.iloc[-2])
        change_pct = ((current - previous) / previous) * 100 if previous else 0.0

        # Technical indicators via pandas-ta or manual fallback
        if HAS_PANDAS_TA:
            rsi_series = ta.rsi(close, length=14)
            ema20_series = ta.ema(close, length=20)
            ema50_series = ta.ema(close, length=50)
            rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and pd.notna(rsi_series.iloc[-1]) else None
            ema20_val = float(ema20_series.iloc[-1]) if ema20_series is not None and pd.notna(ema20_series.iloc[-1]) else None
            ema50_val = float(ema50_series.iloc[-1]) if ema50_series is not None and pd.notna(ema50_series.iloc[-1]) else None
        else:
            rsi_val = _compute_rsi(close, 14)
            ema20_val = _compute_ema(close, 20)
            ema50_val = _compute_ema(close, 50)

        volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None

        result.update(
            {
                "current_price": round(current, 4),
                "previous_close": round(previous, 4),
                "daily_change_pct": round(change_pct, 2),
                "rsi_14": round(rsi_val, 2) if rsi_val is not None else None,
                "ema_20": round(ema20_val, 4) if ema20_val is not None else None,
                "ema_50": round(ema50_val, 4) if ema50_val is not None else None,
                "high_60d": round(float(close.max()), 4),
                "low_60d": round(float(close.min()), 4),
                "volume": volume,
                "success": True,
                "error": None,
            }
        )
        return result

    except Exception as exc:  # noqa: BLE001 — surface any network/parse failure cleanly
        result["error"] = f"Piyasa verisi alınamadı ({yahoo_ticker}): {exc}"
        return result


def get_bist100_index_summary() -> dict[str, Any]:
    """
    Fetch BIST 100 index (^XU100) daily change for the dashboard metric.

    Yahoo Finance symbol for BIST 100 is typically 'XU100.IS'.
    Falls back gracefully if unavailable.
    """
    candidates = ["XU100.IS", "^XU100"]
    last_error = None

    for symbol in candidates:
        try:
            hist = yf.download(
                symbol,
                period="5d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if hist is None or hist.empty or len(hist) < 2:
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close = hist["Close"].astype(float).dropna()
            if len(close) < 2:
                continue

            current = float(close.iloc[-1])
            previous = float(close.iloc[-2])
            change_pct = ((current - previous) / previous) * 100 if previous else 0.0

            return {
                "symbol": symbol,
                "current_price": round(current, 2),
                "daily_change_pct": round(change_pct, 2),
                "success": True,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue

    return {
        "symbol": "XU100.IS",
        "current_price": None,
        "daily_change_pct": None,
        "success": False,
        "error": last_error or "BIST 100 endeks verisi alınamadı.",
    }


def get_summaries_for_tickers(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Batch-fetch summaries for a list of bare tickers.

    Returns a dict keyed by bare ticker symbol.
    """
    unique = []
    seen = set()
    for t in tickers:
        bare = t.upper().strip().replace(".IS", "")
        if bare and bare not in seen:
            seen.add(bare)
            unique.append(bare)

    return {t: get_stock_summary(t) for t in unique}
