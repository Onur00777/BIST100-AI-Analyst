"""
Market data helpers for BIST stocks via yfinance.

- Universal ticker normalization (`format_bist_ticker`): accepts any BIST code
  or common company name (e.g. 'Alcatel' -> 'ALCTL.IS') and appends '.IS'.
- Technical indicators (RSI-14, EMA-20, EMA-50, daily % change) are computed
  with pure pandas for full Python 3.10–3.14 compatibility.
- Every fetch failure returns a graceful fallback dict instead of raising.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yfinance as yf

# BIST 100 constituents (2026 Q3), sorted alphabetically.
# Keep this list explicit so the dropdown remains deterministic and fast.
# Stocks outside the index can still be entered through the manual input.
BIST100_TICKERS = [
    "AEFES",
    "AKBNK",
    "AKSA",
    "AKSEN",
    "ALARK",
    "ALTNY",
    "ANSGR",
    "ARCLK",
    "ASELS",
    "ASTOR",
    "BALSU",
    "BERA",
    "BIMAS",
    "BRSAN",
    "BRYAT",
    "BSOKE",
    "BTCIM",
    "CANTE",
    "CCOLA",
    "CIMSA",
    "CVKMD",
    "CWENE",
    "DAPGM",
    "DOAS",
    "DOHOL",
    "DSTKF",
    "ECILC",
    "EFOR",
    "EKGYO",
    "ENERY",
    "ENJSA",
    "ENKAI",
    "EREGL",
    "ESEN",
    "EUPWR",
    "EUREN",
    "FENER",
    "FROTO",
    "GARAN",
    "GENIL",
    "GESAN",
    "GLRMK",
    "GRSEL",
    "GRTHO",
    "GSRAY",
    "GUBRF",
    "HALKB",
    "HEKTS",
    "IEYHO",
    "ISCTR",
    "ISMEN",
    "IZENR",
    "KCHOL",
    "KLRHO",
    "KRDMD",
    "KTLEV",
    "KUYAS",
    "MAGEN",
    "MAVI",
    "MGROS",
    "MIATK",
    "MPARK",
    "OBAMS",
    "ODAS",
    "ODINE",
    "OTKAR",
    "OYAKC",
    "PAHOL",
    "PASEU",
    "PATEK",
    "PETKM",
    "PGSUS",
    "PSGYO",
    "QUAGR",
    "RALYH",
    "REEDR",
    "SAHOL",
    "SARKY",
    "SASA",
    "SISE",
    "SKBNK",
    "SOKM",
    "TAVHL",
    "TCELL",
    "THYAO",
    "TKFEN",
    "TOASO",
    "TRALT",
    "TRENJ",
    "TRMET",
    "TSKB",
    "TTKOM",
    "TUKAS",
    "TUPRS",
    "TURSG",
    "ULKER",
    "VAKBN",
    "VESTL",
    "YKBNK",
    "ZOREN",
]

# Common company names -> BIST ticker codes (keys are ASCII-uppercased)
COMMON_NAME_MAP = {
    "ALCATEL": "ALCTL",
    "ALCATEL LUCENT": "ALCTL",
    "ALCATEL-LUCENT": "ALCTL",
    "ASELSAN": "ASELS",
    "TURKCELL": "TCELL",
    "GARANTI": "GARAN",
    "GARANTI BBVA": "GARAN",
    "AKBANK": "AKBNK",
    "IS BANKASI": "ISCTR",
    "ISBANK": "ISCTR",
    "THY": "THYAO",
    "TURK HAVA YOLLARI": "THYAO",
    "TURKISH AIRLINES": "THYAO",
    "SISECAM": "SISE",
    "EREGLI": "EREGL",
    "TUPRAS": "TUPRS",
    "BIM": "BIMAS",
    "PEGASUS": "PGSUS",
    "FORD OTOSAN": "FROTO",
    "TOFAS": "TOASO",
    "KOC HOLDING": "KCHOL",
    "SABANCI": "SAHOL",
    "SABANCI HOLDING": "SAHOL",
    "ARCELIK": "ARCLK",
    "VESTEL": "VESTL",
    "HALKBANK": "HALKB",
    "VAKIFBANK": "VAKBN",
    "YAPI KREDI": "YKBNK",
    "MIGROS": "MGROS",
    "ULKER": "ULKER",
    "COCA COLA": "CCOLA",
    "COCA-COLA ICECEK": "CCOLA",
    "TURK TELEKOM": "TTKOM",
}

# Turkish characters -> ASCII equivalents for robust name matching
_TR_TRANSLATION = str.maketrans(
    "çÇğĞıİöÖşŞüÜ",
    "cCgGiIoOsSuU",
)


def _ascii_upper(text: str) -> str:
    """Uppercase text and fold Turkish characters to ASCII."""
    return (text or "").translate(_TR_TRANSLATION).upper().strip()


def format_bist_ticker(ticker: str) -> str:
    """
    Normalize any user input into a Yahoo Finance BIST symbol ('XXX.IS').

    Steps:
      1. Trim spaces, uppercase, fold Turkish characters.
      2. Map common company names (e.g. 'ALCATEL' -> 'ALCTL').
      3. Append '.IS' when missing.

    Examples:
        'Alcatel'  -> 'ALCTL.IS'
        ' thyao '  -> 'THYAO.IS'
        'ASELS.IS' -> 'ASELS.IS'
    """
    cleaned = _ascii_upper(ticker)
    if not cleaned:
        return ""

    # Drop an existing suffix for lookup, remember to re-add later
    base = cleaned[:-3] if cleaned.endswith(".IS") else cleaned
    base = base.strip().strip(".")

    # Company-name mapping (with and without inner spaces)
    if base in COMMON_NAME_MAP:
        base = COMMON_NAME_MAP[base]
    else:
        compact = " ".join(base.split())
        if compact in COMMON_NAME_MAP:
            base = COMMON_NAME_MAP[compact]
        else:
            # Ticker codes never contain spaces — strip them if present
            base = base.replace(" ", "")

    return f"{base}.IS" if base else ""


def bare_ticker(ticker: str) -> str:
    """Return the normalized ticker code without the '.IS' suffix."""
    full = format_bist_ticker(ticker)
    return full[:-3] if full.endswith(".IS") else full


# Backward-compatible alias
def normalize_ticker(ticker: str) -> str:
    """Alias of format_bist_ticker (kept for backward compatibility)."""
    return format_bist_ticker(ticker)


# ---------------------------------------------------------------------------
# Pure-pandas technical indicators (no pandas-ta / numba dependency)
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, length: int = 14) -> Optional[float]:
    """Wilder's RSI computed with pandas ewm; returns latest value or None."""
    if close is None or len(close) < length + 1:
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


def compute_ema(close: pd.Series, length: int) -> Optional[float]:
    """Exponential moving average; returns latest value or None."""
    if close is None or len(close) < length:
        return None
    ema = close.ewm(span=length, adjust=False).mean()
    val = ema.iloc[-1]
    return float(val) if pd.notna(val) else None


def _empty_summary(ticker: str, yahoo_ticker: str, error: str) -> dict[str, Any]:
    """Graceful fallback summary when data cannot be fetched."""
    return {
        "ticker": ticker,
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
        "error": error,
    }


def get_stock_summary(ticker: str, lookback_days: int = 60) -> dict[str, Any]:
    """
    Fetch ~60 trading days of OHLCV data and return a summary dict.

    Returns keys:
        ticker, yahoo_ticker, current_price, previous_close,
        daily_change_pct, rsi_14, ema_20, ema_50,
        high_60d, low_60d, volume, success, error

    Never raises: on failure (invalid ticker / no internet), success=False
    and error contains a human-readable Turkish message.
    """
    yahoo_ticker = format_bist_ticker(ticker)
    bare = yahoo_ticker[:-3] if yahoo_ticker.endswith(".IS") else yahoo_ticker

    if not yahoo_ticker:
        return _empty_summary(ticker, "", "Geçersiz hisse kodu.")

    try:
        # Fetch extra days to ensure enough bars after weekends/holidays
        hist = yf.download(
            yahoo_ticker,
            period=f"{lookback_days + 30}d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if hist is None or hist.empty:
            return _empty_summary(
                bare,
                yahoo_ticker,
                f"'{yahoo_ticker}' için veri bulunamadı. Ticker geçersiz olabilir.",
            )

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist.dropna(subset=["Close"]).tail(lookback_days)
        if len(hist) < 2:
            return _empty_summary(
                bare, yahoo_ticker, f"'{yahoo_ticker}' için yeterli fiyat verisi yok."
            )

        close = hist["Close"].astype(float)
        current = float(close.iloc[-1])
        previous = float(close.iloc[-2])
        change_pct = ((current - previous) / previous) * 100 if previous else 0.0

        rsi_val = compute_rsi(close, 14)
        ema20_val = compute_ema(close, 20)
        ema50_val = compute_ema(close, 50)
        volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None

        return {
            "ticker": bare,
            "yahoo_ticker": yahoo_ticker,
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

    except Exception as exc:  # noqa: BLE001 — never crash the app on market data
        return _empty_summary(
            bare, yahoo_ticker, f"Piyasa verisi alınamadı ({yahoo_ticker}): {exc}"
        )


def get_bist100_index_summary() -> dict[str, Any]:
    """
    Fetch BIST 100 index (XU100) daily change for the dashboard metric.

    Tries 'XU100.IS' first, then '^XU100'. Falls back gracefully.
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
    Batch-fetch summaries for a list of tickers (codes or common names).

    Returns a dict keyed by normalized bare ticker symbol.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        bare = bare_ticker(t)
        if bare and bare not in seen:
            seen.add(bare)
            unique.append(bare)

    return {t: get_stock_summary(t) for t in unique}


# ---------------------------------------------------------------------------
# Sector / company news via yfinance
# ---------------------------------------------------------------------------
def _extract_news_item(raw: dict) -> Optional[dict[str, Any]]:
    """
    Normalize a yfinance news payload (old flat dict OR nested 'content' shape)
    into {title, publisher, summary, link, published}.
    """
    if not isinstance(raw, dict):
        return None

    # Newer yfinance shape: {'id': ..., 'content': {...}}
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw

    title = (
        content.get("title")
        or content.get("headline")
        or raw.get("title")
        or ""
    ).strip()
    if not title:
        return None

    publisher = ""
    provider = content.get("provider") or raw.get("provider")
    if isinstance(provider, dict):
        publisher = str(provider.get("displayName") or provider.get("name") or "")
    if not publisher:
        publisher = str(
            content.get("publisher")
            or raw.get("publisher")
            or content.get("source")
            or ""
        )

    summary = (
        content.get("summary")
        or content.get("description")
        or raw.get("summary")
        or ""
    )
    if isinstance(summary, str):
        summary = summary.strip()
    else:
        summary = ""

    # Link extraction
    link = ""
    click = content.get("clickThroughUrl") or content.get("canonicalUrl")
    if isinstance(click, dict):
        link = str(click.get("url") or "")
    if not link:
        link = str(content.get("link") or raw.get("link") or content.get("url") or "")

    published = (
        content.get("pubDate")
        or content.get("displayTime")
        or raw.get("providerPublishTime")
        or raw.get("published")
        or ""
    )
    if isinstance(published, (int, float)):
        try:
            from datetime import datetime, timezone

            published = datetime.fromtimestamp(published, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (OSError, OverflowError, ValueError):
            published = str(published)

    return {
        "title": title[:240],
        "publisher": (publisher or "Bilinmiyor")[:80],
        "summary": (summary or "")[:500],
        "link": link,
        "published": str(published)[:32],
    }


def get_stock_news(ticker: str, limit: int = 5) -> dict[str, Any]:
    """
    Fetch recent news headlines for a BIST ticker via yfinance.

    Returns:
        {
          ticker, yahoo_ticker, news: [{title, publisher, summary, link, published}],
          success, error
        }
    Never raises — empty news list on failure.
    """
    yahoo_ticker = format_bist_ticker(ticker)
    bare = yahoo_ticker[:-3] if yahoo_ticker.endswith(".IS") else yahoo_ticker
    result: dict[str, Any] = {
        "ticker": bare,
        "yahoo_ticker": yahoo_ticker,
        "news": [],
        "success": False,
        "error": None,
    }

    if not yahoo_ticker:
        result["error"] = "Geçersiz hisse kodu."
        return result

    try:
        stock = yf.Ticker(yahoo_ticker)
        raw_news = getattr(stock, "news", None) or []
        if not isinstance(raw_news, list) or not raw_news:
            result["error"] = f"'{yahoo_ticker}' için haber bulunamadı."
            return result

        items: list[dict[str, Any]] = []
        for raw in raw_news:
            item = _extract_news_item(raw)
            if item:
                items.append(item)
            if len(items) >= max(1, int(limit)):
                break

        if not items:
            result["error"] = f"'{yahoo_ticker}' haberleri okunamadı."
            return result

        result["news"] = items
        result["success"] = True
        return result

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Haberler alınamadı ({yahoo_ticker}): {exc}"
        return result


def get_news_for_tickers(
    tickers: list[str], limit_per_ticker: int = 5
) -> dict[str, dict[str, Any]]:
    """Batch-fetch news for multiple tickers; keyed by bare ticker."""
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        bare = bare_ticker(t)
        if bare and bare not in out:
            out[bare] = get_stock_news(bare, limit=limit_per_ticker)
    return out
