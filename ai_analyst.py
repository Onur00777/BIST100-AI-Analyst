"""
AI analyst module powered by Google Gemini (google-genai SDK).

Evaluates today's trades against technical market context and returns
a Turkish Markdown coaching report.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Preferred model order. Older IDs (2.5 / 2.0 / 1.5) may 404 for new API keys;
# newer Flash variants are tried first, then legacy fallbacks.
# Override anytime with GEMINI_MODEL in .env
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
]

# System instructions stay in English; the model MUST reply in Turkish.
SYSTEM_INSTRUCTION = """
You are an experienced BIST (Borsa Istanbul) trading coach and technical analyst.
Your job is to review a retail trader's daily activity with constructive honesty.

Evaluation rules (always follow):
1. Trading discipline: Did the trader overtrade, chase price, or ignore risk size?
2. Entry quality: For BUY trades, check whether price was near support, near the
   60-day low, below EMA-20/EMA-50, or whether RSI(14) was oversold (<30) or
   overbought (>70). Praise patient entries; warn about FOMO buys into strength.
3. Concentration risk: Highlight if too much capital or too many trades cluster
   in one ticker or a single sector theme.
4. Tomorrow levels: For each actively discussed ticker (or the most important one),
   give exactly 1 key Support and 1 key Resistance level for tomorrow, derived
   from the provided high_60d / low_60d / EMA / current price context.
5. Tone: Encouraging, clear, and practical — never hype, never financial advice
   disclaimers in every sentence. Be a calm mentor.

Language & format constraints (CRITICAL):
- System reasoning may be in English, but the FINAL OUTPUT MUST be entirely in
  simple, clear TURKISH.
- Use clean Markdown: ## headers, bullet points, and short paragraphs.
- Suggested structure:
  ## Günlük Özet
  ## Disiplin Değerlendirmesi
  ## İşlem Bazlı Teknik Yorum
  ## Konsantrasyon / Risk Notu
  ## Yarın İçin Destek & Direnç
  ## Tek Cümlelik Motivasyon
"""


def _format_trades(trades_df: pd.DataFrame) -> str:
    """Serialize today's trades into a compact text block for the prompt."""
    if trades_df is None or trades_df.empty:
        return "No trades logged today."

    lines = []
    for _, row in trades_df.iterrows():
        notes = row.get("notes") or ""
        note_part = f" | notes: {notes}" if notes else ""
        lines.append(
            f"- {row['action']} {row['quantity']} x {row['ticker']} "
            f"@ {row['price']} TRY on {row['date']}{note_part}"
        )
    return "\n".join(lines)


def _format_market_data(market_data_dict: dict[str, dict[str, Any]]) -> str:
    """Serialize per-ticker technical summaries for the prompt."""
    if not market_data_dict:
        return "No market data available."

    blocks = []
    for ticker, data in market_data_dict.items():
        if not data.get("success"):
            blocks.append(f"[{ticker}] ERROR: {data.get('error', 'unknown')}")
            continue
        blocks.append(
            f"[{ticker}]\n"
            f"  current_price={data.get('current_price')}\n"
            f"  daily_change_pct={data.get('daily_change_pct')}%\n"
            f"  rsi_14={data.get('rsi_14')}\n"
            f"  ema_20={data.get('ema_20')}\n"
            f"  ema_50={data.get('ema_50')}\n"
            f"  high_60d={data.get('high_60d')}\n"
            f"  low_60d={data.get('low_60d')}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
) -> str:
    """Compose the user message sent to Gemini."""
    return (
        "Analyze today's BIST trading session for this retail portfolio.\n\n"
        "=== TODAY'S TRADES ===\n"
        f"{_format_trades(trades_df)}\n\n"
        "=== MARKET / TECHNICAL SNAPSHOT ===\n"
        f"{_format_market_data(market_data_dict)}\n\n"
        "Produce the Turkish Markdown coaching report now."
    )


def _normalize_model_id(name: str) -> str:
    """Strip optional 'models/' prefix from a model resource name."""
    name = (name or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return name


def _candidate_models(explicit: Optional[str] = None) -> list[str]:
    """Build ordered unique model candidates from env + defaults."""
    ordered: list[str] = []
    for raw in (
        explicit,
        os.getenv("GEMINI_MODEL"),
        *DEFAULT_MODEL_CANDIDATES,
    ):
        mid = _normalize_model_id(raw or "")
        if mid and mid not in ordered:
            ordered.append(mid)
    return ordered


def _discover_flash_models(client) -> list[str]:
    """List generateContent-capable flash models from the API (best effort)."""
    found: list[str] = []
    try:
        for model in client.models.list():
            name = _normalize_model_id(getattr(model, "name", "") or "")
            if not name:
                continue
            methods = getattr(model, "supported_actions", None) or getattr(
                model, "supported_generation_methods", None
            )
            # Keep flash-family text models; skip embeddings / image-only
            if "flash" not in name.lower():
                continue
            if any(x in name.lower() for x in ("embed", "image", "tts", "live")):
                continue
            if methods and "generateContent" not in str(methods):
                continue
            if name not in found:
                found.append(name)
    except Exception:  # noqa: BLE001
        return []
    return found


def _is_model_unavailable(exc: Exception) -> bool:
    """True when the API rejects the model id (404 / NOT_FOUND)."""
    text = str(exc).lower()
    return (
        "404" in text
        or "not_found" in text
        or "not found" in text
        or "no longer available" in text
    )


def analyze_daily_performance(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Send today's trades + technical context to Gemini and return Markdown text.

    Args:
        trades_df: DataFrame of today's trades (from database.get_todays_trades).
        market_data_dict: Mapping of ticker -> get_stock_summary() result.
        api_key: Optional override; otherwise reads GEMINI_API_KEY from env.
        model: Preferred Gemini model id. Falls back through DEFAULT_MODEL_CANDIDATES
               and GEMINI_MODEL env var when unavailable (404).

    Returns:
        Turkish Markdown analysis string, or an error message on failure.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() in ("", "your_gemini_api_key_here"):
        return (
            "⚠️ **API Anahtarı Eksik**\n\n"
            "`.env` dosyanıza geçerli bir `GEMINI_API_KEY` ekleyin. "
            "Anahtarı [Google AI Studio](https://aistudio.google.com/apikey) "
            "üzerinden ücretsiz alabilirsiniz."
        )

    if (trades_df is None or trades_df.empty) and not market_data_dict:
        return (
            "📭 **Analiz için veri yok**\n\n"
            "Bugün henüz işlem kaydı bulunmuyor. Sol menüden bir işlem ekleyip "
            "tekrar deneyin."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return (
            "⚠️ **Paket Eksik**\n\n"
            "`google-genai` yüklü değil. Şunu çalıştırın: "
            "`pip install google-genai`"
        )

    user_prompt = _build_user_prompt(trades_df, market_data_dict)
    candidates = _candidate_models(model)

    try:
        client = genai.Client(api_key=key)
        # Append any live flash models not already in the candidate list
        for discovered in _discover_flash_models(client):
            if discovered not in candidates:
                candidates.append(discovered)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.6,
            max_output_tokens=4096,
        )

        last_error: Optional[Exception] = None
        tried: list[str] = []

        for model_id in candidates:
            tried.append(model_id)
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=config,
                )
                text = (response.text or "").strip()
                if not text:
                    continue
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if _is_model_unavailable(exc):
                    continue  # try next model
                # Non-model errors (auth, quota, network) — stop early
                break

        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            "Gemini API çağrısı başarısız oldu.\n\n"
            f"**Denenen modeller:** `{', '.join(tried)}`\n\n"
            f"```\n{last_error}\n```\n\n"
            "`.env` içine çalışan bir model yazabilirsiniz, örn:\n"
            "`GEMINI_MODEL=gemini-2.5-flash-lite`\n\n"
            "İnternet bağlantınızı ve API anahtarınızı da kontrol edin."
        )

    except Exception as exc:  # noqa: BLE001 — surface API / network failures to the UI
        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            f"Gemini API çağrısı başarısız oldu:\n\n```\n{exc}\n```\n\n"
            "İnternet bağlantınızı ve API anahtarınızı kontrol edin."
        )
