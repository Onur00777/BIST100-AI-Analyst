"""
AI analyst module powered by Google Gemini (google-genai SDK).

Evaluates today's trades against technical market context and returns
a Turkish Markdown coaching report.

API key resolution order:
  1. Explicit function argument
  2. Streamlit Cloud / local secrets (st.secrets["GEMINI_API_KEY"])
  3. Environment variables (.env via python-dotenv): GEMINI_API_KEY, GOOGLE_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Preferred model order. Override anytime with GEMINI_MODEL in secrets/.env
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


def _clean_api_key(raw: Optional[str]) -> Optional[str]:
    """Strip whitespace / wrapping quotes from an API key value."""
    if raw is None:
        return None
    key = str(raw).strip()
    if (key.startswith('"') and key.endswith('"')) or (
        key.startswith("'") and key.endswith("'")
    ):
        key = key[1:-1].strip()
    # Reject placeholders / empty
    if not key or key.lower() in {
        "your_gemini_api_key_here",
        "none",
        "null",
        "undefined",
    }:
        return None
    return key


def get_gemini_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """
    Resolve GEMINI_API_KEY from Streamlit Secrets, then environment variables.

    Order:
      1. explicit argument
      2. st.secrets["GEMINI_API_KEY"] (Streamlit Cloud / secrets.toml)
      3. os.environ GEMINI_API_KEY / GOOGLE_API_KEY (.env supported via dotenv)
    """
    cleaned = _clean_api_key(explicit)
    if cleaned:
        return cleaned

    # Streamlit Cloud / local .streamlit/secrets.toml
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            cleaned = _clean_api_key(st.secrets["GEMINI_API_KEY"])
            if cleaned:
                return cleaned
        # Optional alternate secret name
        if "GOOGLE_API_KEY" in st.secrets:
            cleaned = _clean_api_key(st.secrets["GOOGLE_API_KEY"])
            if cleaned:
                return cleaned
    except Exception:
        pass

    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        cleaned = _clean_api_key(os.getenv(env_name))
        if cleaned:
            return cleaned

    return None


def get_gemini_client(api_key: Optional[str] = None):
    """
    Build a google-genai Client using Streamlit Secrets or env API key.

    Returns:
        (client, None) on success, or (None, error_markdown) on failure.
    """
    try:
        from google import genai
    except ImportError:
        return None, (
            "⚠️ **Paket Eksik**\n\n"
            "`google-genai` yüklü değil. Şunu çalıştırın: "
            "`pip install google-genai`"
        )

    key = get_gemini_api_key(api_key)
    if not key:
        # Surface in Streamlit UI when available
        try:
            import streamlit as st

            st.error(
                "GEMINI_API_KEY bulunamadı. Lütfen Streamlit Secrets veya "
                ".env dosyanızı kontrol edin."
            )
        except Exception:
            pass
        return None, (
            "⚠️ **API Anahtarı Eksik**\n\n"
            "`GEMINI_API_KEY` bulunamadı. Lütfen **Streamlit Secrets** veya "
            "`.env` dosyanızı kontrol edin.\n\n"
            "Yerel için `.env`:\n"
            "```\nGEMINI_API_KEY=AIza...\n```\n\n"
            "Anahtarı [Google AI Studio](https://aistudio.google.com/apikey) "
            "üzerinden ücretsiz oluşturun (OAuth / Cloud access token değil)."
        )

    # Ensure the Developer API path is used (API key), not Vertex ADC/OAuth.
    os.environ["GOOGLE_API_KEY"] = key
    # Avoid accidental Vertex routing when only an AI Studio key is present
    os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

    client = genai.Client(api_key=key)
    return client, None


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


def _secret_or_env_model() -> Optional[str]:
    """Optional model override from Streamlit secrets or env."""
    try:
        import streamlit as st

        if "GEMINI_MODEL" in st.secrets:
            return _normalize_model_id(str(st.secrets["GEMINI_MODEL"]))
    except Exception:
        pass
    return _normalize_model_id(os.getenv("GEMINI_MODEL") or "") or None


def _candidate_models(explicit: Optional[str] = None) -> list[str]:
    """Build ordered unique model candidates from env/secrets + defaults."""
    ordered: list[str] = []
    for raw in (explicit, _secret_or_env_model(), *DEFAULT_MODEL_CANDIDATES):
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


def _is_auth_error(exc: Exception) -> bool:
    """True for 401 / unauthenticated / bad API key style failures."""
    text = str(exc).lower()
    return (
        "401" in text
        or "unauthenticated" in text
        or "access_token_type_unsupported" in text
        or "invalid authentication" in text
        or "api key not valid" in text
        or "permission_denied" in text
    )


def _auth_error_message(exc: Exception, key_hint: Optional[str]) -> str:
    """User-facing guidance for 401 / bad credential errors."""
    preview = ""
    if key_hint:
        # Never print the full key — only a safe fingerprint
        preview = (
            f"\n\nAlgılanan anahtar özeti: `{key_hint[:4]}…{key_hint[-4:]}` "
            f"(uzunluk: {len(key_hint)})"
        )
        if not key_hint.startswith("AIza"):
            preview += (
                "\n\n⚠️ Bu anahtar tipik bir **Google AI Studio** anahtarı gibi "
                "görünmüyor (genelde `AIza` ile başlar). OAuth / Cloud token "
                "kullanmayın."
            )

    return (
        "⚠️ **Kimlik Doğrulama Hatası (401)**\n\n"
        "Gemini API anahtarınız geçersiz veya yanlış türde.\n\n"
        f"```\n{exc}\n```"
        f"{preview}\n\n"
        "**Ne yapmalısınız?**\n"
        "1. https://aistudio.google.com/apikey adresinden **yeni bir API key** oluşturun\n"
        "2. `.env` dosyanızı şöyle yazın (tırnak işareti olmadan):\n"
        "```\nGEMINI_API_KEY=AIzaSy...\nGEMINI_MODEL=gemini-2.5-flash-lite\n```\n"
        "3. Streamlit Cloud kullanıyorsanız: App settings → Secrets içine aynı anahtarı ekleyin\n"
        "4. Uygulamayı tamamen durdurup (`Ctrl+C`) yeniden başlatın\n"
    )


def analyze_daily_performance(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Send today's trades + technical context to Gemini and return Markdown text.

    Uses a single Client from get_gemini_client() (Streamlit Secrets → .env).
    """
    if (trades_df is None or trades_df.empty) and not market_data_dict:
        return (
            "📭 **Analiz için veri yok**\n\n"
            "Bugün henüz işlem kaydı bulunmuyor. Sol menüden bir işlem ekleyip "
            "tekrar deneyin."
        )

    try:
        from google.genai import types
    except ImportError:
        return (
            "⚠️ **Paket Eksik**\n\n"
            "`google-genai` yüklü değil. Şunu çalıştırın: "
            "`pip install google-genai`"
        )

    client, client_error = get_gemini_client(api_key)
    if client is None:
        return client_error or "⚠️ Gemini istemcisi oluşturulamadı."

    resolved_key = get_gemini_api_key(api_key)
    user_prompt = _build_user_prompt(trades_df, market_data_dict)
    candidates = _candidate_models(model)

    try:
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
                if _is_auth_error(exc):
                    return _auth_error_message(exc, resolved_key)
                if _is_model_unavailable(exc):
                    continue
                break

        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            "Gemini API çağrısı başarısız oldu.\n\n"
            f"**Denenen modeller:** `{', '.join(tried)}`\n\n"
            f"```\n{last_error}\n```\n\n"
            "`.env` / Secrets içine çalışan bir model yazabilirsiniz, örn:\n"
            "`GEMINI_MODEL=gemini-2.5-flash-lite`\n"
        )

    except Exception as exc:  # noqa: BLE001
        if _is_auth_error(exc):
            return _auth_error_message(exc, resolved_key)
        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            f"Gemini API çağrısı başarısız oldu:\n\n```\n{exc}\n```\n\n"
            "İnternet bağlantınızı ve API anahtarınızı kontrol edin."
        )
