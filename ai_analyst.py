"""
AI analyst module powered by Google Gemini (google-genai SDK).

Analyzes either a single day's trades or the current cumulative portfolio
against live technical market context, returning a Turkish Markdown report.

Dual-environment API key resolution (works on localhost AND Streamlit Cloud):
  1. Explicit function argument
  2. st.secrets["GEMINI_API_KEY"] (Streamlit Cloud / .streamlit/secrets.toml)
  3. Environment variables via .env: GEMINI_API_KEY (or GOOGLE_API_KEY)

Model strategy:
  Prefer free-tier Flash models that still have quota for new API keys
  (gemini-2.5-flash-lite first). Retired / zero-quota models (404 / 429
  with limit:0) are skipped automatically so analysis keeps working.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Free-tier friendly models FIRST. gemini-2.0-flash / 1.5-flash often return
# 429 with limit:0 for new Google AI Studio keys — keep them last as legacy.
# Override anytime with GEMINI_MODEL in Streamlit Secrets or .env
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# System instructions stay in English; the model MUST reply in Turkish.
SYSTEM_INSTRUCTION = """
You are an experienced BIST (Borsa Istanbul) trading coach and technical analyst.
You review either a single day's trades or a cumulative portfolio with
constructive honesty, incorporating recent company/sector news when provided.

Evaluation rules (always follow):
1. Trading discipline: Did the trader overtrade, chase price, or ignore risk size?
2. Entry quality: For BUY trades / open positions, check whether price was near
   support, near the 60-day low, below EMA-20/EMA-50, or whether RSI(14) was
   oversold (<30) or overbought (>70). Praise patient entries; warn about FOMO
   buys into strength.
3. Concentration risk: Highlight if too much capital or too many trades cluster
   in one ticker or a single sector theme.
4. News & sector context: Use provided headlines/summaries to flag positive and
   negative developments that could move the stock near-term.
5. Tomorrow levels: For each discussed ticker (or the most important one), give
   exactly 1 key Support and 1 key Resistance level, derived from the provided
   high_60d / low_60d / EMA / current price context.
6. Expected-change score: Assign ONE overall score from -10 to +10 for the
   near-term outlook of the analyzed scope (portfolio or day), combining
   technicals + news. Be honest; do not inflate scores.
7. Strategy: End with one short, practical strategy suggestion.
8. Tone: Encouraging, clear, and practical — never hype. Be a calm mentor.

Language & format constraints (CRITICAL):
- The FINAL OUTPUT MUST be entirely in simple, clear, encouraging TURKISH
  EXCEPT for the mandatory machine-readable score line (see below).
- Use clean Markdown: ## headers, bullet points, and short paragraphs.
- ALWAYS include these sections (in addition to the review structure):
  ## Sektörel & Şirket Haberleri Özeti
     - Bullet key positive and negative developments from the news feed.
  ## Beklenen Değişim Puanı
     - First line MUST be exactly: SCORE: <integer from -10 to +10>
       Example: SCORE: +7
     - Second line: human label like `+7/10 - Yükseliş Beklentisi Kuvvetli`
       or `-4/10 - Kısa Vadeli Düzeltme Baskısı`
     - Then 2–3 short bullet reasons supporting the score.
- Suggested structure for a daily review:
  ## Günlük Özet
  ## Disiplin Değerlendirmesi
  ## İşlem Bazlı Teknik Yorum
  ## Sektörel & Şirket Haberleri Özeti
  ## Konsantrasyon / Risk Notu
  ## Yarın İçin Destek & Direnç
  ## Beklenen Değişim Puanı
  ## Strateji Önerisi
- Suggested structure for a portfolio review:
  ## Portföy Özeti
  ## Pozisyon Bazlı Teknik Yorum
  ## Sektörel & Şirket Haberleri Özeti
  ## Konsantrasyon / Risk Notu
  ## Destek & Direnç Seviyeleri
  ## Beklenen Değişim Puanı
  ## Strateji Önerisi
"""


# ---------------------------------------------------------------------------
# Score parsing (SCORE: +7 emitted by Gemini)
# ---------------------------------------------------------------------------
def parse_expected_score(report_text: str) -> dict[str, Any]:
    """
    Extract the -10..+10 expected-change score from a Gemini report.

    Returns:
        {score: int|None, label: str, reasons: list[str], success: bool}
    """
    result: dict[str, Any] = {
        "score": None,
        "label": "",
        "reasons": [],
        "success": False,
    }
    if not report_text:
        return result

    text = report_text.strip()

    # Primary: SCORE: +7  / SCORE: -4  / SCORE: 3
    m = re.search(r"(?im)^\s*SCORE\s*:\s*([+-]?\d{1,2})\s*$", text)
    if not m:
        # Fallback patterns inside the score section
        m = re.search(
            r"(?i)(?:SCORE|puan)\s*[:=]\s*([+-]?\d{1,2})\s*(?:/10)?",
            text,
        )
    if not m:
        m = re.search(r"(?m)^\s*([+-]?\d{1,2})\s*/\s*10\b", text)

    if m:
        try:
            score = int(m.group(1))
            score = max(-10, min(10, score))
            result["score"] = score
            result["success"] = True
        except ValueError:
            pass

    # Human label near the score (e.g. "+7/10 - Yükseliş Beklentisi Kuvvetli")
    label_m = re.search(
        r"(?m)^\s*([+-]?\d{1,2}\s*/\s*10\s*[-–—:].+)$",
        text,
    )
    if label_m:
        result["label"] = label_m.group(1).strip()
    elif result["score"] is not None:
        s = result["score"]
        if s >= 6:
            tone = "Yükseliş Beklentisi Kuvvetli"
        elif s >= 2:
            tone = "Hafif Pozitif Görünüm"
        elif s >= -1:
            tone = "Nötr / Yatay Beklenti"
        elif s >= -5:
            tone = "Kısa Vadeli Düzeltme Baskısı"
        else:
            tone = "Güçlü Negatif Baskı"
        sign = f"+{s}" if s > 0 else str(s)
        result["label"] = f"{sign}/10 - {tone}"

    # Pull a few bullets from the score section as reasons
    score_section = re.search(
        r"(?is)##\s*Beklenen Değişim Puanı\s*(.*?)(?=\n##\s|\Z)",
        text,
    )
    if score_section:
        bullets = re.findall(r"(?m)^\s*[-*•]\s+(.+)$", score_section.group(1))
        result["reasons"] = [b.strip() for b in bullets[:3] if b.strip()]

    return result


def score_badge_meta(score: Optional[int]) -> dict[str, str]:
    """CSS tone helpers for the UI score card."""
    if score is None:
        return {"tone": "flat", "emoji": "➖", "title": "Puan Yok"}
    if score >= 6:
        return {"tone": "pos", "emoji": "🚀", "title": "Güçlü Pozitif"}
    if score >= 2:
        return {"tone": "pos", "emoji": "📈", "title": "Pozitif"}
    if score >= -1:
        return {"tone": "flat", "emoji": "➖", "title": "Nötr"}
    if score >= -5:
        return {"tone": "neg", "emoji": "📉", "title": "Negatif"}
    return {"tone": "neg", "emoji": "⚠️", "title": "Güçlü Negatif"}


# ---------------------------------------------------------------------------
# API key + client resolution (localhost .env AND Streamlit Cloud secrets)
# ---------------------------------------------------------------------------
def _clean_api_key(raw: Optional[str]) -> Optional[str]:
    """Strip whitespace / wrapping quotes; reject placeholder values."""
    if raw is None:
        return None
    key = str(raw).strip()
    if (key.startswith('"') and key.endswith('"')) or (
        key.startswith("'") and key.endswith("'")
    ):
        key = key[1:-1].strip()
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
    Resolve the Gemini API key: argument -> st.secrets -> environment (.env).
    """
    cleaned = _clean_api_key(explicit)
    if cleaned:
        return cleaned

    # Streamlit Cloud secrets / local .streamlit/secrets.toml
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            cleaned = _clean_api_key(st.secrets["GEMINI_API_KEY"])
            if cleaned:
                return cleaned
        if "GOOGLE_API_KEY" in st.secrets:
            cleaned = _clean_api_key(st.secrets["GOOGLE_API_KEY"])
            if cleaned:
                return cleaned
    except Exception:
        # st.secrets raises when no secrets file exists locally — that's fine
        pass

    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        cleaned = _clean_api_key(os.getenv(env_name))
        if cleaned:
            return cleaned

    return None


def get_gemini_client(api_key: Optional[str] = None):
    """
    Build a google-genai Client using the resolved API key.

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
        try:
            import streamlit as st

            st.error(
                "GEMINI_API_KEY bulunamadı. Lütfen .env dosyanızı veya "
                "Streamlit Secrets ayarlarınızı kontrol edin."
            )
        except Exception:
            pass
        return None, (
            "⚠️ **API Anahtarı Eksik**\n\n"
            "`GEMINI_API_KEY` bulunamadı. Lütfen **.env** dosyanızı veya "
            "**Streamlit Secrets** ayarlarınızı kontrol edin.\n\n"
            "Yerel kullanım için `.env`:\n"
            "```\nGEMINI_API_KEY=AIza...\n```\n\n"
            "Streamlit Cloud için: App settings → Secrets →\n"
            "```\nGEMINI_API_KEY = \"AIza...\"\n```\n\n"
            "Anahtarı [Google AI Studio](https://aistudio.google.com/apikey) "
            "üzerinden ücretsiz oluşturun."
        )

    # Force the Gemini Developer API path (API key), never Vertex OAuth/ADC —
    # prevents 401 ACCESS_TOKEN_TYPE_UNSUPPORTED errors.
    os.environ["GOOGLE_API_KEY"] = key
    os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

    client = genai.Client(api_key=key)
    return client, None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _format_trades(trades_df: pd.DataFrame) -> str:
    """Serialize trades into a compact text block for the prompt."""
    if trades_df is None or trades_df.empty:
        return "No trades logged for this date."

    lines = []
    for _, row in trades_df.iterrows():
        notes = row.get("notes") or ""
        note_part = f" | notes: {notes}" if notes else ""
        lines.append(
            f"- {row['action']} {row['quantity']} x {row['ticker']} "
            f"@ {row['price']} TRY on {row['date']}{note_part}"
        )
    return "\n".join(lines)


def _format_holdings(holdings_df: pd.DataFrame) -> str:
    """Serialize current holdings into a compact text block for the prompt."""
    if holdings_df is None or holdings_df.empty:
        return "No open positions."

    lines = []
    for _, row in holdings_df.iterrows():
        lines.append(
            f"- {row['ticker']}: {row['quantity']} shares, "
            f"avg buy price {row['avg_buy_price']} TRY, "
            f"total cost {row['cost_basis']} TRY"
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


def _format_news(news_dict: Optional[dict[str, dict[str, Any]]]) -> str:
    """Serialize recent headlines for Gemini (sector/company context)."""
    if not news_dict:
        return "No recent news available."

    blocks = []
    for ticker, payload in news_dict.items():
        items = (payload or {}).get("news") or []
        if not items:
            err = (payload or {}).get("error") or "no headlines"
            blocks.append(f"[{ticker}] NEWS: {err}")
            continue
        lines = [f"[{ticker}] recent news:"]
        for i, n in enumerate(items, 1):
            title = n.get("title") or ""
            publisher = n.get("publisher") or ""
            summary = n.get("summary") or ""
            published = n.get("published") or ""
            line = f"  {i}. {title}"
            if publisher:
                line += f" ({publisher})"
            if published:
                line += f" [{published}]"
            lines.append(line)
            if summary:
                lines.append(f"     summary: {summary[:280]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No recent news available."


def _build_daily_prompt(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    analysis_date: Optional[str] = None,
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """User prompt for a specific date's trades."""
    date_part = f" (date: {analysis_date})" if analysis_date else ""
    return (
        f"Analyze this BIST trading session{date_part} for a retail portfolio.\n\n"
        "=== TRADES ===\n"
        f"{_format_trades(trades_df)}\n\n"
        "=== MARKET / TECHNICAL SNAPSHOT ===\n"
        f"{_format_market_data(market_data_dict)}\n\n"
        "=== RECENT COMPANY / SECTOR NEWS ===\n"
        f"{_format_news(news_dict)}\n\n"
        "Produce the Turkish Markdown daily coaching report now "
        "(use the daily review structure, including news summary and SCORE line)."
    )


def _build_portfolio_prompt(
    holdings_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """User prompt for the whole active portfolio."""
    return (
        "Analyze this retail investor's CURRENT ACTIVE BIST PORTFOLIO "
        "(cumulative open positions).\n\n"
        "=== OPEN POSITIONS ===\n"
        f"{_format_holdings(holdings_df)}\n\n"
        "=== MARKET / TECHNICAL SNAPSHOT ===\n"
        f"{_format_market_data(market_data_dict)}\n\n"
        "=== RECENT COMPANY / SECTOR NEWS ===\n"
        f"{_format_news(news_dict)}\n\n"
        "For each position, compare current price vs average buy price "
        "(unrealized P/L direction), evaluate technical posture and news, then "
        "produce the Turkish Markdown portfolio report "
        "(include news summary and SCORE line)."
    )


# ---------------------------------------------------------------------------
# Model selection + error classification
# ---------------------------------------------------------------------------
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
    """Build ordered unique model candidates from overrides + defaults."""
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


def _is_quota_error(exc: Exception) -> bool:
    """True for 429 / RESOURCE_EXHAUSTED (per-model free-tier quota)."""
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "rate_limit" in text
    )


def _quota_retry_seconds(exc: Exception) -> float:
    """Parse RetryInfo delay when present; else 0."""
    text = str(exc)
    # e.g. "Please retry in 13.64895555s." or "'retryDelay': '13s'"
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, re.I)
    if m:
        return min(float(m.group(1)), 20.0)
    m = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s?", text, re.I)
    if m:
        return min(float(m.group(1)), 20.0)
    return 0.0


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
        "```\nGEMINI_API_KEY=AIzaSy...\n```\n"
        "3. Streamlit Cloud kullanıyorsanız: App settings → Secrets içine aynı anahtarı ekleyin\n"
        "4. Uygulamayı tamamen durdurup (`Ctrl+C`) yeniden başlatın\n"
    )


# ---------------------------------------------------------------------------
# Core generation with model fallback
# ---------------------------------------------------------------------------
def _run_gemini(
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Send a prompt to Gemini with model fallback; return Markdown or error."""
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
    candidates = _candidate_models(model)

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.6,
            max_output_tokens=4096,
        )

        last_error: Optional[Exception] = None
        tried: list[str] = []
        saw_quota = False

        def _try_models(model_ids: list[str]) -> Optional[str]:
            nonlocal last_error, saw_quota
            for model_id in model_ids:
                if model_id in tried:
                    continue
                tried.append(model_id)
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=user_prompt,
                        config=config,
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if _is_auth_error(exc):
                        return _auth_error_message(exc, resolved_key)
                    if _is_model_unavailable(exc) or _is_quota_error(exc):
                        # 404 retired OR 429 limit:0 / per-model quota → next model
                        saw_quota = saw_quota or _is_quota_error(exc)
                        # Short pause only when API asks for a retry delay
                        delay = _quota_retry_seconds(exc)
                        if 0 < delay <= 3:
                            time.sleep(delay)
                        continue
                    # Unknown network / server error — stop
                    return None
            return None

        # Pass 1: preferred free-tier candidates
        result = _try_models(candidates)
        if isinstance(result, str):
            return result

        # Pass 2: any other flash models the API still lists
        discovered = [
            m for m in _discover_flash_models(client) if m not in tried
        ]
        result = _try_models(discovered)
        if isinstance(result, str):
            return result

        if saw_quota:
            return (
                "⚠️ **Kota Aşıldı (429)**\n\n"
                "Bu API anahtarının ücretsiz kotası şu an dolu veya seçilen "
                "modeller için `limit: 0` (ücretsiz erişim yok).\n\n"
                f"**Denenen modeller:** `{', '.join(tried)}`\n\n"
                f"```\n{last_error}\n```\n\n"
                "**Ne yapın?**\n"
                "1. `.env` ve Streamlit Secrets içine şunu ekleyin:\n"
                "```\nGEMINI_MODEL=gemini-2.5-flash-lite\n```\n"
                "2. 1–2 dakika bekleyip tekrar deneyin\n"
                "3. https://aistudio.google.com/apikey üzerinden yeni bir key "
                "oluşturun veya https://ai.dev/rate-limit adresinden kotayı kontrol edin\n"
            )

        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            "Gemini API çağrısı başarısız oldu.\n\n"
            f"**Denenen modeller:** `{', '.join(tried)}`\n\n"
            f"```\n{last_error}\n```\n\n"
            "`.env` / Secrets içine çalışan bir model yazın:\n"
            "`GEMINI_MODEL=gemini-2.5-flash-lite`\n"
        )

    except Exception as exc:  # noqa: BLE001
        if _is_auth_error(exc):
            return _auth_error_message(exc, resolved_key)
        if _is_quota_error(exc):
            return (
                "⚠️ **Kota Aşıldı (429)**\n\n"
                f"```\n{exc}\n```\n\n"
                "1–2 dakika bekleyip tekrar deneyin veya "
                "`GEMINI_MODEL=gemini-2.5-flash-lite` kullanın."
            )
        return (
            "⚠️ **AI Analiz Hatası**\n\n"
            f"Gemini API çağrısı başarısız oldu:\n\n```\n{exc}\n```\n\n"
            "İnternet bağlantınızı ve API anahtarınızı kontrol edin."
        )


# ---------------------------------------------------------------------------
# Public analysis entry points
# ---------------------------------------------------------------------------
def analyze_daily_performance(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    analysis_date: Optional[str] = None,
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """
    Analyze trades from one specific date (default flow: today).

    Returns a Turkish Markdown report, or a Markdown error message.
    """
    if (trades_df is None or trades_df.empty) and not market_data_dict:
        return (
            "📭 **Analiz için veri yok**\n\n"
            "Seçilen tarihte işlem kaydı bulunmuyor. Farklı bir tarih seçin "
            "veya önce bir işlem ekleyin."
        )

    prompt = _build_daily_prompt(
        trades_df, market_data_dict, analysis_date, news_dict=news_dict
    )
    return _run_gemini(prompt, api_key=api_key, model=model)


def analyze_portfolio(
    holdings_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """
    Analyze the current cumulative portfolio (all open positions).

    Returns a Turkish Markdown report, or a Markdown error message.
    """
    if holdings_df is None or holdings_df.empty:
        return (
            "📭 **Aktif pozisyon yok**\n\n"
            "Portföy analizi için en az bir açık pozisyon gerekli. "
            "Önce bir BUY işlemi ekleyin."
        )

    prompt = _build_portfolio_prompt(
        holdings_df, market_data_dict, news_dict=news_dict
    )
    return _run_gemini(prompt, api_key=api_key, model=model)
