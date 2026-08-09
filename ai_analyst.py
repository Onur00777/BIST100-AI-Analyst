"""
AI analyst module powered by Google Gemini (google-genai SDK).

Strict Turkish BIST lead-analyst engine:
- Absolute coverage: EVERY ticker in the portfolio/day appears in
  Destek & Direnç table, hisse-bazlı verdicts, and news/sector notes.
- Direct verdict language with mandatory tags:
  [ÇOK İYİ HAMLE] / [RİSKLİ / NÖTR] / [HATALI / TEHLİKELİ]
- English headlines must be translated to Turkish; missing news → sector outlook.

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

# System instructions: strict Turkish BIST lead analyst — full coverage, direct verdicts.
SYSTEM_INSTRUCTION = """
Sen Borsa İstanbul (BIST) konusunda uzman, sözünü sakınmayan sert ve rasyonel bir Baş Analistsin.

MUTLAK KURALLAR (İHLAL YASAK):
1. Sana iletilen portföydeki / işlem listesindeki TÜM hisseleri İSTİSNASIZ analiz et.
   Tek bir hisseyi bile atlamak, özet geçmek veya "diğerleri benzer" demek YASAKTIR.
2. Çıktının tamamı sade, net, doğrudan TÜRKÇE olmalıdır (SCORE satırı hariç).
3. İngilizce haber başlıklarını / özetlerini ANINDA Türkçe'ye çevirip sun.
4. Haberi olmayan her hisse için o hissenin sektörüne kısa bir sektör görünümü yaz
   (ör. "Enerji sektörü genel görünümü..."). Her hisse haber bölümünde yer almalı.
5. Belirsiz / pasif dil YASAK: "düşünülebilir", "bakılabilir", "değerlendirilebilir",
   "izlenebilir" gibi kaçamak ifadeler kullanma.
6. Direkt dil ZORUNLU. Örnekler:
   - "Alman/Eklemen harika bir hamle olmuş"
   - "Bu fiyattan alman riskli bir hata olmuş"
   - "Derhal stop-loss düşünülmeli"
   - "Kâr al bölgesi"
7. Her hisse için şu etiketlerden BİRİNİ açıkça ver:
   [ÇOK İYİ HAMLE]  veya  [RİSKLİ / NÖTR]  veya  [HATALI / TEHLİKELİ]
8. Destek & Direnç tablosunda portföydeki HER hisse için satır üret. Satır atlama yok.
9. Beklenen değişim puanı: -10 ile +10 arası TEK skor. Dürüst ol; şişirme.

ZORUNLU RAPOR İSKELETİ (Markdown, bu sırayla):

## 📊 Genel Portföy Kararı ve Değişim Puanı
- Net skor ve 3 maddelik özet.
- İlk satır mutlaka: SCORE: <integer -10..+10>
  Örnek: SCORE: +4
- İkinci satır: `+4/10 - ...` formatında etiket
- Ardından 2–3 kısa gerekçe maddesi

## 🎯 Destek & Direnç Tablosu
Portföydeki TÜM hisseler için Markdown tablosu (hiçbir hisse eksik kalmasın):

| Hisse | Anlık Fiyat | Destek 1 | Destek 2 | Direnç 1 | Direnç 2 | Aksiyon / Tavsiye |
| --- | --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | ... | ... |

Seviyeleri verilen current_price / EMA / high_60d / low_60d bağlamından üret.

## 🔍 Hisse Bazlı İsabet Değerlendirmesi
Her hisse için AYRI alt başlık (### THYAO gibi) ve:
- Etiket: [ÇOK İYİ HAMLE] / [RİSKLİ / NÖTR] / [HATALI / TEHLİKELİ]
- "Bu hisseyi bu fiyattan almak/tutmak iyi olmuş çünkü..." VEYA
  "Bu alım hatalı olmuş çünkü..." şeklinde direkt, babacan/sert eleştiri
- Kısa teknik gerekçe (RSI, EMA, maliyet vs fiyat)

## 📰 Sektörel & Şirket Haberleri
Portföydeki HER hisse için en az bir madde:
- Varsa şirket haberini Türkçe özetle
- Yoksa: "{Hisse} — {Sektör} sektörü genel görünümü: ..."

## 💡 Net Strateji ve Reçete
- Hangi hissenin ağırlığı azaltılmalı, hangisi tutulmalı, nereden stop olunmalı
  açık açık söyle. Yuvarlak laflar yok.
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
        r"(?is)##\s*.*?(?:Beklenen Değişim|Genel Portföy Kararı|Değişim Puanı).*?\n(.*?)(?=\n##\s|\Z)",
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


def list_required_tickers(
    market_data_dict: Optional[dict[str, dict[str, Any]]] = None,
    holdings_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Deduplicated ticker universe that MUST appear in the report."""
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        t = str(raw or "").strip().upper().replace(".IS", "")
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)

    if holdings_df is not None and not holdings_df.empty and "ticker" in holdings_df.columns:
        for t in holdings_df["ticker"].tolist():
            _add(t)
    if trades_df is not None and not trades_df.empty and "ticker" in trades_df.columns:
        for t in trades_df["ticker"].tolist():
            _add(t)
    if market_data_dict:
        for t in market_data_dict.keys():
            _add(t)
    return ordered


def coverage_checklist(tickers: list[str]) -> str:
    """Explicit checklist injected into the user prompt."""
    if not tickers:
        return "(Hisse listesi boş)"
    n = len(tickers)
    lines = [
        f"ZORUNLU KAPSAM: {n} hisse — HEPSİ tabloda, hisse bazlı kararda ve haberlerde olmalı:",
        ", ".join(tickers),
        "",
        "Her hisse için doğrula:",
    ]
    for t in tickers:
        lines.append(
            f"- [ ] {t}: Destek/Direnç satırı + [ÇOK İYİ HAMLE|RİSKLİ / NÖTR|HATALI / TEHLİKELİ] + haber/sektör notu"
        )
    return "\n".join(lines)


def report_missing_tickers(report_text: str, required: list[str]) -> list[str]:
    """Return tickers that do not appear anywhere in the generated report."""
    if not report_text or not required:
        return list(required or [])
    upper = report_text.upper()
    return [t for t in required if t.upper() not in upper]


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
    """
    Serialize recent headlines for Gemini.

    Every ticker in news_dict gets a block — empty feeds are marked so the
    model MUST write a Turkish sector outlook instead of skipping the name.
    """
    if not news_dict:
        return "No recent news available."

    blocks = []
    for ticker, payload in news_dict.items():
        sector = (payload or {}).get("sector") or "Genel Piyasa"
        items = (payload or {}).get("news") or []
        needs_fallback = bool((payload or {}).get("needs_sector_fallback", not items))

        if not items or needs_fallback:
            err = (payload or {}).get("error") or "doğrudan haber yok"
            blocks.append(
                f"[{ticker}] SECTOR={sector}\n"
                f"  STATUS: NO_DIRECT_NEWS ({err})\n"
                f"  INSTRUCTION: '{ticker}' için '{sector}' sektörünün "
                f"genel görünümünü 1–2 cümle Türkçe yaz. Atlama."
            )
            continue

        lines = [f"[{ticker}] SECTOR={sector} | recent news (translate ALL to Turkish):"]
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


def _user_report_contract(ticker_count: int, tickers: list[str]) -> str:
    """Hard user-facing contract pasted into every analysis prompt."""
    joined = ", ".join(tickers) if tickers else "(yok)"
    return f"""
=== ANALİZ SÖZLEŞMESİ (ZORUNLU) ===
Sen Borsa İstanbul (BIST) konusunda uzman, sözünü sakınmayan sert ve rasyonel bir Baş Analistsin.
Sana iletilen portföydeki {ticker_count} hissenin İSTİSNASIZ HEPSİNİ analiz edeceksin.
TEK BİR HİSSEYİ BİLE ATLAMAK VEYA ÖZET GEÇMEK YASAKTIR.

Hisse listesi ({ticker_count}): {joined}

Rapor Formatın Şu Şekilde Olmalıdır:

1. 📊 GENEL PORTFÖY KARARI VE DEĞİŞİM PUANI (-10 ile +10)
- Net Skor ve 3 maddelik özet.
- İlk satır: SCORE: <tamsayı>

2. 🎯 TÜM HİSSELER İÇİN DESTEK & DİRENÇ TABLOSU (TAM LİSTE)
| Hisse | Anlık Fiyat | Destek 1 | Destek 2 | Direnç 1 | Direnç 2 | Aksiyon / Tavsiye |
(Portföydeki tüm {ticker_count} hisse burada tablo halinde LİSTELENECEK).

3. 🔍 HİSSE BAZLI İSABET DEĞERLENDİRMESİ (DOĞRUDAN VE NET DİL)
Her hisse için tek tek:
- [ÇOK İYİ HAMLE] / [RİSKLİ / NÖTR] / [HATALI / TEHLİKELİ] etiketi ver.
- "Bu hisseyi bu fiyattan almak/tutmak iyi olmuş çünkü..." veya
  "Bu alım hatalı olmuş çünkü..." şeklinde direkt babacan/sert bir dille eleştir.
- "düşünülebilir / bakılabilir" gibi kaçamak dil KULLANMA.

4. 📰 TÜM ŞİRKETLER İÇİN SEKTÖREL VE ŞİRKET HABERLERİ (TAMAMI TÜRKÇE)
- İngilizce haberleri anında Türkçe'ye çevir.
- Haberi olmayan hisseler için sektörün genel durumunu yaz.
- {ticker_count} hissenin tamamı bu bölümde geçmeli.

5. 💡 NET STRATEJİ VE REÇETE
- Hangi hissenin ağırlığı azaltılmalı, hangisi tutulmalı, nereden stop olunmalı açık açık söyle.

{coverage_checklist(tickers)}
""".strip()


def _build_daily_prompt(
    trades_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    analysis_date: Optional[str] = None,
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """User prompt for a specific date's trades."""
    tickers = list_required_tickers(
        market_data_dict=market_data_dict, trades_df=trades_df
    )
    ticker_count = len(tickers)
    date_part = f" (tarih: {analysis_date})" if analysis_date else ""
    return (
        f"Bu BIST işlem gününü{date_part} analiz et.\n\n"
        f"{_user_report_contract(ticker_count, tickers)}\n\n"
        "=== İŞLEMLER ===\n"
        f"{_format_trades(trades_df)}\n\n"
        "=== TEKNİK GÖRÜNÜM ===\n"
        f"{_format_market_data(market_data_dict)}\n\n"
        "=== ŞİRKET / SEKTÖR HABERLERİ ===\n"
        f"{_format_news(news_dict)}\n\n"
        f"Şimdi {ticker_count} hissenin tamamını kapsayan Türkçe Markdown raporu yaz. "
        "Tablo satırı eksik bırakma."
    )


def _build_portfolio_prompt(
    holdings_df: pd.DataFrame,
    market_data_dict: dict[str, dict[str, Any]],
    news_dict: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """User prompt for the whole active portfolio."""
    tickers = list_required_tickers(
        market_data_dict=market_data_dict, holdings_df=holdings_df
    )
    ticker_count = len(tickers)
    return (
        "Bu yatırımcının GÜNCEL AKTİF BIST PORTFÖYÜNÜ (açık pozisyonlar) analiz et.\n\n"
        f"{_user_report_contract(ticker_count, tickers)}\n\n"
        "=== AÇIK POZİSYONLAR ===\n"
        f"{_format_holdings(holdings_df)}\n\n"
        "=== TEKNİK GÖRÜNÜM ===\n"
        f"{_format_market_data(market_data_dict)}\n\n"
        "=== ŞİRKET / SEKTÖR HABERLERİ ===\n"
        f"{_format_news(news_dict)}\n\n"
        "Her pozisyon için maliyet vs güncel fiyat (gerçekleşmemiş K/Z yönü) değerlendir. "
        f"Şimdi {ticker_count} hissenin tamamını kapsayan Türkçe Markdown portföy raporu yaz. "
        "Tablo satırı eksik bırakma."
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
            temperature=0.45,
            # Large portfolios (10–20 tickers) need room for full tables + verdicts
            max_output_tokens=8192,
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

    # Guarantee every analyzed ticker has a news/sector slot
    required = list_required_tickers(
        market_data_dict=market_data_dict, trades_df=trades_df
    )
    news_dict = _ensure_news_coverage(required, news_dict)

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

    required = list_required_tickers(
        market_data_dict=market_data_dict, holdings_df=holdings_df
    )
    news_dict = _ensure_news_coverage(required, news_dict)

    prompt = _build_portfolio_prompt(
        holdings_df, market_data_dict, news_dict=news_dict
    )
    return _run_gemini(prompt, api_key=api_key, model=model)


def _ensure_news_coverage(
    tickers: list[str],
    news_dict: Optional[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    Ensure every required ticker has a news payload (sector fallback if missing).
    """
    out: dict[str, dict[str, Any]] = dict(news_dict or {})
    try:
        import market_data as md
    except ImportError:
        md = None  # type: ignore

    for t in tickers:
        if t in out and isinstance(out[t], dict):
            # Backfill sector if older payloads omit it
            if not out[t].get("sector") and md is not None:
                out[t]["sector"] = md.get_ticker_sector(t)
            if "needs_sector_fallback" not in out[t]:
                out[t]["needs_sector_fallback"] = not bool(out[t].get("news"))
            continue
        if md is not None:
            out[t] = {
                "ticker": t,
                "yahoo_ticker": f"{t}.IS",
                "sector": md.get_ticker_sector(t),
                "news": [],
                "success": False,
                "error": "Haber verisi eksik — sektör görünümü yazılmalı.",
                "needs_sector_fallback": True,
            }
        else:
            out[t] = {
                "ticker": t,
                "sector": "Genel Piyasa",
                "news": [],
                "success": False,
                "error": "Haber verisi eksik — sektör görünümü yazılmalı.",
                "needs_sector_fallback": True,
            }
    return out
