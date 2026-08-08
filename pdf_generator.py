"""
PDF report generator for BIST 100 Daily AI Analyst.

Uses reportlab with a Unicode TTF (DejaVu / Arial / Liberation) so Turkish
characters (ğüşıöç ĞÜŞİÖÇ) render correctly on Windows and Streamlit Cloud.
"""

from __future__ import annotations

import io
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Registered once per process
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONTS_READY = False


def _candidate_font_paths() -> list[tuple[str, str]]:
    """
    Return [(regular_ttf, bold_ttf), ...] in preference order.
    Covers Streamlit Cloud (Debian DejaVu), Windows Arial, macOS, etc.
    """
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return [
        # Linux / Streamlit Cloud
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ),
        # Windows
        (
            str(Path(windir) / "Fonts" / "arial.ttf"),
            str(Path(windir) / "Fonts" / "arialbd.ttf"),
        ),
        (
            str(Path(windir) / "Fonts" / "segoeui.ttf"),
            str(Path(windir) / "Fonts" / "segoeuib.ttf"),
        ),
        # macOS
        (
            "/Library/Fonts/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ),
    ]


def _ensure_fonts() -> tuple[str, str]:
    """Register a Unicode TTF pair; fall back to Helvetica if none found."""
    global _FONT_REGULAR, _FONT_BOLD, _FONTS_READY
    if _FONTS_READY:
        return _FONT_REGULAR, _FONT_BOLD

    for regular, bold in _candidate_font_paths():
        if not Path(regular).is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("BISTSans", regular))
            bold_path = bold if Path(bold).is_file() else regular
            pdfmetrics.registerFont(TTFont("BISTSans-Bold", bold_path))
            _FONT_REGULAR = "BISTSans"
            _FONT_BOLD = "BISTSans-Bold"
            _FONTS_READY = True
            return _FONT_REGULAR, _FONT_BOLD
        except Exception:  # noqa: BLE001
            continue

    _FONTS_READY = True
    return _FONT_REGULAR, _FONT_BOLD


def _esc(text: Any) -> str:
    """Escape text for reportlab Paragraph (XML entities)."""
    if text is None:
        return ""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _md_to_flowables(md_text: str, styles: dict) -> list:
    """Very small Markdown → Paragraph converter for the AI report body."""
    flow: list = []
    if not md_text:
        return flow

    for raw in md_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flow.append(Spacer(1, 3))
            continue

        # Skip machine SCORE line in body (shown separately)
        if re.match(r"(?i)^\s*SCORE\s*:", line):
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line.strip())
        if heading:
            level = len(heading.group(1))
            style = styles["h2"] if level <= 2 else styles["h3"]
            flow.append(Paragraph(_esc(heading.group(2)), style))
            continue

        bullet = re.match(r"^[-*•]\s+(.*)$", line.strip())
        if bullet:
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(bullet.group(1)))
            flow.append(Paragraph(f"• {body}", styles["body"]))
            continue

        body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(line.strip()))
        # Strip leftover backticks
        body = body.replace("`", "")
        flow.append(Paragraph(body, styles["body"]))

    return flow


def _build_styles(font_r: str, font_b: str) -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "BISTTitle",
            parent=base["Title"],
            fontName=font_b,
            fontSize=16,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "BISTSub",
            parent=base["Normal"],
            fontName=font_r,
            fontSize=9,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "BISTH2",
            parent=base["Heading2"],
            fontName=font_b,
            fontSize=11,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "BISTH3",
            parent=base["Heading3"],
            fontName=font_b,
            fontSize=10,
            textColor=colors.HexColor("#334155"),
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BISTBody",
            parent=base["Normal"],
            fontName=font_r,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "score": ParagraphStyle(
            "BISTScore",
            parent=base["Normal"],
            fontName=font_b,
            fontSize=13,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "muted": ParagraphStyle(
            "BISTMuted",
            parent=base["Normal"],
            fontName=font_r,
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
        ),
    }


def generate_pdf_report(
    ticker: str,
    report_date: str,
    metrics: Optional[dict[str, Any]] = None,
    ai_report_text: str = "",
    trades: Optional[list[dict[str, Any]]] = None,
    news: Optional[list[dict[str, Any]]] = None,
    score_info: Optional[dict[str, Any]] = None,
    scope_label: str = "",
) -> bytes:
    """
    Build a downloadable PDF report and return raw bytes.

    Args:
        ticker: Primary ticker (or comma-joined list / 'PORTFOY').
        report_date: ISO date string used in the header / filename.
        metrics: Technical snapshot dict (or multi-ticker map values flattened).
        ai_report_text: Full Gemini Markdown report.
        trades: Optional list of trade dicts for a detail table.
        news: Optional list of news dicts {title, publisher, summary}.
        score_info: Output of ai_analyst.parse_expected_score(...).
        scope_label: Human scope text shown under the title.
    """
    font_r, font_b = _ensure_fonts()
    styles = _build_styles(font_r, font_b)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"BIST 100 AI Analyst — {ticker}",
        author="BIST 100 Daily AI Analyst",
    )

    story: list = []
    story.append(Paragraph("BIST 100 AI Analyst Report", styles["title"]))
    story.append(
        Paragraph(
            _esc(
                f"{ticker}  ·  {report_date or date.today().isoformat()}"
                + (f"  ·  {scope_label}" if scope_label else "")
            ),
            styles["subtitle"],
        )
    )

    # --- Score banner ---
    score_info = score_info or {}
    score = score_info.get("score")
    label = score_info.get("label") or ""
    if score is not None:
        sign = f"+{score}" if score > 0 else str(score)
        story.append(
            Paragraph(
                _esc(f"Beklenen Değişim Puanı: {sign}/10"),
                styles["score"],
            )
        )
        if label:
            story.append(Paragraph(_esc(label), styles["muted"]))
        reasons = score_info.get("reasons") or []
        for r in reasons[:3]:
            story.append(Paragraph(f"• {_esc(r)}", styles["body"]))
        story.append(Spacer(1, 6))

    # --- Technical metrics table ---
    metrics = metrics or {}
    # Support either a single summary dict or {ticker: summary}
    metric_rows = [["Hisse", "Fiyat", "Günlük %", "RSI", "EMA20", "EMA50"]]

    def _add_metric_row(tkr: str, m: dict) -> None:
        if not m:
            return
        metric_rows.append(
            [
                str(tkr),
                _fmt(m.get("current_price")),
                _fmt(m.get("daily_change_pct"), pct=True),
                _fmt(m.get("rsi_14")),
                _fmt(m.get("ema_20")),
                _fmt(m.get("ema_50")),
            ]
        )

    if metrics.get("ticker") or metrics.get("current_price") is not None:
        _add_metric_row(metrics.get("ticker") or ticker, metrics)
    else:
        for tkr, m in metrics.items():
            if isinstance(m, dict):
                _add_metric_row(tkr, m)

    if len(metric_rows) > 1:
        story.append(Paragraph("Teknik Göstergeler", styles["h2"]))
        table = Table(metric_rows, colWidths=[70, 70, 70, 55, 70, 70])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e222d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), font_b),
                    ("FONTNAME", (0, 1), (-1, -1), font_r),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f1f5f9")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --- Trades table ---
    if trades:
        story.append(Paragraph("İşlem Detayları", styles["h2"]))
        trade_rows = [["Tarih", "Hisse", "İşlem", "Adet", "Fiyat", "Not"]]
        for t in trades:
            trade_rows.append(
                [
                    str(t.get("date", "")),
                    str(t.get("ticker", "")),
                    str(t.get("action", "")),
                    _fmt(t.get("quantity")),
                    _fmt(t.get("price")),
                    str(t.get("notes", "") or "")[:40],
                ]
            )
        ttable = Table(trade_rows, colWidths=[65, 55, 45, 50, 60, 120])
        ttable.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2962ff")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), font_b),
                    ("FONTNAME", (0, 1), (-1, -1), font_r),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#eff6ff")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(ttable)
        story.append(Spacer(1, 8))

    # --- News ---
    if news:
        story.append(Paragraph("Sektörel & Şirket Haberleri", styles["h2"]))
        for n in news[:8]:
            title = n.get("title") or ""
            publisher = n.get("publisher") or ""
            summary = n.get("summary") or ""
            head = title if not publisher else f"{title} ({publisher})"
            story.append(Paragraph(f"• <b>{_esc(head)}</b>", styles["body"]))
            if summary:
                story.append(Paragraph(_esc(summary[:280]), styles["body"]))
        story.append(Spacer(1, 6))

    # --- AI analysis body ---
    story.append(Paragraph("Gemini AI Analiz Raporu", styles["h2"]))
    story.extend(_md_to_flowables(ai_report_text or "", styles))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Bu rapor bilgilendirme amaçlıdır; yatırım tavsiyesi değildir.",
            styles["muted"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None or value == "":
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pct:
        sign = "+" if num > 0 else ""
        return f"{sign}{num:.2f}%"
    if abs(num - int(num)) < 1e-9:
        return str(int(num))
    return f"{num:,.2f}"


def pdf_filename(ticker: str, report_date: str) -> str:
    """Build `BIST100_Analiz_[TICKER]_[DATE].pdf` with safe characters."""
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", (ticker or "PORTFOY").upper())[:40]
    safe_date = re.sub(r"[^0-9\-]", "", report_date or date.today().isoformat())
    return f"BIST100_Analiz_{safe_ticker}_{safe_date}.pdf"
