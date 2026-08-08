"""
PDF report generator for BIST 100 Daily AI Analyst.

Embeds bundled Noto Sans TTF fonts (assets/fonts/) so Turkish characters
(ğ, ü, ş, ı, ö, ç, Ğ, Ü, Ş, İ, Ö, Ç) always render — never depends on
Helvetica (Latin-1 only), which produces ■ boxes for Turkish glyphs.
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

# Directory of this module → assets/fonts (bundled with the app)
_FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# Registered once per process
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONT_SOURCE = "fallback-helvetica"
_FONTS_READY = False


def _candidate_font_paths() -> list[tuple[str, str]]:
    """
    Prefer bundled Noto Sans (ships with the repo), then system Unicode fonts.
    """
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return [
        # Bundled — works on localhost AND Streamlit Cloud
        (
            str(_FONTS_DIR / "NotoSans-Regular.ttf"),
            str(_FONTS_DIR / "NotoSans-Bold.ttf"),
        ),
        (
            str(_FONTS_DIR / "DejaVuSans.ttf"),
            str(_FONTS_DIR / "DejaVuSans-Bold.ttf"),
        ),
        # Linux / Streamlit Cloud system packages
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
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
    """
    Register a Unicode TTF pair that supports Turkish glyphs.

    Raises RuntimeError only if literally no TTF is available — Helvetica is
    NEVER used for body text because it cannot draw ı/ğ/ş/İ.
    """
    global _FONT_REGULAR, _FONT_BOLD, _FONT_SOURCE, _FONTS_READY
    if _FONTS_READY and _FONT_REGULAR != "Helvetica":
        return _FONT_REGULAR, _FONT_BOLD

    errors: list[str] = []
    for regular, bold in _candidate_font_paths():
        if not Path(regular).is_file():
            continue
        try:
            # Unique names avoid collisions if Streamlit reloads the module
            pdfmetrics.registerFont(TTFont("BISTNoto", regular))
            bold_path = bold if Path(bold).is_file() else regular
            pdfmetrics.registerFont(TTFont("BISTNoto-Bold", bold_path))

            # Sanity-check: Turkish dotted-I / soft-g must exist in the font
            face = pdfmetrics.getFont("BISTNoto").face
            # reportlab TrueType face stores charToGlyph; missing glyphs → .notdef
            test_chars = "ığüşöçİĞÜŞÖÇ"
            missing = [ch for ch in test_chars if ord(ch) not in face.charToGlyph]
            if missing:
                errors.append(f"{regular}: missing glyphs {missing}")
                continue

            _FONT_REGULAR = "BISTNoto"
            _FONT_BOLD = "BISTNoto-Bold"
            _FONT_SOURCE = regular
            _FONTS_READY = True
            return _FONT_REGULAR, _FONT_BOLD
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{regular}: {exc}")
            continue

    # Last resort: still register any available TTF even if glyph check failed
    for regular, bold in _candidate_font_paths():
        if not Path(regular).is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("BISTNoto", regular))
            bold_path = bold if Path(bold).is_file() else regular
            pdfmetrics.registerFont(TTFont("BISTNoto-Bold", bold_path))
            _FONT_REGULAR = "BISTNoto"
            _FONT_BOLD = "BISTNoto-Bold"
            _FONT_SOURCE = f"{regular} (glyph-check skipped)"
            _FONTS_READY = True
            return _FONT_REGULAR, _FONT_BOLD
        except Exception as exc:  # noqa: BLE001
            errors.append(f"retry {regular}: {exc}")

    raise RuntimeError(
        "Unicode PDF font bulunamadı. "
        "`assets/fonts/NotoSans-Regular.ttf` dosyasının projede olduğundan emin olun. "
        f"Detay: {'; '.join(errors[:3])}"
    )


def get_active_font_source() -> str:
    """Return the filesystem path of the font currently in use (for debugging)."""
    _ensure_fonts()
    return _FONT_SOURCE


def _esc(text: Any) -> str:
    """Escape text for reportlab Paragraph (XML entities). Keep Unicode intact."""
    if text is None:
        return ""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    """Paragraph helper that preserves Turkish characters."""
    return Paragraph(_esc(text), style)


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
            # reportlab <b> uses the bold face of the same font family when registered
            flow.append(Paragraph(f"• {body}", styles["body"]))
            continue

        body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(line.strip()))
        body = body.replace("`", "")
        flow.append(Paragraph(body, styles["body"]))

    return flow


def _build_styles(font_r: str, font_b: str) -> dict:
    base = getSampleStyleSheet()
    # Map <b> tags inside Paragraphs to our bold TTF
    try:
        from reportlab.pdfbase.pdfmetrics import registerFontFamily

        registerFontFamily(
            font_r,
            normal=font_r,
            bold=font_b,
            italic=font_r,
            boldItalic=font_b,
        )
    except Exception:  # noqa: BLE001
        pass

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
        "cell": ParagraphStyle(
            "BISTCell",
            parent=base["Normal"],
            fontName=font_r,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1e293b"),
        ),
        "cell_header": ParagraphStyle(
            "BISTCellHeader",
            parent=base["Normal"],
            fontName=font_b,
            fontSize=8,
            leading=10,
            textColor=colors.whitesmoke,
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


def _table_row(values: list[Any], style: ParagraphStyle) -> list:
    """Wrap every cell in a Paragraph so Unicode fonts always apply."""
    return [_p(v, style) for v in values]


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

    Turkish characters are rendered via the bundled Noto Sans font.
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
    metric_rows = [
        _table_row(
            ["Hisse", "Fiyat", "Günlük %", "RSI", "EMA20", "EMA50"],
            styles["cell_header"],
        )
    ]

    def _add_metric_row(tkr: str, m: dict) -> None:
        if not m:
            return
        metric_rows.append(
            _table_row(
                [
                    str(tkr),
                    _fmt(m.get("current_price")),
                    _fmt(m.get("daily_change_pct"), pct=True),
                    _fmt(m.get("rsi_14")),
                    _fmt(m.get("ema_20")),
                    _fmt(m.get("ema_50")),
                ],
                styles["cell"],
            )
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
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), font_r),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
        trade_rows = [
            _table_row(
                ["Tarih", "Hisse", "İşlem", "Adet", "Fiyat", "Not"],
                styles["cell_header"],
            )
        ]
        for t in trades:
            trade_rows.append(
                _table_row(
                    [
                        str(t.get("date", "")),
                        str(t.get("ticker", "")),
                        str(t.get("action", "")),
                        _fmt(t.get("quantity")),
                        _fmt(t.get("price")),
                        str(t.get("notes", "") or "")[:40],
                    ],
                    styles["cell"],
                )
            )
        ttable = Table(trade_rows, colWidths=[65, 55, 45, 50, 60, 120])
        ttable.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2962ff")),
                    ("FONTNAME", (0, 0), (-1, -1), font_r),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
