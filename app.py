"""
BIST 100 Daily AI Analyst — Streamlit main dashboard.

Modern dark financial terminal UI (TradingView / Bloomberg inspired):
- Trade logging sidebar with universal BIST ticker support (codes or names)
- KPI cards, technical indicator cards
- Full trade history management (edit / delete any trade)
- Scoped Gemini AI analysis (active portfolio OR a specific date's trades)
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ai_analyst
import database as db
import market_data as md
import pdf_generator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BIST 100 Daily AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — TradingView / Bloomberg dark terminal + Inter typography
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

:root {
    --bg-root: #131722;
    --bg-panel: #1e222d;
    --bg-elevated: #252a37;
    --bg-hover: #2a2e39;
    --border: #2a2e39;
    --border-soft: #363a45;
    --text-primary: #d1d4dc;
    --text-secondary: #9598a1;
    --text-muted: #787b86;
    --accent: #2962ff;
    --green: #26a69a;
    --green-dim: rgba(38, 166, 154, 0.14);
    --red: #ef5350;
    --red-dim: rgba(239, 83, 80, 0.14);
    --amber: #ff9800;
    --amber-dim: rgba(255, 152, 0, 0.12);
    --blue-dim: rgba(41, 98, 255, 0.14);
}

.stApp {
    background: var(--bg-root);
    color: var(--text-primary);
}

/* Keep header transparent but DO NOT hide it — sidebar toggle lives here */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3rem;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ========== Sidebar expand/collapse controls (ALWAYS visible) ========== */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
button[kind="headerNoPadding"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    position: fixed !important;
    top: 0.65rem !important;
    left: 0.65rem !important;
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
    padding: 0.15rem !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] button,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg {
    color: #d1d4dc !important;
    fill: #d1d4dc !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stExpandSidebarButton"]:hover {
    border-color: var(--accent) !important;
    background: var(--bg-elevated) !important;
}

/* Sidebar panel */
section[data-testid="stSidebar"] {
    background: #0f1219 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
    background: #0f1219 !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaption"] {
    color: var(--text-muted) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
.stDateInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Main content breathing room under fixed toggle */
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1280px;
}

/* ========== Header ========== */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    background: linear-gradient(135deg, #1a1f2e 0%, #1e222d 55%, #181c27 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.app-badge {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: linear-gradient(145deg, #2962ff, #1e88e5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 0 18px rgba(41,98,255,0.35);
}
.app-header h1 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
    line-height: 1.25;
}
.app-header p {
    margin: 0.2rem 0 0 0;
    color: var(--text-secondary);
    font-size: 0.82rem;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid rgba(38,166,154,0.35);
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
}
.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}

/* ========== Weekend closed notice ========== */
.weekend-banner {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    background: linear-gradient(135deg, rgba(255,152,0,0.12) 0%, rgba(30,34,45,0.95) 55%);
    border: 1px solid rgba(255,152,0,0.35);
    border-left: 4px solid var(--amber);
    border-radius: 12px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.22);
}
.weekend-banner-icon {
    flex-shrink: 0;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--amber-dim);
    border: 1px solid rgba(255,152,0,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
}
.weekend-banner-body {
    flex: 1;
    min-width: 0;
}
.weekend-banner-title {
    margin: 0;
    font-size: 0.92rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.35;
}
.weekend-banner-sub {
    margin: 0.25rem 0 0 0;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.weekend-badge {
    flex-shrink: 0;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--amber);
    background: var(--amber-dim);
    border: 1px solid rgba(255,152,0,0.35);
    border-radius: 999px;
    padding: 0.28rem 0.65rem;
    white-space: nowrap;
}

/* ========== KPI cards ========== */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.15rem;
}
.kpi-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 18px rgba(0,0,0,0.22);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.kpi-blue::before { background: linear-gradient(90deg, #2962ff, #64b5f6); }
.kpi-card.kpi-teal::before { background: linear-gradient(90deg, #26a69a, #80cbc4); }
.kpi-card.kpi-up::before { background: linear-gradient(90deg, #26a69a, #66bb6a); }
.kpi-card.kpi-down::before { background: linear-gradient(90deg, #ef5350, #e57373); }
.kpi-card.kpi-flat::before { background: linear-gradient(90deg, #787b86, #9598a1); }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', 'Inter', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.1;
}
.kpi-sub {
    margin-top: 0.35rem;
    font-size: 0.8rem;
    font-weight: 600;
}
.kpi-sub.pos { color: var(--green); }
.kpi-sub.neg { color: var(--red); }
.kpi-sub.flat { color: var(--text-muted); }

/* ========== Section / cards ========== */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    margin: 0.4rem 0 0.75rem 0;
}
.section-title span.icon { font-size: 1.05rem; }

.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}

/* Tech cards */
.tech-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.tech-card:hover {
    border-color: #3d4454;
    transform: translateY(-1px);
}
.tech-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--border);
}
.tech-card h3 {
    margin: 0;
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
}
.chip {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.chip.pos { background: var(--green-dim); color: var(--green); }
.chip.neg { background: var(--red-dim); color: var(--red); }
.chip.flat { background: rgba(120,123,134,0.2); color: var(--text-secondary); }

.tech-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.65rem 0.75rem;
}
.tech-item label {
    display: block;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
}
.tech-item span {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}
.pos { color: var(--green) !important; }
.neg { color: var(--red) !important; }
.warn { color: var(--amber) !important; }

/* Sidebar trade chips */
.trade-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.4rem;
}
.trade-chip .meta {
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.badge-buy {
    display: inline-block;
    background: var(--green-dim);
    color: var(--green);
    font-weight: 700;
    font-size: 0.68rem;
    padding: 0.12rem 0.4rem;
    border-radius: 4px;
    margin-right: 0.35rem;
    letter-spacing: 0.04em;
}
.badge-sell {
    display: inline-block;
    background: var(--red-dim);
    color: var(--red);
    font-weight: 700;
    font-size: 0.68rem;
    padding: 0.12rem 0.4rem;
    border-radius: 4px;
    margin-right: 0.35rem;
    letter-spacing: 0.04em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.3rem;
    margin-bottom: 0.85rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: #fff !important;
    box-shadow: inset 0 -2px 0 var(--accent);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #2962ff, #1e88e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(41,98,255,0.4) !important;
    color: #fff !important;
}
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--bg-panel);
}
div[data-testid="stDataFrame"] * {
    font-family: 'Inter', sans-serif !important;
}

/* AI alert cards */
.ai-wrap { margin-top: 0.5rem; }
.ai-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    border-left: 3px solid var(--accent);
}
.ai-card.card-summary { border-left-color: #2962ff; }
.ai-card.card-discipline { border-left-color: #26a69a; }
.ai-card.card-technical { border-left-color: #42a5f5; }
.ai-card.card-risk { border-left-color: #ef5350; }
.ai-card.card-levels { border-left-color: #ff9800; }
.ai-card.card-strategy { border-left-color: #ab47bc; }
.ai-card.card-news { border-left-color: #00bcd4; }
.ai-card.card-score { border-left-color: #ff9800; }
.ai-card.card-default { border-left-color: #787b86; }

/* Score indicator badge */
.score-banner {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}
.score-banner.tone-pos { border-left: 4px solid var(--green); }
.score-banner.tone-neg { border-left: 4px solid var(--red); }
.score-banner.tone-flat { border-left: 4px solid var(--text-muted); }
.score-left { display: flex; align-items: center; gap: 0.85rem; }
.score-emoji { font-size: 1.8rem; }
.score-meta label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
}
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.score-label {
    margin-top: 0.35rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
}
.score-reasons {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0;
    padding-left: 1.1rem;
}

.ai-card-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 0.55rem 0;
}
.ai-card-body {
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.55;
}
.ai-card-body ul {
    margin: 0.35rem 0 0.2rem 1.1rem;
    padding: 0;
}
.ai-card-body li { margin-bottom: 0.25rem; }
.ai-card-body p { margin: 0.25rem 0; }
.ai-card-body strong { color: #fff; }
.ai-card-body table.md-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.55rem 0 0.75rem 0;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', 'Inter', monospace;
}
.ai-card-body table.md-table th,
.ai-card-body table.md-table td {
    border: 1px solid var(--border);
    padding: 0.4rem 0.5rem;
    text-align: left;
    vertical-align: top;
}
.ai-card-body table.md-table th {
    background: var(--bg-elevated);
    color: #fff;
    font-weight: 700;
    white-space: nowrap;
}
.ai-card-body table.md-table tr:nth-child(even) td {
    background: rgba(37, 42, 55, 0.55);
}
.ai-card-body .verdict-good { color: var(--green); font-weight: 700; }
.ai-card-body .verdict-risk { color: var(--amber); font-weight: 700; }
.ai-card-body .verdict-bad { color: var(--red); font-weight: 700; }

.empty-state {
    background: var(--bg-panel);
    border: 1px dashed var(--border-soft);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: var(--text-secondary);
}

@media (max-width: 900px) {
    .kpi-row { grid-template-columns: 1fr; }
    .tech-grid { grid-template-columns: repeat(2, 1fr); }
    .app-header { flex-direction: column; align-items: flex-start; }
    .weekend-banner { flex-wrap: wrap; }
    .weekend-badge { margin-left: auto; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_price(value) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _change_class(value) -> str:
    if value is None:
        return "flat"
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "flat"


def _rsi_class(rsi) -> str:
    if rsi is None:
        return ""
    if rsi >= 70:
        return "neg"
    if rsi <= 30:
        return "pos"
    return "warn"


def _parse_iso_date(value: str) -> date:
    """Parse an ISO date string, falling back to today on bad data."""
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return date.today()


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def is_weekend(day: date | None = None) -> bool:
    """True when the given (or local system) date falls on Saturday or Sunday."""
    d = day or date.today()
    return d.weekday() >= 5  # 5=Saturday, 6=Sunday


def weekend_day_label(day: date | None = None) -> str:
    """Turkish weekday name for Saturday / Sunday."""
    d = day or date.today()
    return "Cumartesi" if d.weekday() == 5 else "Pazar"


def render_weekend_closed_notice() -> None:
    """
    Show a prominent notice when Borsa Istanbul is closed for the weekend.
    Only renders on Saturday (weekday=5) or Sunday (weekday=6).
    """
    if not is_weekend():
        return

    day_name = weekend_day_label()
    st.markdown(
        f"""
        <div class="weekend-banner" role="status" aria-live="polite">
          <div class="weekend-banner-icon">🔒</div>
          <div class="weekend-banner-body">
            <p class="weekend-banner-title">
              Borsa İstanbul hafta sonu nedeniyle kapalıdır. (Cumartesi / Pazar)
            </p>
            <p class="weekend-banner-sub">
              Bugün {html.escape(day_name)} — piyasa verileri son işlem gününe ait olabilir.
            </p>
          </div>
          <span class="weekend-badge">Kapalı</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_clock() -> None:
    """
    Client-side HH:MM:SS digital clock (Europe/Istanbul).

    Uses an HTML/JS iframe so the clock ticks every second without
    triggering a Streamlit rerun (page stays responsive).
    """
    # Seed with server time so the first paint is never blank/00:00:00
    seed = datetime.now().strftime("%H:%M:%S")
    components.html(
        f"""
        <style>
          html, body {{
            margin: 0; padding: 0; background: transparent; overflow: hidden;
            font-family: 'JetBrains Mono', 'Inter', ui-monospace, monospace;
          }}
          .clock-shell {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: center;
            height: 52px;
            padding: 0 0.15rem;
            box-sizing: border-box;
          }}
          .clock-label {{
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #787b86;
            margin-bottom: 0.2rem;
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
          }}
          .digital-clock {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: 0.06em;
            font-variant-numeric: tabular-nums;
            background: #1e222d;
            border: 1px solid #363a45;
            border-radius: 10px;
            padding: 0.28rem 0.7rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.28);
            line-height: 1.2;
            min-width: 6.6ch;
            text-align: center;
          }}
        </style>
        <div class="clock-shell" title="Canlı saat (Europe/Istanbul)">
          <div class="clock-label">⏱ Canlı Saat</div>
          <div id="bist-live-clock" class="digital-clock">{seed}</div>
        </div>
        <script>
        (function () {{
          const el = document.getElementById('bist-live-clock');
          if (!el) return;

          const fmt = new Intl.DateTimeFormat('tr-TR', {{
            timeZone: 'Europe/Istanbul',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
          }});

          function tick() {{
            // tr-TR may use '.' as separator — normalize to HH:MM:SS
            el.textContent = fmt.format(new Date()).replace(/\\./g, ':');
          }}

          tick();
          setInterval(tick, 1000);
        }})();
        </script>
        """,
        height=56,
    )


def render_sidebar_toggle_button() -> None:
    """
    Reliable sidebar toggle: HTML button lives in the component iframe and
    clicks Streamlit's native expand/collapse controls in the parent document.
    """
    components.html(
        """
        <style>
          html, body {
            margin: 0; padding: 0; background: transparent; overflow: hidden;
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
          }
          button#bist-sidebar-toggle {
            width: 100%;
            height: 42px;
            cursor: pointer;
            border-radius: 8px;
            border: 1px solid #363a45;
            background: #252a37;
            color: #d1d4dc;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            transition: border-color 0.12s ease, background 0.12s ease;
          }
          button#bist-sidebar-toggle:hover {
            border-color: #2962ff;
            background: #2a2e39;
            color: #fff;
          }
        </style>
        <button id="bist-sidebar-toggle" title="Sol paneli aç / kapat" type="button">
          ☰ Panel
        </button>
        <script>
        (function () {
          function findToggle(doc) {
            const selectors = [
              '[data-testid="stSidebarCollapseButton"] button',
              '[data-testid="stSidebarCollapseButton"]',
              '[data-testid="stExpandSidebarButton"] button',
              '[data-testid="stExpandSidebarButton"]',
              '[data-testid="stSidebarCollapsedControl"] button',
              '[data-testid="stSidebarCollapsedControl"]',
              '[data-testid="collapsedControl"] button',
              '[data-testid="collapsedControl"]',
              'button[kind="headerNoPadding"]',
              'button[kind="header"]'
            ];
            for (const sel of selectors) {
              const el = doc.querySelector(sel);
              if (el) return el;
            }
            const buttons = Array.from(doc.querySelectorAll('button'));
            return buttons.find((b) => {
              const label = (b.getAttribute('aria-label') || '').toLowerCase();
              return label.includes('sidebar') || label.includes('side bar')
                     || label.includes('kenar çubuğu') || label.includes('kenar cubugu');
            }) || null;
          }

          function toggleSidebar() {
            const doc = window.parent.document;
            const btn = findToggle(doc);
            if (btn) { btn.click(); return; }
            try {
              doc.dispatchEvent(new KeyboardEvent('keydown', {
                key: '[', code: 'BracketLeft', keyCode: 219, which: 219, bubbles: true
              }));
            } catch (e) {}
          }

          document.getElementById('bist-sidebar-toggle')
            .addEventListener('click', function (e) {
              e.preventDefault();
              toggleSidebar();
            });
        })();
        </script>
        """,
        height=48,
    )


def style_trades_table(df: pd.DataFrame):
    """Apply green/red action coloring to a trades dataframe."""
    def color_action(val):
        v = str(val).upper()
        if v == "BUY":
            return "color: #26a69a; font-weight: 700; background-color: rgba(38,166,154,0.12);"
        if v == "SELL":
            return "color: #ef5350; font-weight: 700; background-color: rgba(239,83,80,0.12);"
        return ""

    fmt: dict = {}
    if "Fiyat" in df.columns:
        fmt["Fiyat"] = "{:,.2f}"
    if "Adet" in df.columns:
        fmt["Adet"] = "{:g}"

    return (
        df.style
        .map(color_action, subset=["İşlem"] if "İşlem" in df.columns else [])
        .format(fmt, na_rep="—")
        .set_properties(**{
            "background-color": "#1e222d",
            "color": "#d1d4dc",
            "border-color": "#2a2e39",
        })
    )


def show_trades_df(df: pd.DataFrame) -> None:
    """Render a trades dataframe with styling and a safe fallback."""
    try:
        st.dataframe(style_trades_table(df), use_container_width=True, hide_index=True)
    except Exception:  # noqa: BLE001 — Styler edge cases on some pandas builds
        st.dataframe(df, use_container_width=True, hide_index=True)


TRADE_COLUMN_LABELS = {
    "id": "ID",
    "date": "Tarih",
    "ticker": "Hisse",
    "action": "İşlem",
    "quantity": "Adet",
    "price": "Fiyat",
    "notes": "Not",
}


def render_kpi_row(trade_count: int, active_positions: int, bist: dict) -> None:
    """Render three Bloomberg-style KPI cards."""
    bist_ok = bist.get("success")
    delta = bist.get("daily_change_pct") if bist_ok else None
    trend_class = _change_class(delta)
    kpi_trend = {"pos": "kpi-up", "neg": "kpi-down", "flat": "kpi-flat"}.get(
        trend_class, "kpi-flat"
    )

    if bist_ok:
        bist_value = _fmt_price(bist.get("current_price"))
        bist_sub = f'<div class="kpi-sub {trend_class}">{_fmt_pct(delta)} günlük</div>'
    else:
        bist_value = "Veri yok"
        bist_sub = (
            f'<div class="kpi-sub flat">'
            f'{html.escape(str(bist.get("error") or "Endeks alınamadı"))}</div>'
        )

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card kpi-blue">
            <div class="kpi-label">Bugünün İşlem Sayısı</div>
            <div class="kpi-value">{trade_count}</div>
            <div class="kpi-sub flat">bugün kaydedilen</div>
          </div>
          <div class="kpi-card kpi-teal">
            <div class="kpi-label">Aktif Pozisyonlar</div>
            <div class="kpi-value">{active_positions}</div>
            <div class="kpi-sub flat">açık net pozisyon</div>
          </div>
          <div class="kpi-card {kpi_trend}">
            <div class="kpi-label">BIST 100 Trend</div>
            <div class="kpi-value">{bist_value}</div>
            {bist_sub}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tech_card(summary: dict) -> None:
    """Render a technical indicator card for one ticker."""
    if not summary.get("success"):
        st.warning(f"**{summary.get('ticker', '?')}**: {summary.get('error', 'Veri yok')}")
        return

    change = summary.get("daily_change_pct")
    rsi = summary.get("rsi_14")
    chip_cls = _change_class(change)

    st.markdown(
        f"""
        <div class="tech-card">
            <div class="tech-card-head">
                <h3>{html.escape(str(summary['ticker']))}</h3>
                <span class="chip {chip_cls}">{_fmt_pct(change)}</span>
            </div>
            <div class="tech-grid">
                <div class="tech-item">
                    <label>Son Fiyat</label>
                    <span>{_fmt_price(summary.get('current_price'))} ₺</span>
                </div>
                <div class="tech-item">
                    <label>Günlük %</label>
                    <span class="{chip_cls}">{_fmt_pct(change)}</span>
                </div>
                <div class="tech-item">
                    <label>RSI (14)</label>
                    <span class="{_rsi_class(rsi)}">{rsi if rsi is not None else '—'}</span>
                </div>
                <div class="tech-item">
                    <label>EMA 20</label>
                    <span>{_fmt_price(summary.get('ema_20'))}</span>
                </div>
                <div class="tech-item">
                    <label>EMA 50</label>
                    <span>{_fmt_price(summary.get('ema_50'))}</span>
                </div>
                <div class="tech-item">
                    <label>60G Aralık</label>
                    <span>{_fmt_price(summary.get('low_60d'))} – {_fmt_price(summary.get('high_60d'))}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# AI report rendering (markdown -> structured alert cards)
# ---------------------------------------------------------------------------
def _inline_md(text: str) -> str:
    """Minimal inline markdown -> HTML (bold / code / escape + verdict tags)."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Highlight mandatory verdict tags
    text = re.sub(
        r"\[ÇOK İYİ HAMLE\]",
        r'<span class="verdict-good">[ÇOK İYİ HAMLE]</span>',
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[RİSKLİ\s*/\s*NÖTR\]|\[RISKLI\s*/\s*NOTR\]",
        r'<span class="verdict-risk">[RİSKLİ / NÖTR]</span>',
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[HATALI\s*/\s*TEHLİKELİ\]|\[HATALI\s*/\s*TEHLIKELI\]",
        r'<span class="verdict-bad">[HATALI / TEHLİKELİ]</span>',
        text,
        flags=re.I,
    )
    return text


def _split_md_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [c.strip() for c in raw.split("|")]


def _is_md_separator(line: str) -> bool:
    cells = _split_md_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c != "")


def _md_table_to_html(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    thead = "".join(f"<th>{_inline_md(c)}</th>" for c in header)
    trs = []
    for r in body:
        # Pad short rows to header width
        padded = r + [""] * max(0, len(header) - len(r))
        trs.append(
            "<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in padded[: len(header)]) + "</tr>"
        )
    return (
        '<div style="overflow-x:auto">'
        f'<table class="md-table"><thead><tr>{thead}</tr></thead>'
        f"<tbody>{''.join(trs)}</tbody></table></div>"
    )


def _body_to_html(body: str) -> str:
    """Convert a markdown body chunk into HTML (lists, paragraphs, full tables)."""
    lines = body.strip().splitlines()
    parts: list[str] = []
    in_list = False
    i = 0

    def _close_list() -> None:
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # Markdown pipe table
        if (
            "|" in line
            and i + 1 < len(lines)
            and "|" in lines[i + 1]
            and _is_md_separator(lines[i + 1])
        ):
            _close_list()
            rows = [_split_md_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                if _is_md_separator(lines[i]):
                    i += 1
                    continue
                rows.append(_split_md_row(lines[i]))
                i += 1
            parts.append(_md_table_to_html(rows))
            continue

        if not line.strip():
            _close_list()
            i += 1
            continue

        bullet = re.match(r"^[-*•]\s+(.*)$", line.strip())
        numbered = re.match(r"^\d+[.)]\s+(.*)$", line.strip())
        if bullet or numbered:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            content = bullet.group(1) if bullet else numbered.group(1)
            parts.append(f"<li>{_inline_md(content)}</li>")
        else:
            _close_list()
            cleaned = re.sub(r"^#{1,6}\s*", "", line.strip())
            parts.append(f"<p>{_inline_md(cleaned)}</p>")
        i += 1

    _close_list()
    return "\n".join(parts) if parts else f"<p>{_inline_md(body)}</p>"


def _classify_section(title: str) -> tuple[str, str]:
    """Map section title -> (css_class, icon)."""
    t = title.lower()
    if any(k in t for k in ("haber", "sektörel", "sektorel", "news")):
        return "card-news", "📰"
    if any(
        k in t
        for k in (
            "değişim puan",
            "degisim puan",
            "beklenen",
            "score",
            "puan",
            "genel portföy",
            "genel portfoy",
            "kararı",
            "karari",
        )
    ):
        return "card-score", "📊"
    if any(k in t for k in ("özet", "ozet", "summary", "günlük", "portföy", "portfoy")):
        return "card-summary", "📋"
    if any(k in t for k in ("disiplin", "discipline")):
        return "card-discipline", "✅"
    if any(
        k in t
        for k in (
            "isabet",
            "hisse bazlı",
            "hisse bazli",
            "teknik",
            "işlem",
            "islem",
            "pozisyon",
            "yorum",
            "değerlendirme",
            "degerlendirme",
        )
    ):
        return "card-technical", "📈"
    if any(k in t for k in ("risk", "konsantrasyon", "concentration")):
        return "card-risk", "⚠️"
    if any(k in t for k in ("destek", "direnç", "direnc", "seviye", "yarın", "yarin")):
        return "card-levels", "🎯"
    if any(
        k in t
        for k in ("strateji", "strategy", "motivasyon", "öneri", "oneri", "reçete", "recete")
    ):
        return "card-strategy", "💡"
    return "card-default", "📌"


def render_score_banner(score_info: dict) -> None:
    """Prominent -10..+10 expected-change score card."""
    score = score_info.get("score")
    meta = ai_analyst.score_badge_meta(score)
    if score is None:
        return

    sign = f"+{score}" if score > 0 else str(score)
    label = html.escape(score_info.get("label") or meta["title"])
    reasons = score_info.get("reasons") or []
    reasons_html = ""
    if reasons:
        items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons[:3])
        reasons_html = f"<ul class='score-reasons'>{items}</ul>"

    st.markdown(
        f"""
        <div class="score-banner tone-{meta['tone']}">
          <div class="score-left">
            <div class="score-emoji">{meta['emoji']}</div>
            <div class="score-meta">
              <label>Beklenen Değişim Puanı</label>
              <div class="score-value">{sign}/10</div>
              <div class="score-label">{label}</div>
            </div>
          </div>
          <div>{reasons_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_report(report: str) -> None:
    """
    Parse Gemini Markdown into structured alert cards.
    Falls back to a single panel if no headers are found.
    """
    if not report or not report.strip():
        st.info("Rapor boş geldi.")
        return

    chunks = re.split(r"(?m)^(#{1,3})\s+(.+)$", report.strip())

    if len(chunks) < 4:
        st.markdown(
            f"""
            <div class="ai-card card-summary">
              <div class="ai-card-title">🤖 AI Analiz Raporu</div>
              <div class="ai-card-body">{_body_to_html(report)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    preamble = chunks[0].strip()
    if preamble:
        st.markdown(
            f"""
            <div class="ai-card card-summary">
              <div class="ai-card-title">🤖 Genel Not</div>
              <div class="ai-card-body">{_body_to_html(preamble)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    i = 1
    while i + 2 < len(chunks):
        title = chunks[i + 1].strip()
        body = chunks[i + 2].strip()
        css_cls, icon = _classify_section(title)
        st.markdown(
            f"""
            <div class="ai-card {css_cls}">
              <div class="ai-card-title">{icon} {html.escape(title)}</div>
              <div class="ai-card-body">{_body_to_html(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        i += 3


# ---------------------------------------------------------------------------
# Cached market data fetchers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_summary(ticker: str) -> dict:
    """Cache market summaries for 5 minutes to reduce Yahoo rate limits."""
    return md.get_stock_summary(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_bist100() -> dict:
    """Cache BIST 100 index snapshot for 5 minutes."""
    return md.get_bist100_index_summary()


@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_news(ticker: str) -> dict:
    """Cache news headlines for 5 minutes."""
    return md.get_stock_news(ticker, limit=5)


def fetch_market_dict(tickers: list[str]) -> dict[str, dict]:
    """Fetch cached summaries for a list of tickers (normalized, de-duplicated)."""
    result: dict[str, dict] = {}
    for t in tickers:
        bare = md.bare_ticker(t)
        if bare and bare not in result:
            result[bare] = cached_stock_summary(bare)
    return result


def fetch_news_dict(tickers: list[str]) -> dict[str, dict]:
    """Fetch cached news for EVERY ticker (sector fallback when empty)."""
    result: dict[str, dict] = {}
    for t in tickers:
        bare = md.bare_ticker(t)
        if bare and bare not in result:
            result[bare] = cached_stock_news(bare)
    return result


def clear_market_data_caches() -> None:
    """Drop cached Yahoo/BIST snapshots so the next render fetches fresh quotes."""
    cached_stock_summary.clear()
    cached_bist100.clear()
    cached_stock_news.clear()


def render_market_refresh_bar() -> None:
    """
    Compact toolbar under the KPI row: refresh market caches + last-refresh time.

    Quotes are cached for 5 minutes; this lets the user force a live pull without
    restarting the app.
    """
    if "market_last_refresh" not in st.session_state:
        st.session_state["market_last_refresh"] = datetime.now()

    left, right = st.columns([3.2, 1])
    with left:
        stamp = st.session_state["market_last_refresh"].strftime("%H:%M:%S")
        st.caption(
            f"Piyasa önbelleği · son yenileme {stamp} · "
            "otomatik TTL 5 dk (Yahoo rate-limit koruması)"
        )
    with right:
        if st.button(
            "🔄 Verileri Yenile",
            key="btn_refresh_market",
            use_container_width=True,
            help="BIST 100 ve hisse özet önbelleğini temizleyip taze veri çeker.",
        ):
            clear_market_data_caches()
            st.session_state["market_last_refresh"] = datetime.now()
            st.toast("Piyasa verileri yenilendi.", icon="📈")
            st.rerun()


def _flatten_news(news_dict: dict[str, dict]) -> list[dict]:
    """
    Flatten per-ticker news for the PDF.

    Tickers with no headlines still get a sector placeholder row so the PDF
    never silently drops a portfolio name from the news section.
    """
    flat: list[dict] = []
    for ticker, payload in (news_dict or {}).items():
        sector = (payload or {}).get("sector") or md.get_ticker_sector(ticker)
        items = (payload or {}).get("news") or []
        if items:
            for item in items:
                row = dict(item)
                row["ticker"] = ticker
                row["sector"] = sector
                title = row.get("title") or ""
                if ticker and ticker not in title:
                    row["title"] = f"[{ticker}] {title}"
                flat.append(row)
        else:
            flat.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "title": f"[{ticker}] Doğrudan haber yok — {sector} sektör görünümü",
                    "publisher": "Sektör Notu",
                    "summary": (payload or {}).get("error")
                    or f"{ticker} için şirket haberi bulunamadı; sektör bağlamı kullanılmalı.",
                    "link": "",
                    "published": "",
                }
            )
    return flat


# ---------------------------------------------------------------------------
# Sidebar — trade logger
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✍️ İşlem Kaydı")
    st.caption("BIST hissesi için AL / SAT kaydı oluşturun.")

    with st.form("trade_form", clear_on_submit=True):
        trade_date = st.date_input("Tarih", value=date.today())

        ticker_mode = st.radio(
            "Ticker girişi",
            options=["Listeden seç", "Manuel yaz"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if ticker_mode == "Listeden seç":
            ticker_input = st.selectbox("Hisse", options=md.BIST100_TICKERS, index=0)
        else:
            ticker_input = st.text_input(
                "Hisse Kodu veya Adı",
                placeholder="örn. THYAO veya Alcatel",
                help="Kod ya da şirket adı yazın — `.IS` otomatik eklenir "
                     "(Alcatel → ALCTL.IS).",
            )

        action = st.selectbox(
            "İşlem",
            options=["BUY", "SELL"],
            format_func=lambda x: "AL (BUY)" if x == "BUY" else "SAT (SELL)",
        )
        quantity = st.number_input("Adet", min_value=0.01, value=100.0, step=1.0)
        price = st.number_input(
            "Birim Fiyat (₺)", min_value=0.01, value=100.0, step=0.01, format="%.4f"
        )
        notes = st.text_input("Not (opsiyonel)", placeholder="Kısa not...")

        submitted = st.form_submit_button("Kaydet", use_container_width=True)

        if submitted:
            clean_ticker = md.bare_ticker(ticker_input or "")
            if not clean_ticker:
                st.error("Lütfen geçerli bir hisse kodu veya adı girin.")
            else:
                try:
                    trade_id = db.add_trade(
                        ticker=clean_ticker,
                        action=action,
                        quantity=float(quantity),
                        price=float(price),
                        trade_date=trade_date.isoformat(),
                        notes=notes.strip(),
                    )
                    st.success(f"#{trade_id} kaydedildi: {action} {clean_ticker}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Kayıt hatası: {exc}")

    st.markdown("---")
    st.markdown("### 🗓️ Bugünün İşlemleri")

    todays_sidebar = db.get_todays_trades()
    if todays_sidebar.empty:
        st.caption("Bugün henüz işlem yok.")
    else:
        for _, row in todays_sidebar.iterrows():
            badge = "badge-buy" if row["action"] == "BUY" else "badge-sell"
            st.markdown(
                f"""
                <div class="trade-chip">
                  <div>
                    <span class="{badge}">{row['action']}</span>
                    <strong style="color:#fff">{html.escape(str(row['ticker']))}</strong>
                    <div class="meta">{row['quantity']:g} × {_fmt_price(row['price'])} ₺</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Sil", key=f"sb_del_{row['id']}", help="Bu işlemi sil"):
                db.delete_trade(int(row["id"]))
                st.rerun()

    st.markdown("---")
    holdings_sidebar = db.get_current_holdings()
    st.markdown("### 📦 Açık Pozisyonlar")
    if holdings_sidebar.empty:
        st.caption("Aktif pozisyon yok.")
    else:
        st.dataframe(
            holdings_sidebar.rename(
                columns={
                    "ticker": "Hisse",
                    "quantity": "Adet",
                    "avg_buy_price": "Ort. Maliyet",
                    "cost_basis": "Maliyet",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main page — header + live clock + weekend notice + KPI row
# ---------------------------------------------------------------------------
top_l, top_clock, top_r = st.columns([4.4, 1.15, 1])
with top_l:
    st.markdown(
        """
        <div class="app-header">
          <div class="app-header-left">
            <div class="app-badge">📈</div>
            <div>
              <h1>BIST 100 Kişisel Portföy & Yapay Zeka Asistanı</h1>
              <p>Günlük işlem kaydı · teknik göstergeler · Gemini AI koçluk raporu</p>
            </div>
          </div>
          <div class="live-pill"><span class="live-dot"></span> LIVE TERMINAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_clock:
    st.write("")  # vertical align with header
    render_live_clock()
with top_r:
    st.write("")  # vertical align with header
    st.caption("Sol paneli aç / kapat")
    render_sidebar_toggle_button()

# Weekend market-closed banner (Saturday / Sunday only)
render_weekend_closed_notice()

todays_trades = db.get_todays_trades()
active_positions = db.count_active_positions()
bist = cached_bist100()
render_kpi_row(len(todays_trades), active_positions, bist)
render_market_refresh_bar()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Bugünün İşlemleri & Teknik Durum",
        "📜 Tüm İşlem Geçmişi & Yönetim",
        "🤖 AI Analiz Raporu",
    ]
)

# ============================ TAB 1 — Today ================================
with tab1:
    st.markdown(
        '<div class="section-title"><span class="icon">📒</span> Bugün Kaydedilen İşlemler</div>',
        unsafe_allow_html=True,
    )

    if todays_trades.empty:
        st.markdown(
            '<div class="empty-state">Soldaki panelden ilk işleminizi ekleyerek başlayın.<br/>'
            'Sol üstteki <strong>›</strong> okuna veya <strong>☰ Panel</strong> butonuna basarak '
            'menüyü açabilirsiniz.</div>',
            unsafe_allow_html=True,
        )
    else:
        show_trades_df(todays_trades.rename(columns=TRADE_COLUMN_LABELS))

    st.markdown(
        '<div class="section-title" style="margin-top:1.1rem">'
        '<span class="icon">📡</span> Teknik Gösterge Kartları</div>',
        unsafe_allow_html=True,
    )

    tickers_to_show: list[str] = []
    if not todays_trades.empty:
        tickers_to_show = todays_trades["ticker"].dropna().unique().tolist()
    else:
        holdings_df = db.get_current_holdings()
        if not holdings_df.empty:
            tickers_to_show = holdings_df["ticker"].tolist()

    if not tickers_to_show:
        st.caption("Teknik kartlar için bugün işlem yapın veya açık pozisyon bulundurun.")
    else:
        with st.spinner("Piyasa verileri çekiliyor..."):
            summaries = fetch_market_dict(tickers_to_show)

        cols = st.columns(2)
        for i, (ticker, summary) in enumerate(summaries.items()):
            with cols[i % 2]:
                render_tech_card(summary)

# ==================== TAB 2 — Full history & management ====================
with tab2:
    st.markdown(
        '<div class="section-title"><span class="icon">📜</span> Tüm İşlem Geçmişi</div>',
        unsafe_allow_html=True,
    )

    # Flash message from the previous update/delete action (survives rerun)
    flash = st.session_state.pop("_mgmt_flash", None)
    if flash:
        level, message = flash
        (st.success if level == "success" else st.error)(message)

    all_trades = db.get_all_trades()

    if all_trades.empty:
        st.markdown(
            '<div class="empty-state">Henüz hiç işlem kaydı yok. '
            "Soldaki panelden ilk işleminizi ekleyin.</div>",
            unsafe_allow_html=True,
        )
    else:
        show_trades_df(all_trades.rename(columns=TRADE_COLUMN_LABELS))

        st.markdown(
            '<div class="section-title" style="margin-top:1.1rem">'
            '<span class="icon">🛠️</span> İşlem Düzenle / Sil</div>',
            unsafe_allow_html=True,
        )

        id_options = all_trades["id"].tolist()
        label_by_id = {
            int(row["id"]): (
                f"#{int(row['id'])} · {row['date']} · {row['action']} "
                f"{row['ticker']} · {row['quantity']:g} × {_fmt_price(row['price'])} ₺"
            )
            for _, row in all_trades.iterrows()
        }

        selected_id = st.selectbox(
            "Düzenlenecek / silinecek işlemi seçin",
            options=id_options,
            format_func=lambda i: label_by_id.get(int(i), f"#{i}"),
        )

        trade = db.get_trade_by_id(int(selected_id)) if selected_id is not None else None

        if trade is None:
            st.warning("Seçilen işlem bulunamadı. Sayfayı yenileyin.")
        else:
            # Key the form by trade id so defaults refresh when selection changes
            with st.form(f"edit_form_{trade['id']}"):
                c1, c2 = st.columns(2)
                with c1:
                    e_date = st.date_input("Tarih", value=_parse_iso_date(trade["date"]))
                    e_ticker = st.text_input(
                        "Hisse Kodu veya Adı",
                        value=str(trade["ticker"]),
                        help="Kod ya da şirket adı — otomatik normalize edilir.",
                    )
                    e_action = st.selectbox(
                        "İşlem",
                        options=["BUY", "SELL"],
                        index=0 if str(trade["action"]).upper() == "BUY" else 1,
                        format_func=lambda x: "AL (BUY)" if x == "BUY" else "SAT (SELL)",
                    )
                with c2:
                    e_quantity = st.number_input(
                        "Adet", min_value=0.01, value=float(trade["quantity"]), step=1.0
                    )
                    e_price = st.number_input(
                        "Birim Fiyat (₺)",
                        min_value=0.01,
                        value=float(trade["price"]),
                        step=0.01,
                        format="%.4f",
                    )
                    e_notes = st.text_input("Not", value=str(trade["notes"] or ""))

                bc1, bc2 = st.columns(2)
                update_clicked = bc1.form_submit_button(
                    "💾 Güncelle", use_container_width=True
                )
                delete_clicked = bc2.form_submit_button(
                    "🗑️ Sil", use_container_width=True
                )

            if update_clicked:
                try:
                    normalized = md.bare_ticker(e_ticker)
                    if not normalized:
                        raise ValueError("Geçerli bir hisse kodu girin.")
                    ok = db.update_trade(
                        int(trade["id"]),
                        e_date.isoformat(),
                        normalized,
                        e_action,
                        float(e_quantity),
                        float(e_price),
                        e_notes.strip(),
                    )
                    st.session_state["_mgmt_flash"] = (
                        ("success", f"#{trade['id']} güncellendi ({normalized}).")
                        if ok
                        else ("error", f"#{trade['id']} bulunamadı — güncellenemedi.")
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["_mgmt_flash"] = ("error", f"Güncelleme hatası: {exc}")
                st.rerun()

            if delete_clicked:
                try:
                    ok = db.delete_trade(int(trade["id"]))
                    st.session_state["_mgmt_flash"] = (
                        ("success", f"#{trade['id']} silindi.")
                        if ok
                        else ("error", f"#{trade['id']} bulunamadı — silinemedi.")
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["_mgmt_flash"] = ("error", f"Silme hatası: {exc}")
                st.rerun()

# ========================= TAB 3 — AI analysis =============================
with tab3:
    st.markdown(
        '<div class="section-title"><span class="icon">🤖</span> Gemini AI Analiz</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Seçtiğiniz kapsamdaki işlemler/pozisyonlar, canlı teknik göstergeler ve "
        "sektör haberleri Gemini Flash modeline gönderilir; Türkçe koçluk raporu "
        "ve -10…+10 beklenen değişim puanı üretilir."
    )

    scope = st.radio(
        "Analiz kapsamı",
        options=["Tüm Aktif Portföy", "Belirli Tarihteki İşlemler"],
        horizontal=True,
    )

    selected_date = None
    if scope == "Belirli Tarihteki İşlemler":
        selected_date = st.date_input(
            "Analiz edilecek tarih",
            value=date.today(),
            help="Dünü veya geçmiş herhangi bir günü seçebilirsiniz.",
        )

    b1, b2, _ = st.columns([2, 1, 3])
    with b1:
        analyze_btn = st.button(
            "Analiz Et & Yorumla", type="primary", use_container_width=True
        )
    with b2:
        clear_btn = st.button("Raporu Temizle", use_container_width=True)

    if clear_btn and "ai_report" in st.session_state:
        for key in (
            "ai_report",
            "ai_report_scope",
            "ai_report_tickers",
            "ai_report_date",
            "ai_report_metrics",
            "ai_report_news",
            "ai_report_trades",
            "ai_score",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    if analyze_btn:
        if scope == "Tüm Aktif Portföy":
            holdings = db.get_current_holdings()
            if holdings.empty:
                st.warning("Analiz için en az bir açık pozisyon gerekli (önce BUY ekleyin).")
            else:
                tickers = holdings["ticker"].tolist()
                with st.spinner(
                    "Piyasa verileri + haberler alınıyor ve Gemini analiz ediyor..."
                ):
                    market_dict = fetch_market_dict(tickers)
                    news_dict = fetch_news_dict(tickers)
                    report = ai_analyst.analyze_portfolio(
                        holdings, market_dict, news_dict=news_dict
                    )
                st.session_state["ai_report"] = report
                st.session_state["ai_report_scope"] = "Tüm Aktif Portföy"
                st.session_state["ai_report_tickers"] = tickers
                st.session_state["ai_report_date"] = date.today().isoformat()
                st.session_state["ai_report_metrics"] = market_dict
                st.session_state["ai_report_news"] = news_dict
                st.session_state["ai_report_trades"] = []
                st.session_state["ai_score"] = ai_analyst.parse_expected_score(report)
        else:
            iso = selected_date.isoformat() if selected_date else date.today().isoformat()
            day_trades = db.get_trades_by_date(iso)
            if day_trades.empty:
                st.warning(f"{iso} tarihinde kayıtlı işlem yok. Farklı bir tarih seçin.")
            else:
                tickers = day_trades["ticker"].dropna().unique().tolist()
                with st.spinner(
                    "Piyasa verileri + haberler alınıyor ve Gemini analiz ediyor..."
                ):
                    market_dict = fetch_market_dict(tickers)
                    news_dict = fetch_news_dict(tickers)
                    report = ai_analyst.analyze_daily_performance(
                        day_trades,
                        market_dict,
                        analysis_date=iso,
                        news_dict=news_dict,
                    )
                st.session_state["ai_report"] = report
                st.session_state["ai_report_scope"] = f"{iso} işlemleri"
                st.session_state["ai_report_tickers"] = tickers
                st.session_state["ai_report_date"] = iso
                st.session_state["ai_report_metrics"] = market_dict
                st.session_state["ai_report_news"] = news_dict
                st.session_state["ai_report_trades"] = day_trades.to_dict(orient="records")
                st.session_state["ai_score"] = ai_analyst.parse_expected_score(report)

    if "ai_report" in st.session_state:
        scope_label = st.session_state.get("ai_report_scope", "")
        if scope_label:
            st.caption(f"Kapsam: {scope_label}")

        required_tickers = st.session_state.get("ai_report_tickers") or []
        missing = ai_analyst.report_missing_tickers(
            st.session_state["ai_report"], required_tickers
        )
        if missing:
            st.warning(
                "Rapor kapsam uyarısı — şu hisseler metinde net geçmiyor olabilir: "
                + ", ".join(missing)
                + ". Tekrar analiz etmeyi deneyin."
            )
        elif required_tickers:
            st.caption(
                f"Kapsam kontrolü: {len(required_tickers)} hisse raporda izlendi "
                f"({', '.join(required_tickers)})."
            )

        score_info = st.session_state.get("ai_score") or ai_analyst.parse_expected_score(
            st.session_state["ai_report"]
        )
        render_score_banner(score_info)

        st.markdown('<div class="ai-wrap">', unsafe_allow_html=True)
        render_ai_report(st.session_state["ai_report"])
        st.markdown("</div>", unsafe_allow_html=True)

        # --- PDF download ---
        tickers = st.session_state.get("ai_report_tickers") or []
        report_date = st.session_state.get("ai_report_date") or date.today().isoformat()
        primary_ticker = (
            "_".join(tickers[:3]) if tickers else "PORTFOY"
        )
        try:
            pdf_bytes = pdf_generator.generate_pdf_report(
                ticker=primary_ticker,
                report_date=report_date,
                metrics=st.session_state.get("ai_report_metrics") or {},
                ai_report_text=st.session_state["ai_report"],
                trades=st.session_state.get("ai_report_trades") or [],
                news=_flatten_news(st.session_state.get("ai_report_news") or {}),
                score_info=score_info,
                scope_label=scope_label,
            )
            st.download_button(
                label="📄 Raporu PDF Olarak İndir",
                data=pdf_bytes,
                file_name=pdf_generator.pdf_filename(primary_ticker, report_date),
                mime="application/pdf",
                use_container_width=False,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"PDF oluşturulamadı: {exc}")

        with st.expander("Ham Markdown çıktısını göster"):
            st.markdown(st.session_state["ai_report"])
    else:
        st.markdown(
            '<div class="empty-state">Kapsamı seçip <strong>Analiz Et &amp; Yorumla</strong> '
            "butonuna tıklayın.<br/>Sonuç; 📊 Değişim Puanı, 🎯 Destek/Direnç tablosu "
            "(tüm hisseler), 🔍 Hisse bazlı isabet etiketleri, 📰 Türkçe haber/sektör notları "
            "ve 💡 Net strateji + PDF indirme olarak gösterilir.</div>",
            unsafe_allow_html=True,
        )
