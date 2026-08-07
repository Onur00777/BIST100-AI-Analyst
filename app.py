"""
BIST 100 Daily AI Analyst — Streamlit main dashboard.

Modern dark financial terminal UI (TradingView / Bloomberg inspired).
"""

from __future__ import annotations

import html
import re
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ai_analyst
import database as db
import market_data as md

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
# Custom CSS — TradingView / Bloomberg dark terminal
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

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

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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

/* Sidebar expand chevron inside open sidebar */
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="headerNoPadding"] {
    visibility: visible !important;
    opacity: 1 !important;
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
.section-title span.icon {
    font-size: 1.05rem;
}

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
    font-weight: 800;
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
    font-weight: 800;
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
.stButton > button {
    background: linear-gradient(135deg, #2962ff, #1e88e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover {
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
.ai-card.card-motivation { border-left-color: #ab47bc; }
.ai-card.card-default { border-left-color: #787b86; }

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

.empty-state {
    background: var(--bg-panel);
    border: 1px dashed var(--border-soft);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: var(--text-secondary);
}

.toolbar {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.75rem;
}

@media (max-width: 900px) {
    .kpi-row { grid-template-columns: 1fr; }
    .tech-grid { grid-template-columns: repeat(2, 1fr); }
    .app-header { flex-direction: column; align-items: flex-start; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
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


def render_sidebar_toggle_button() -> None:
    """
    Reliable sidebar toggle: HTML button lives in the component iframe and
    clicks Streamlit's native expand/collapse controls in the parent document.
    (st.button + post-rerun JS injection is unreliable.)
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
            // Aria-label fallback (locale-agnostic-ish)
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
            if (btn) {
              btn.click();
              return;
            }
            // Last resort: keyboard shortcut Streamlit uses for sidebar ([)
            try {
              doc.dispatchEvent(new KeyboardEvent('keydown', {
                key: '[', code: 'BracketLeft', keyCode: 219, which: 219, bubbles: true
              }));
            } catch (e) {}
          }

          const button = document.getElementById('bist-sidebar-toggle');
          button.addEventListener('click', function (e) {
            e.preventDefault();
            toggleSidebar();
          });
        })();
        </script>
        """,
        height=48,
    )


def style_trades_table(df: pd.DataFrame):
    """Apply green/red action coloring to the trades dataframe."""
    def color_action(val):
        v = str(val).upper()
        if v == "BUY":
            return "color: #26a69a; font-weight: 700; background-color: rgba(38,166,154,0.12);"
        if v == "SELL":
            return "color: #ef5350; font-weight: 700; background-color: rgba(239,83,80,0.12);"
        return ""

    styler = (
        df.style
        .map(color_action, subset=["İşlem"] if "İşlem" in df.columns else [])
        .format({"Fiyat": "{:,.2f}", "Adet": "{:g}"}, na_rep="—")
        .set_properties(**{
            "background-color": "#1e222d",
            "color": "#d1d4dc",
            "border-color": "#2a2e39",
        })
    )
    return styler


def render_kpi_row(trade_count: int, active_positions: int, bist: dict) -> None:
    """Render three Bloomberg-style KPI cards."""
    bist_ok = bist.get("success")
    delta = bist.get("daily_change_pct") if bist_ok else None
    trend_class = _change_class(delta)
    kpi_trend = {
        "pos": "kpi-up",
        "neg": "kpi-down",
        "flat": "kpi-flat",
    }.get(trend_class, "kpi-flat")

    if bist_ok:
        bist_value = _fmt_price(bist.get("current_price"))
        bist_sub = f'<div class="kpi-sub {trend_class}">{_fmt_pct(delta)} günlük</div>'
    else:
        bist_value = "Veri yok"
        bist_sub = f'<div class="kpi-sub flat">{html.escape(str(bist.get("error") or "Endeks alınamadı"))}</div>'

    html_block = f"""
    <div class="kpi-row">
      <div class="kpi-card kpi-blue">
        <div class="kpi-label">Bugünün İşlem Sayısı</div>
        <div class="kpi-value">{trade_count}</div>
        <div class="kpi-sub flat">bugün kaydedilen</div>
      </div>
      <div class="kpi-card kpi-teal">
        <div class="kpi-label">Aktif Pozisyonlar</div>
        <div class="kpi-value">{active_positions}</div>
        <div class="kpi-sub flat">açık net lot</div>
      </div>
      <div class="kpi-card {kpi_trend}">
        <div class="kpi-label">BIST 100 Trend</div>
        <div class="kpi-value">{bist_value}</div>
        {bist_sub}
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def render_tech_card(summary: dict) -> None:
    """Render a technical indicator card for one ticker."""
    if not summary.get("success"):
        st.warning(f"**{summary.get('ticker', '?')}**: {summary.get('error', 'Veri yok')}")
        return

    change = summary.get("daily_change_pct")
    rsi = summary.get("rsi_14")
    chip_cls = _change_class(change)

    card = f"""
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
    """
    st.markdown(card, unsafe_allow_html=True)


def _inline_md(text: str) -> str:
    """Minimal inline markdown → HTML (bold / code / escape)."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _body_to_html(body: str) -> str:
    """Convert a markdown body chunk into simple HTML lists/paragraphs."""
    lines = body.strip().splitlines()
    parts: list[str] = []
    in_list = False

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                parts.append("</ul>")
                in_list = False
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
            if in_list:
                parts.append("</ul>")
                in_list = False
            # skip nested markdown headers inside body
            cleaned = re.sub(r"^#{1,6}\s*", "", line.strip())
            parts.append(f"<p>{_inline_md(cleaned)}</p>")

    if in_list:
        parts.append("</ul>")
    return "\n".join(parts) if parts else f"<p>{_inline_md(body)}</p>"


def _classify_section(title: str) -> tuple[str, str]:
    """Map section title → (css_class, icon)."""
    t = title.lower()
    if any(k in t for k in ("özet", "ozet", "summary", "günlük")):
        return "card-summary", "📋"
    if any(k in t for k in ("disiplin", "discipline")):
        return "card-discipline", "✅"
    if any(k in t for k in ("teknik", "işlem", "islem", "yorum")):
        return "card-technical", "📈"
    if any(k in t for k in ("risk", "konsantrasyon", "concentration")):
        return "card-risk", "⚠️"
    if any(k in t for k in ("destek", "direnç", "direnc", "seviye", "yarın", "yarin")):
        return "card-levels", "🎯"
    if any(k in t for k in ("motivasyon", "motivation", "cümle", "cumle")):
        return "card-motivation", "💡"
    return "card-default", "📌"


def render_ai_report(report: str) -> None:
    """
    Parse Gemini Markdown into structured alert cards.
    Falls back to a single panel if no ## headers are found.
    """
    if not report or not report.strip():
        st.info("Rapor boş geldi.")
        return

    # Split on markdown H2/H1 headers
    chunks = re.split(r"(?m)^(#{1,3})\s+(.+)$", report.strip())

    # re.split with groups → [preamble, hashes, title, body, hashes, title, body, ...]
    if len(chunks) < 4:
        # No headers — wrap whole report
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

    # Iterate triples: (hashes, title, body)
    i = 1
    while i + 2 < len(chunks):
        title = chunks[i + 1].strip()
        body = chunks[i + 2].strip()
        css_cls, icon = _classify_section(title)
        # Prefer domain icons over generic when title already has emoji-ish meaning
        display_title = title
        st.markdown(
            f"""
            <div class="ai-card {css_cls}">
              <div class="ai-card-title">{icon} {html.escape(display_title)}</div>
              <div class="ai-card-body">{_body_to_html(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        i += 3


@st.cache_data(ttl=300, show_spinner=False)
def cached_stock_summary(ticker: str) -> dict:
    """Cache market summaries for 5 minutes to reduce Yahoo rate limits."""
    return md.get_stock_summary(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_bist100() -> dict:
    """Cache BIST 100 index snapshot for 5 minutes."""
    return md.get_bist100_index_summary()


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
                "Hisse Kodu",
                placeholder="örn. THYAO",
                help="`.IS` eklemenize gerek yok — otomatik eklenir.",
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
            clean_ticker = (ticker_input or "").strip().upper().replace(".IS", "")
            if not clean_ticker:
                st.error("Lütfen geçerli bir hisse kodu girin.")
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
                    st.cache_data.clear()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Kayıt hatası: {exc}")

    st.markdown("---")
    st.markdown("### 🗓️ Bugünün İşlemleri")

    todays = db.get_todays_trades()
    if todays.empty:
        st.caption("Bugün henüz işlem yok.")
    else:
        for _, row in todays.iterrows():
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
            if st.button("Sil", key=f"del_{row['id']}", help="Bu işlemi sil"):
                db.delete_trade(int(row["id"]))
                st.rerun()

    st.markdown("---")
    holdings = db.get_current_holdings()
    st.markdown("### 📦 Açık Pozisyonlar")
    if holdings.empty:
        st.caption("Aktif pozisyon yok.")
    else:
        st.dataframe(
            holdings.rename(
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
# Main page
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([5, 1])
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
with top_r:
    st.write("")  # vertical align with header
    st.caption("Sol paneli aç / kapat")
    render_sidebar_toggle_button()

# --- Summary KPIs ---
todays_trades = db.get_todays_trades()
active_positions = db.count_active_positions()
bist = cached_bist100()
render_kpi_row(len(todays_trades), active_positions, bist)

# --- Tabs ---
tab1, tab2 = st.tabs(
    [
        "📊 Bugünün İşlemleri & Teknik Durum",
        "🤖 AI Analiz Raporu",
    ]
)

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
        display_df = todays_trades.copy().rename(
            columns={
                "id": "ID",
                "date": "Tarih",
                "ticker": "Hisse",
                "action": "İşlem",
                "quantity": "Adet",
                "price": "Fiyat",
                "notes": "Not",
            }
        )
        try:
            st.dataframe(
                style_trades_table(display_df),
                use_container_width=True,
                hide_index=True,
            )
        except Exception:  # noqa: BLE001 — Styler edge cases on some pandas builds
            st.dataframe(display_df, use_container_width=True, hide_index=True)

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
            summaries = {t: cached_stock_summary(t) for t in tickers_to_show}

        cols = st.columns(2)
        for i, ticker in enumerate(tickers_to_show):
            with cols[i % 2]:
                render_tech_card(summaries[ticker])

with tab2:
    st.markdown(
        '<div class="section-title"><span class="icon">🤖</span> Gemini AI Günlük Analiz</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Bugünkü işlemleriniz ve teknik göstergeler Gemini Flash modeline "
        "gönderilir; Türkçe koçluk raporu üretilir."
    )

    b1, b2, _ = st.columns([2, 1, 3])
    with b1:
        analyze_btn = st.button("Günü Analiz Et & Yorumla", type="primary", use_container_width=True)
    with b2:
        clear_btn = st.button("Raporu Temizle", use_container_width=True)

    if clear_btn and "ai_report" in st.session_state:
        del st.session_state["ai_report"]
        st.session_state.pop("ai_report_tickers", None)
        st.rerun()

    if analyze_btn:
        if todays_trades.empty:
            st.warning("Analiz için bugün en az bir işlem kaydı gerekli.")
        else:
            tickers = todays_trades["ticker"].dropna().unique().tolist()
            with st.spinner("Piyasa verileri alınıyor ve Gemini analiz ediyor..."):
                market_dict = {t: cached_stock_summary(t) for t in tickers}
                report = ai_analyst.analyze_daily_performance(todays_trades, market_dict)
            st.session_state["ai_report"] = report
            st.session_state["ai_report_tickers"] = tickers

    if "ai_report" in st.session_state:
        st.markdown('<div class="ai-wrap">', unsafe_allow_html=True)
        render_ai_report(st.session_state["ai_report"])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Ham Markdown çıktısını göster"):
            st.markdown(st.session_state["ai_report"])
    else:
        st.markdown(
            '<div class="empty-state">Rapor üretmek için <strong>Günü Analiz Et &amp; Yorumla</strong> '
            "butonuna tıklayın.<br/>Sonuç; Destek/Direnç, Risk ve Disiplin kartları olarak gösterilir.</div>",
            unsafe_allow_html=True,
        )
