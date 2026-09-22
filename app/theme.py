"""SYNK design system for the Streamlit dashboard.

Implements the specified visual language: dark by default, calm and
trustworthy, monospaced technical values, restrained accent usage.  Danger
colors appear only where clinically relevant (risk categories).
"""

from __future__ import annotations

import streamlit as st

# ---- palette --------------------------------------------------------------
BG = "#09090B"
SURFACE = "#111113"
CARD = "#16161A"
PRIMARY = "#4F8CFF"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
TEXT = "#FAFAFA"
TEXT_SECONDARY = "#A1A1AA"
BORDER = "rgba(255,255,255,0.06)"

RISK_COLORS = {
    "LOW": SUCCESS,
    "MODERATE": WARNING,
    "HIGH": "#F97316",
    "CRITICAL": DANGER,
}

DISCLAIMER = "Research prototype — not for clinical diagnosis or treatment."


def apply_theme() -> None:
    st.set_page_config(
        page_title="SYNK — Clinical Intelligence",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --bg: {BG}; --surface: {SURFACE}; --card: {CARD};
            --primary: {PRIMARY}; --text: {TEXT}; --muted: {TEXT_SECONDARY};
            --border: {BORDER};
        }}
        .stApp {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', 'Geist', -apple-system, sans-serif;
        }}
        h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
        .mono, code, .stMetricValue {{ font-family: 'JetBrains Mono', monospace; }}

        [data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}

        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
        }}

        .synk-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
        }}
        .synk-hero {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 3rem 3rem;
            text-align: center;
            margin: 1.5rem 0;
        }}
        .synk-hero h1 {{
            font-size: 4.2rem; font-weight: 700; margin-bottom: 0.2rem;
            letter-spacing: 0.08em;
        }}
        .synk-tagline {{
            color: var(--muted); font-size: 1.15rem; margin-bottom: 0.4rem;
        }}
        .synk-sub {{
            color: var(--muted); max-width: 720px; margin: 0 auto 1.2rem auto; font-size: 1rem;
        }}
        .synk-disclaimer {{
            color: var(--muted); font-size: 0.82rem;
            border: 1px solid var(--border); border-radius: 10px;
            padding: 0.6rem 0.9rem; display: inline-block; margin-top: 1rem;
        }}
        .synk-status {{
            display: inline-flex; align-items: center; gap: 0.5rem;
            font-size: 0.85rem; color: var(--muted);
        }}
        .synk-dot {{
            width: 8px; height: 8px; border-radius: 50%;
            background: {SUCCESS}; display: inline-block;
            box-shadow: 0 0 6px {SUCCESS}55;
        }}
        .risk-pill {{
            display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
            font-weight: 600; font-size: 0.8rem; letter-spacing: 0.06em;
            border: 1px solid var(--border);
        }}
        .synk-metric {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 14px; padding: 1rem 1.2rem;
        }}
        .synk-metric .label {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }}
        .synk-metric .value {{ font-size: 1.6rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
        .synk-note {{
            background: var(--surface); border-left: 3px solid var(--primary);
            border-radius: 8px; padding: 0.8rem 1rem; color: var(--text); font-size: 0.92rem;
        }}
        .evidence-chip {{
            display: inline-block; padding: 0.15rem 0.55rem; margin: 0.1rem;
            border-radius: 999px; font-size: 0.75rem;
            background: {PRIMARY}22; color: {PRIMARY}; border: 1px solid {PRIMARY}44;
        }}
        .evidence-chip.negated {{
            background: {BORDER}; color: var(--muted); border-color: var(--border);
            text-decoration: line-through;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Apply the matching plotly/chart template globally via matplotlib defaults.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": CARD,
        "axes.facecolor": CARD,
        "savefig.facecolor": CARD,
        "text.color": TEXT,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "grid.color": "rgba(255,255,255,0.08)",
        "font.family": "Inter",
    })


def risk_pill(category: str) -> str:
    color = RISK_COLORS.get(category, TEXT_SECONDARY)
    return f'<span class="risk-pill" style="color:{color}; border-color:{color}55; background:{color}18;">{category}</span>'


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div style="color:{TEXT_SECONDARY}; margin-top:-0.5rem; margin-bottom:1rem;">{subtitle}</div>',
                    unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    color_style = f"color:{color};" if color else ""
    sub_html = f'<div style="color:{TEXT_SECONDARY}; font-size:0.78rem; margin-top:0.25rem;">{sub}</div>' if sub else ""
    return (
        f'<div class="synk-metric"><div class="label">{label}</div>'
        f'<div class="value" style="{color_style}">{value}</div>{sub_html}</div>'
    )


def disclaimer_banner() -> None:
    st.markdown(
        f'<div class="synk-disclaimer">⚠ {DISCLAIMER} All displayed data is synthetic demonstration data.</div>',
        unsafe_allow_html=True,
    )


def status_dot(online: bool = True, label: str = "System operational") -> str:
    color = SUCCESS if online else DANGER
    return (
        f'<span class="synk-status"><span class="synk-dot" style="background:{color}; '
        f'box-shadow:0 0 6px {color}55;"></span>{label}</span>'
    )


@st.cache_data(ttl=120, show_spinner=False)
def plotly_layout():
    import plotly.graph_objects as go

    return {
        "paper_bgcolor": CARD,
        "plot_bgcolor": CARD,
        "font": {"color": TEXT, "family": "Inter"},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.15)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.15)"},
        "margin": {"l": 10, "r": 10, "t": 30, "b": 10},
    }
