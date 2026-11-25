"""
ui_kit.py — gaya & komponen tampilan MTRAX (mengikuti preview "Dashboard final").
Dipakai oleh app.py dan cost_avoidance.py.
"""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#062440"
BLUE = "#0D7FCC"
BLUE_L = "#5DBBEA"
BLUE_XL = "#B9E3F7"
TEXT = "#1F2A36"
MUTED = "#5A6B7A"
BORDER = "#DDE4EA"
RED = "#C2412F"
GREY = "#7D8C99"
ORANGE = "#C27A1B"
GREEN = "#1E6B45"

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=20, t=10, b=30),
    hoverlabel=dict(font_family="Inter, system-ui, sans-serif"),
)

STATUS_STYLE = {
    "Kompetitif": ("#E6F2EA", GREEN),
    "Perlu ditinjau": ("#FFF1DC", "#8A4B00"),
    "Tidak kompetitif": ("#FBE7E4", "#8A2A1C"),
}


def esc(v) -> str:
    return html.escape(str(v))


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"], .stMarkdown, button, input, select, textarea { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #F2F5F8; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] * { visibility: visible !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DDE4EA; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* Tab utama: garis bawah biru */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #DDE4EA; background: transparent; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { padding: 10px 12px; font-size: 13px; color: #5A6B7A; }
.stTabs [aria-selected="true"] { color: #0A6AAD !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #0D7FCC !important; }

/* Tombol */
.stButton > button[kind="primary"], .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: #0D7FCC; border: 1px solid #0D7FCC; color: #FFFFFF; border-radius: 6px; font-weight: 600; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background: #0A6AAD; border-color: #0A6AAD; color: #FFFFFF; }
.stButton > button { border-radius: 6px; }

/* Kartu (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #DDE4EA !important; border-radius: 8px; }

/* Kartu KPI */
.mx-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(var(--mx-min, 180px), 1fr)); gap: 12px; margin: 2px 0 14px; }
.mx-kpi { background: #FFFFFF; border: 1px solid #DDE4EA; border-radius: 8px; padding: 14px 16px; }
.mx-kpi.dark { background: #062440; border-color: #062440; color: #FFFFFF; }
.mx-kpi .l { font-size: 12px; color: #5A6B7A; }
.mx-kpi.dark .l, .mx-kpi.dark .s { color: #C9DCEB; }
.mx-kpi .v { font-size: 24px; font-weight: 700; margin-top: 6px; line-height: 1.2; color: #1F2A36; }
.mx-kpi.dark .v { color: #FFFFFF; }
.mx-kpi .s { font-size: 12px; color: #5A6B7A; margin-top: 2px; }
.mx-mini { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 0 0 14px; }
.mx-mini div { background: #FFFFFF; border: 1px solid #DDE4EA; border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; font-size: 13px; }
.mx-mini span { color: #5A6B7A; }

/* Judul */
.mx-h1 { font-size: 24px; font-weight: 700; color: #062440; margin: 0; }
.mx-sub { font-size: 13px; color: #5A6B7A; margin: 2px 0 0; }
.mx-h2 { font-size: 16px; font-weight: 700; color: #062440; margin: 0; }
.mx-h2sub { font-size: 13px; color: #5A6B7A; margin: 4px 0 8px; }
.mx-page { font-size: 20px; font-weight: 700; color: #062440; margin: 6px 0 0; }

/* Lain-lain */
.mx-note { background: #F7FAFC; border: 1px solid #DDE4EA; border-radius: 8px; padding: 12px 14px; font-size: 13px; line-height: 1.6; color: #3A4A58; }
.mx-warn { background: #FFF8EC; border: 1px solid #F1D29A; border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.5; color: #7A4A00; }
.mx-badge { border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 600; display: inline-block; }
.mx-userbox { border: 1px solid #DDE4EA; border-radius: 8px; padding: 12px; margin: 6px 0 12px; }
.mx-userbox .n { font-size: 14px; font-weight: 600; color: #1F2A36; }
.mx-userbox .r { font-size: 12px; color: #5A6B7A; }
.mx-userbox .t { font-size: 12px; margin-top: 8px; }
.mx-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.mx-brand .m { width: 36px; height: 36px; border-radius: 8px; background: #0D7FCC; color: #FFFFFF; font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; }
.mx-brand .t { font-weight: 700; letter-spacing: .12em; font-size: 14px; color: #062440; }
.mx-src { font-size: 12px; line-height: 1.55; color: #3A4A58; border-top: 1px solid #EEF1F4; padding-top: 12px; margin-top: 6px; }
.mx-cluster { background: #FFFFFF; border: 1px solid #DDE4EA; border-radius: 8px; padding: 14px 16px; font-size: 13px; display: flex; flex-direction: column; gap: 6px; }
.mx-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.mx-table th { font-weight: 500; color: #5A6B7A; text-align: right; padding: 7px 8px; border-bottom: 1px solid #DDE4EA; white-space: nowrap; }
.mx-table th:first-child, .mx-table td:first-child { text-align: left; }
.mx-table td { padding: 7px 8px; border-bottom: 1px solid #EEF1F4; text-align: right; white-space: nowrap; }
</style>""", unsafe_allow_html=True)


def kpi(label, value, sub="", dark=False) -> str:
    return (f'<div class="mx-kpi{" dark" if dark else ""}"><div class="l">{esc(label)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>')


def kpi_row(items, min_width=180):
    st.markdown(f'<div class="mx-kpis" style="--mx-min:{min_width}px">{"".join(items)}</div>', unsafe_allow_html=True)


def mini_row(pairs):
    cells = "".join(f'<div><span>{esc(a)}</span><strong>{esc(b)}</strong></div>' for a, b in pairs)
    st.markdown(f'<div class="mx-mini">{cells}</div>', unsafe_allow_html=True)


def page_title(title, sub=""):
    st.markdown(f'<div class="mx-page">{esc(title)}</div>' + (f'<div class="mx-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="mx-h2">{esc(title)}</div>' + (f'<div class="mx-h2sub">{esc(sub)}</div>' if sub else
                '<div style="height:8px"></div>'), unsafe_allow_html=True)


def note(text_html, warn=False):
    st.markdown(f'<div class="{"mx-warn" if warn else "mx-note"}">{text_html}</div>', unsafe_allow_html=True)


def badge(status) -> str:
    bg, fg = STATUS_STYLE.get(status, ("#EEF1F4", MUTED))
    return f'<span class="mx-badge" style="background:{bg};color:{fg}">{esc(status)}</span>'


def choice(label, options, key, default=None):
    """Pilihan berbentuk pill (segmented control), fallback ke radio horizontal."""
    default = default if default is not None else options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key)
        return v if v is not None else default
    return st.radio(label, options, index=options.index(default), horizontal=True, key=key)


def fmt_int(v) -> str:
    try:
        return f"{float(v):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def fmt_rp(v) -> str:
    try:
        return "Rp" + f"{float(v):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def fmt_big(v) -> str:
    """Rp 1,23 M / Rp 456 jt / Rp 12.345"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if abs(v) >= 1e9:
        return "Rp" + f"{v / 1e9:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " M"
    if abs(v) >= 1e6:
        return "Rp" + f"{v / 1e6:,.0f}".replace(",", ".") + " jt"
    return fmt_rp(v)


def fmt_pct(v, signed=False, digits=1) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    s = f"{abs(v):.{digits}f}".replace(".", ",") + "%"
    if signed:
        return ("+" if v > 0 else "−" if v < 0 else "") + s
    return s
