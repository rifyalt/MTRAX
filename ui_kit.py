"""
ui_kit.py — design system MTRAX (dipakai app.py dan cost_avoidance.py).

Warna dipakai menurut FUNGSI, bukan dekorasi:
  Primary   #1BA0E2 data utama · #1494C6 sekunder · #0D7FCC aksi/penekanan
  Neutral   #F0F0F0 latar · #DEDEDE garis · #434343 teks
  Accent    #98EA16 pembanding/positif · #E2F871 sorotan lembut · #FFDC00 peringatan
            #FF5E1F perhatian · #E13230 negatif (isi grafik) · #D61B54 negatif (teks)
Untuk TEKS di atas putih dipakai turunan yang lebih gelap bila warna aslinya tidak memenuhi
kontras WCAG AA (4,5:1): POS_TEXT, WARN_TEXT, HILITE_DARK.
"""
from __future__ import annotations

import base64
import html
import os

import streamlit as st

# ---------------------------------------------------------------- brand tokens
P1 = "#1BA0E2"          # primary — seri data utama
P2 = "#1494C6"          # primary — seri kedua / hover
P3 = "#0D7FCC"          # primary — aksi (tombol, tab aktif, link)
BG = "#F0F0F0"
LINE = "#DEDEDE"
INK = "#434343"
LIME = "#98EA16"        # pembanding / kategori kedua
LIME_SOFT = "#E2F871"
YELLOW = "#FFDC00"
ORANGE_A = "#FF5E1F"
RED_FILL = "#E13230"
CRIMSON = "#D61B54"

# turunan untuk teks & permukaan gelap (kontras AA)
ACTION = "#0C79C2"       # #0D7FCC yang sedikit digelapkan: teks putih di tombol & teks tab aktif 4,6:1
HILITE_DARK = "#0A5D96"  # kartu KPI utama (teks putih 7:1)
POS_TEXT = "#3B7A0B"     # teks positif (5,3:1 di atas putih)
NEG_TEXT = CRIMSON       # teks negatif (5,0:1)
WARN_TEXT = "#A3420C"    # teks perhatian (5,6:1)
MUTED = "#6B6B6B"        # teks sekunder (5,3:1)
TINT = "#D6EEFA"         # isi lembut primary (batang "sisanya")
SURFACE = "#FFFFFF"

# nama lama yang masih dipakai modul lain
NAVY = HILITE_DARK
BLUE = P1
BLUE_D = P3
BLUE_L = P2
BLUE_XL = TINT
TEAL = P2
LIME_L = LIME_SOFT
TEXT = INK
BORDER = LINE
RED = RED_FILL
GREY = "#9A9A9A"
ORANGE = LIME            # kategori kedua (mis. Non-Patra Jasa)
GREEN = POS_TEXT

CATEGORICAL = [P1, LIME, P3, ORANGE_A, YELLOW, CRIMSON, P2, "#7A7A7A"]
SEQUENTIAL = [[0, "#F5FAFD"], [0.25, TINT], [0.55, P1], [0.8, P3], [1, HILITE_DARK]]
GRADIENT = f"linear-gradient(90deg, {LIME_SOFT} 0%, {P1} 60%, {P3} 100%)"

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, colorway=CATEGORICAL,
    margin=dict(l=10, r=20, t=10, b=30),
    hoverlabel=dict(font_family="Inter, system-ui, sans-serif", font_size=13, bgcolor=SURFACE,
                    bordercolor=LINE),
    xaxis=dict(gridcolor="#EDEDED", linecolor=LINE, zerolinecolor=LINE, tickfont=dict(size=12, color=MUTED)),
    yaxis=dict(gridcolor="#EDEDED", linecolor=LINE, zerolinecolor=LINE, tickfont=dict(size=12, color=MUTED)),
    legend=dict(font=dict(size=12, color=INK)),
)

STATUS_STYLE = {
    "Kompetitif": ("#EEF9DC", POS_TEXT),
    "Perlu ditinjau": ("#FFF6CC", "#6B5800"),
    "Tidak kompetitif": ("#FBE3EA", NEG_TEXT),
}


def esc(v) -> str:
    return html.escape(str(v))


def _plotly_template():
    """Template Plotly 'mtrax' supaya grafik yang belum memakai PLOTLY_LAYOUT pun konsisten."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        if "mtrax" not in pio.templates:
            t = go.layout.Template(layout=go.Layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}))
            # label donut di luar irisan (teks gelap di atas putih, kontras aman)
            t.data.pie = [go.Pie(textposition="outside", textfont=dict(color=INK, size=13),
                                 marker=dict(line=dict(color=SURFACE, width=2)))]
            pio.templates["mtrax"] = t
            pio.templates.default = "plotly_white+mtrax"
    except Exception:
        pass


def inject_css():
    _plotly_template()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ --p1:{P1}; --p2:{P2}; --p3:{P3}; --bg:{BG}; --line:{LINE}; --ink:{INK}; --muted:{MUTED};
        --lime:{LIME}; --lime-soft:{LIME_SOFT}; --action:{ACTION}; --yellow:{YELLOW}; --dark:{HILITE_DARK};
        --r:12px; --shadow:0 1px 2px rgba(20,20,20,.04), 0 1px 3px rgba(20,20,20,.06);
        --shadow-h:0 4px 12px rgba(20,20,20,.08); }}
html, body, .stApp, [class*="css"], .stMarkdown, button, input, select, textarea {{
  font-family:'Inter', system-ui, sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family:'Material Symbols Rounded' !important; }}
.stApp {{ background:var(--bg); color:var(--ink); }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] * {{ visibility:visible !important; }}
.block-container {{ padding-top:1.75rem; padding-bottom:3rem; max-width:1440px; }}
.stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {{ font-size:13px; }}
[data-testid="stCaptionContainer"] {{ color:var(--muted); }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.25rem; }}

/* Tab: satu warna aksi, garis 2px */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--line); background:transparent; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ padding:10px 12px; font-size:14px; color:var(--muted); transition:color .15s; }}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--ink); }}
.stTabs [aria-selected="true"] {{ color:var(--action) !important; font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:var(--p3) !important; height:2px !important; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* Tombol */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{
  border-radius:8px; font-weight:500; transition:background .15s, border-color .15s, box-shadow .15s, transform .15s; }}
.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
  background:var(--action); border:1px solid var(--action); color:#FFFFFF; font-weight:600; }}
.stButton > button[kind="primary"]:hover {{
  background:var(--dark); border-color:var(--dark); color:#FFFFFF; box-shadow:var(--shadow-h); }}
/* Unduh: tombol sekunder bergaris, ukuran ringkas, selalu berjarak dari elemen di atasnya */
.stDownloadButton {{ margin-top:4px; }}
.stDownloadButton > button {{ background:#FFFFFF; border:1px solid var(--line); color:var(--action);
  font-weight:600; font-size:13.5px; padding:6px 14px; min-height:36px; }}
.stDownloadButton > button:hover {{ background:#F3F9FD; border-color:var(--action); color:var(--action); box-shadow:none; }}
.stDownloadButton > button p {{ color:var(--action); }}
.stButton > button[kind="secondary"]:hover {{ border-color:var(--p3); color:var(--p3); }}

/* Kartu */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF; border-color:var(--line) !important;
  border-radius:var(--r); box-shadow:var(--shadow); }}
div[data-testid="stExpander"] details {{ border-color:var(--line); border-radius:var(--r); background:#FFFFFF; }}
div[data-testid="stExpander"] summary {{ font-weight:500; }}
[data-testid="stDataFrame"] {{ border-radius:8px; }}

/* KPI */
.mx-kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(var(--mx-min, 180px), 1fr)); gap:12px; margin:4px 0 16px; }}
.mx-kpi {{ background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); padding:16px 18px;
  box-shadow:var(--shadow); transition:box-shadow .15s, transform .15s; }}
.mx-kpi:hover {{ box-shadow:var(--shadow-h); transform:translateY(-1px); }}
.mx-kpi.dark {{ background:var(--dark); border-color:var(--dark); color:#FFFFFF; position:relative; overflow:hidden; }}
.mx-kpi.dark::before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:{GRADIENT}; }}
.mx-kpi .l {{ font-size:12.5px; font-weight:500; color:var(--muted); }}
.mx-kpi.dark .l, .mx-kpi.dark .s {{ color:#D7E9F6; }}
.mx-kpi .v {{ font-size:26px; font-weight:700; margin-top:6px; line-height:1.15; color:var(--ink); letter-spacing:-.01em;
  font-variant-numeric:tabular-nums; }}
.mx-kpi.dark .v {{ color:#FFFFFF; }}
.mx-kpi .s {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
.mx-mini {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:0 0 16px; }}
.mx-mini div {{ background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); padding:12px 16px;
  display:flex; justify-content:space-between; align-items:center; font-size:13.5px; box-shadow:var(--shadow); }}
.mx-mini span {{ color:var(--muted); }}
.mx-mini strong {{ color:var(--ink); font-variant-numeric:tabular-nums; }}

/* Judul */
.mx-h1 {{ font-size:26px; font-weight:700; color:var(--ink); margin:0; letter-spacing:-.015em; }}
.mx-h1::after {{ content:""; display:block; width:48px; height:3px; border-radius:2px; margin:10px 0 4px; background:{GRADIENT}; }}
.mx-sub {{ font-size:13.5px; color:var(--muted); margin:2px 0 0; }}
.mx-page {{ font-size:21px; font-weight:700; color:var(--ink); margin:8px 0 0; letter-spacing:-.01em; }}
.mx-h2 {{ font-size:16px; font-weight:600; color:var(--ink); margin:0; }}
.mx-h2sub {{ font-size:13px; color:var(--muted); margin:4px 0 10px; }}

/* Catatan */
.mx-note {{ background:#FFFFFF; border:1px solid var(--line); border-left:3px solid var(--p1); border-radius:8px;
  padding:12px 16px; font-size:13.5px; line-height:1.65; color:var(--ink); box-shadow:var(--shadow); margin-bottom:14px; }}
.mx-warn {{ background:#FFFBE0; border:1px solid #F2DC5D; border-left:3px solid var(--yellow); border-radius:8px;
  padding:10px 14px; font-size:13px; line-height:1.55; color:#5C4A00; margin-bottom:14px; }}
.mx-badge {{ border-radius:999px; padding:3px 10px; font-size:12.5px; font-weight:600; display:inline-block; }}

/* Sidebar komponen */
.mx-brand {{ display:flex; align-items:center; gap:10px; margin:2px 0 16px; }}
.mx-wordmark {{ height:30px; width:auto; display:block; }}
.mx-brand .m {{ width:38px; height:38px; border-radius:10px; background:#FFFFFF; border:1px solid var(--line);
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px; color:var(--p3); }}
.mx-brand .m img {{ width:24px; height:auto; }}
.mx-brand .t {{ font-weight:700; letter-spacing:.14em; font-size:14px; color:var(--ink); }}
.mx-userbox {{ border:1px solid var(--line); border-radius:var(--r); padding:12px 14px; margin:4px 0 14px; background:#FAFAFA; }}
.mx-userbox .n {{ font-size:14px; font-weight:600; color:var(--ink); }}
.mx-userbox .r {{ font-size:12.5px; color:var(--muted); }}
.mx-userbox .t {{ font-size:12.5px; margin-top:8px; display:flex; align-items:center; gap:6px; }}
.mx-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.mx-src {{ font-size:12.5px; line-height:1.6; color:var(--muted); border-top:1px solid var(--line); padding-top:12px; margin-top:8px; }}
.mx-src b {{ color:var(--ink); }}

/* Lain-lain */
.mx-cluster {{ background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); padding:14px 16px; font-size:13.5px;
  display:flex; flex-direction:column; gap:6px; box-shadow:var(--shadow); }}
.mx-table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
.mx-table th {{ font-weight:500; color:var(--muted); text-align:right; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
.mx-table th:first-child, .mx-table td:first-child {{ text-align:left; }}
.mx-table td {{ padding:8px 10px; border-bottom:1px solid #EDEDED; text-align:right; white-space:nowrap; }}
.mx-table tbody tr:hover td {{ background:#F7FBFE; }}
.mx-footer {{ text-align:center; padding:20px 0 4px; color:var(--muted); font-size:12.5px; }}
.mx-footer a {{ color:var(--p3); text-decoration:none; }}

@media (max-width: 760px) {{
  .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .mx-h1 {{ font-size:22px; }}
  .mx-kpi .v {{ font-size:22px; }}
}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- komponen
WORDMARK = "assets/mtrax_wordmark.png"


def img_b64(path) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def brand(logo_path=WORDMARK):
    """Wordmark MTRAX di sidebar (fallback: teks)."""
    b64 = img_b64(logo_path) if os.path.exists(logo_path) else ""
    mark = (f'<img src="data:image/png;base64,{b64}" alt="MTRAX" class="mx-wordmark"/>' if b64
            else '<div class="t">MTRAX</div>')
    st.markdown(f'<div class="mx-brand">{mark}</div>', unsafe_allow_html=True)


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
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="mx-h2">{esc(title)}</div>' + (f'<div class="mx-h2sub">{esc(sub)}</div>' if sub else
                '<div style="height:10px"></div>'), unsafe_allow_html=True)


def note(text_html, warn=False):
    st.markdown(f'<div class="{"mx-warn" if warn else "mx-note"}">{text_html}</div>', unsafe_allow_html=True)


def badge(status) -> str:
    bg, fg = STATUS_STYLE.get(status, ("#F0F0F0", MUTED))
    return f'<span class="mx-badge" style="background:{bg};color:{fg}">{esc(status)}</span>'


def status_line(text, tone="ok") -> str:
    """Baris status kecil dengan titik warna (pengganti emoji)."""
    color = {"ok": POS_TEXT, "warn": WARN_TEXT, "muted": MUTED}.get(tone, MUTED)
    dot = {"ok": LIME, "warn": ORANGE_A, "muted": "#BDBDBD"}.get(tone, "#BDBDBD")
    return f'<div class="t" style="color:{color}"><span class="mx-dot" style="background:{dot}"></span>{esc(text)}</div>'


def choice(label, options, key, default=None):
    default = default if default is not None else options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key)
        return v if v is not None else default
    return st.radio(label, options, index=options.index(default), key=key, horizontal=True)


def fmt_int(v) -> str:
    try:
        return f"{float(v):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def fmt_rp(v) -> str:
    try:
        if v != v:
            return "-"
        return "Rp" + f"{float(v):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


def fmt_big(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if v != v:
        return "-"
    if abs(v) >= 1e12:
        return "Rp" + f"{v / 1e12:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " T"
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
    if v != v:
        return "-"
    s = f"{abs(v):.{digits}f}".replace(".", ",") + "%"
    if signed:
        return ("+" if v > 0 else "−" if v < 0 else "") + s
    return s
