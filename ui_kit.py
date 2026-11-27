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
import functools
import html
import os

import streamlit as st

UI_VERSION = "2026.11.27-m"   # tampil di footer; cek ini bila tampilan terasa belum berubah

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
GRADIENT_V = f"linear-gradient(180deg, {LIME_SOFT} 0%, {P1} 55%, {P3} 100%)"

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, colorway=CATEGORICAL,
    margin=dict(l=10, r=20, t=10, b=30),
    hoverlabel=dict(font_family="Inter, system-ui, sans-serif", font_size=13, bgcolor=SURFACE,
                    bordercolor=LINE),
    xaxis=dict(gridcolor="#EDEDED", linecolor=LINE, zerolinecolor=LINE, tickfont=dict(size=12, color=MUTED)),
    yaxis=dict(gridcolor="#EDEDED", linecolor=LINE, zerolinecolor=LINE, tickfont=dict(size=12, color=MUTED)),
    legend=dict(font=dict(size=12, color=INK)),
    separators=",.",                      # format Indonesia: 1.234,5
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

/* Header (hero) */
.mx-hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:20px; flex-wrap:wrap;
  background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:22px 26px; margin:0 0 14px;
  box-shadow:var(--shadow); position:relative; overflow:hidden; }}
.mx-hero::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:{GRADIENT_V}; }}
.mx-eyebrow {{ font-size:12px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--action); }}
.mx-hero-t {{ font-size:26px; font-weight:700; color:var(--ink); letter-spacing:-.015em; margin-top:4px; }}
.mx-hero-s {{ font-size:13.5px; color:var(--muted); margin-top:4px; max-width:620px; }}
.mx-hero-r {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.mx-chip {{ display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:999px; font-size:12.5px;
  font-weight:500; color:var(--ink); background:#F5F8FA; border:1px solid #E6EBEF; white-space:nowrap; }}
.mx-chip svg {{ width:14px; height:14px; fill:none; stroke:var(--action); stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }}

/* Navigasi halaman: bar pil putih. Hanya halaman aktif yang dihitung. */
.st-key-mx_nav {{ padding:4px 0 8px; margin-bottom:4px; }}
.st-key-mx_nav [role="radiogroup"] {{ gap:2px !important; flex-wrap:wrap; background:#FFFFFF; border:1px solid var(--line);
  border-radius:14px; padding:6px; box-shadow:var(--shadow); }}
.st-key-mx_nav label[data-testid="stRadioOption"], .st-key-mx_nav label[data-baseweb="radio"] {{
  padding:7px 11px !important; margin:0 !important; border-radius:10px; cursor:pointer;
  transition:background .15s, color .15s; }}
.st-key-mx_nav label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]),
.st-key-mx_nav label[data-baseweb="radio"] > div:first-child {{ display:none !important; }}
.st-key-mx_nav label p {{ font-size:13.5px; font-weight:500; color:var(--muted); margin:0; display:flex; align-items:center; gap:6px; }}
.st-key-mx_nav label [data-testid="stIconMaterial"] {{ font-size:17px !important; color:#9AA4AD; }}
.st-key-mx_nav label:hover {{ background:#F3F6F8; }}
.st-key-mx_nav label:hover p {{ color:var(--ink); }}
.st-key-mx_nav label[data-selected="true"], .st-key-mx_nav label:has(input:checked) {{ background:#E8F4FC; }}
.st-key-mx_nav label[data-selected="true"] p, .st-key-mx_nav label:has(input:checked) p {{ color:var(--action); font-weight:600; }}
.st-key-mx_nav label[data-selected="true"] [data-testid="stIconMaterial"],
.st-key-mx_nav label:has(input:checked) [data-testid="stIconMaterial"] {{ color:var(--action); }}

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
.mx-kpi {{ background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); padding:15px 14px 14px 16px;
  box-shadow:var(--shadow); transition:box-shadow .15s, transform .15s, border-color .15s;
  display:flex; flex-direction:column; min-width:0; }}
.mx-kpi:hover {{ box-shadow:var(--shadow-h); transform:translateY(-1px); border-color:#CFE6F5; }}
.mx-kpi.dark {{ background:var(--dark); border-color:var(--dark); color:#FFFFFF; position:relative; overflow:hidden; }}
.mx-kpi.dark::before {{ content:""; position:absolute; left:0; top:0; right:0; height:3px; background:{GRADIENT}; }}
.mx-kpi .hd {{ display:flex; align-items:center; justify-content:space-between; gap:6px; min-width:0; }}
.mx-kpi .ic {{ order:2; flex:none; width:26px; height:26px; border-radius:8px; background:#EAF5FC; display:flex;
  align-items:center; justify-content:center; }}
.mx-kpi .ic svg {{ width:15px; height:15px; fill:none; stroke:var(--p3); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
.mx-kpi .l {{ font-size:12.5px; font-weight:500; color:var(--muted); line-height:1.3; min-width:0; white-space:nowrap; }}
.mx-kpi.dark .l, .mx-kpi.dark .s {{ color:#D7E9F6; }}
.mx-kpi .v {{ font-size:25px; font-weight:700; margin-top:10px; line-height:1.15; color:var(--ink); letter-spacing:-.015em;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
.mx-kpi .v small {{ font-size:14px; font-weight:600; color:var(--muted); margin-left:3px; letter-spacing:0; }}
.mx-kpi.dark .v {{ color:#FFFFFF; }}
.mx-kpi .s {{ font-size:12px; color:var(--muted); margin-top:6px; min-height:16px; line-height:1.35; }}
.mx-kpi .s b {{ color:var(--ink); font-weight:600; }}
.mx-kpi .dl {{ display:flex; align-items:center; gap:6px; margin-top:7px; font-size:11.5px; color:var(--muted); white-space:nowrap; }}
.mx-kpi .dl .pill {{ display:inline-flex; align-items:center; gap:3px; padding:2px 7px 2px 6px; border-radius:999px;
  font-weight:700; font-size:11.5px; font-variant-numeric:tabular-nums; line-height:1.35; }}
.mx-kpi .dl .pill svg {{ width:10px; height:10px; fill:currentColor; }}
.mx-kpi .dl .good {{ background:#E7F6EC; color:#1B7F3B; }}
.mx-kpi .dl .bad {{ background:#FDECEC; color:#C0362C; }}
.mx-kpi .dl .flat {{ background:#F0F2F4; color:var(--muted); }}
/* Baris statistik ringkas: satu kartu, dibagi garis tipis */
.mx-mini {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); margin:0 0 18px;
  background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); box-shadow:var(--shadow); overflow:hidden; }}
.mx-mini > div {{ padding:13px 18px 12px; display:flex; flex-direction:column; gap:8px;
  box-shadow:-1px -1px 0 var(--line); min-width:0; }}
.mx-mini .bar.e {{ visibility:hidden; }}
.mx-mini .r {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; font-size:13px; }}
.mx-mini span {{ color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mx-mini strong {{ color:var(--ink); font-size:15px; font-weight:700; font-variant-numeric:tabular-nums; }}
.mx-mini .bar {{ height:4px; border-radius:2px; background:#EEF2F5; overflow:hidden; }}
.mx-mini .bar i {{ display:block; height:100%; border-radius:2px; background:{GRADIENT}; }}

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
.mx-guide {{ font-size:13.5px; line-height:1.7; color:var(--ink); }}
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
.mx-scope {{ background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); box-shadow:var(--shadow);
  margin:4px 0 16px; overflow:hidden; }}
.mx-scope .row {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); }}
.mx-scope .it {{ padding:14px 18px 13px; box-shadow:-1px -1px 0 var(--line); min-width:0; display:flex; flex-direction:column; }}
.mx-scope .hd {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
.mx-scope .l {{ font-size:12.5px; font-weight:500; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mx-scope .p {{ font-size:13px; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }}
.mx-scope .v {{ font-size:20px; font-weight:700; color:var(--ink); margin-top:8px; letter-spacing:-.01em;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
.mx-scope .v small {{ font-size:13px; font-weight:600; color:var(--muted); margin-left:2px; letter-spacing:0; }}
.mx-scope .s {{ font-size:12px; color:var(--muted); margin-top:5px; min-height:16px; }}
.mx-scope .bar {{ height:4px; border-radius:2px; background:#EEF2F5; overflow:hidden; margin-top:10px; }}
.mx-scope .bar i {{ display:block; height:100%; border-radius:2px; background:{GRADIENT}; }}
.mx-scope .ft {{ border-top:1px solid var(--line); padding:9px 18px; font-size:12.5px; color:var(--muted); background:#FAFBFC;
  display:flex; align-items:center; gap:8px; }}
.mx-scope .ft::before {{ content:"i"; flex:none; width:16px; height:16px; border-radius:50%; background:#EAF5FC; color:var(--p3);
  font-size:11px; font-weight:700; font-style:italic; display:flex; align-items:center; justify-content:center; font-family:Georgia,serif; }}
[data-testid="stPopover"] button {{ border-radius:8px; font-size:13.5px; }}
.mx-footer {{ text-align:center; padding:20px 0 4px; color:var(--muted); font-size:12.5px; }}
.mx-footer a {{ color:var(--p3); text-decoration:none; }}

@media (max-width: 760px) {{
  .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .mx-h1 {{ font-size:22px; }}
  .mx-kpi .v {{ font-size:22px; }}
}}

/* ---- kartu "data dimuat" beranimasi (angka menghitung naik, CSS murni) ---- */
@property --v {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@counter-style mxpad3 {{ system: extends decimal; pad: 3 "0"; }}
.mx-load {{ border: 1px solid #DEDEDE; border-radius: 10px; background: #fff; padding: 10px 12px; margin: 4px 0 16px;
  animation: mxLoadIn .35s ease-out both; }}
.mx-load .top {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #6B6B6B; }}
.mx-load .top i {{ width: 6px; height: 6px; border-radius: 50%; background: #3B7A0B; display: inline-block; }}
.mx-load .big {{ font-size: 20px; font-weight: 600; color: #434343; line-height: 1.2; margin-top: 4px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
.mx-load .big small {{ font-size: 12px; font-weight: 400; color: #6B6B6B; margin-left: 4px; }}
.mx-load .sub {{ font-size: 11px; color: #6B6B6B; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.mx-load .sub .sec {{ color: #3B7A0B; font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ margin-top: 2px; margin-bottom: 6px; }}
/* ---- kartu laporan (Reports) ---- */
.mx-rep-g {{ display:flex; align-items:baseline; gap:10px; margin:14px 0 2px; flex-wrap:wrap; }}
.mx-rep-g b {{ font-size:15px; font-weight:600; color:var(--ink); }}
.mx-rep-g span {{ font-size:12.5px; color:var(--muted); }}
.mx-rep .hd {{ display:flex; align-items:center; gap:10px; }}
.mx-rep .ic {{ width:34px; height:34px; border-radius:9px; background:#E9F5FC; display:inline-flex; align-items:center;
  justify-content:center; flex:0 0 34px; }}
.mx-rep .ic svg {{ width:18px; height:18px; fill:none; stroke:var(--action); stroke-width:1.8; stroke-linecap:round;
  stroke-linejoin:round; }}
.mx-rep .t {{ font-size:15px; font-weight:600; color:var(--ink); line-height:1.25; }}
.mx-rep .d {{ font-size:13px; color:var(--muted); line-height:1.45; margin-top:10px; min-height:38px;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
.mx-rep .m {{ display:flex; justify-content:space-between; font-size:11.5px; color:#8A8A8A; margin:10px 0 2px;
  padding-top:10px; border-top:1px solid #EFEFEF; font-variant-numeric:tabular-nums; }}
[class*="st-key-rep_"] {{ background:#fff; transition:border-color .15s, box-shadow .15s; }}
[class*="st-key-rep_"]:hover {{ border-color:#BFE3F6 !important; box-shadow:0 4px 14px rgba(13,127,204,.08); }}
/* ---- daftar kota di samping peta (Overview) ---- */
.mx-maplist {{ font-size:12.5px; color:var(--ink); padding-top:6px; }}
.mx-maplist > b {{ display:block; font-size:13px; font-weight:600; margin-bottom:6px; }}
.mx-maplist .r {{ display:grid; grid-template-columns:18px minmax(0,1fr) auto; align-items:center; column-gap:8px;
  row-gap:4px; padding:6px 0; border-bottom:1px solid #F3F3F3; }}
.mx-maplist .n {{ color:#9A9A9A; text-align:right; font-size:11.5px; }}
.mx-maplist .c {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.mx-maplist .b {{ grid-column:2 / 4; height:4px; background:#F0F4F7; border-radius:2px; overflow:hidden; }}
.mx-maplist .b i {{ display:block; height:100%; background:var(--p1, #1BA0E2); border-radius:3px; }}
.mx-maplist .v {{ text-align:right; font-weight:600; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.mx-maplist .ft {{ font-size:11.5px; color:#8A8A8A; margin-top:8px; line-height:1.4; }}
/* ---- grafik berdampingan: susun ke bawah bila layar sempit ---- */
@media (max-width: 1500px) {{ .mx-kpi .l {{ font-size:12px; }} .mx-kpi .ic {{ width:24px; height:24px; }} }}
@media (max-width: 1200px) {{
  [data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) {{ flex-wrap: wrap !important; }}
  [data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) > [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }}
}}
.mx-n {{ counter-reset: c var(--v); animation: mxCount 1.3s cubic-bezier(.15,.8,.25,1) both; }}
.mx-n::after {{ content: counter(c); }}
.mx-n.p3::after {{ content: counter(c, mxpad3); }}
@keyframes mxCount {{ from {{ --v: 0; }} }}
@keyframes mxBar {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes mxLoadIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes mxPop {{ from {{ transform: scale(0); }} to {{ transform: scale(1); }} }}
@keyframes mxDraw {{ to {{ stroke-dashoffset: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .mx-load, .mx-load *, .mx-n {{ animation: none !important; }} }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- komponen
_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "assets")
# nama file wordmark yang dikenali (urutan prioritas); file lain berisi kata "wordmark" juga dipakai
WORDMARK_NAMES = ("mtrax_wordmark.png", "MTRAX_Wordmark_Gradien_Polos.png")


def _find_wordmark() -> str:
    for n in WORDMARK_NAMES:
        p = os.path.join(_ASSETS, n)
        if os.path.exists(p):
            return p
    try:
        for n in sorted(os.listdir(_ASSETS)):
            if "wordmark" in n.lower() and n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                return os.path.join(_ASSETS, n)
    except Exception:
        pass
    return ""


WORDMARK = _find_wordmark()


def img_b64(path) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


@functools.lru_cache(maxsize=4)
def wordmark_b64(height: int = 96) -> str:
    """Wordmark yang sudah dirapikan: latar putih jadi transparan, tepi kosong dipotong, diperkecil.
    Hasil di-cache supaya gambar besar tidak dikirim berulang ke browser."""
    path = _find_wordmark()
    if not path:
        return ""
    try:
        from io import BytesIO
        import numpy as np
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        a = np.array(im)
        white = (a[:, :, 0] > 245) & (a[:, :, 1] > 245) & (a[:, :, 2] > 245)
        a[white, 3] = 0
        im = Image.fromarray(a)
        if im.getbbox():
            im = im.crop(im.getbbox())
        if im.height > height:
            im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return img_b64(path)


def brand():
    """Wordmark MTRAX di sidebar (fallback: teks)."""
    b64 = wordmark_b64()
    mark = (f'<img src="data:image/png;base64,{b64}" alt="MTRAX" class="mx-wordmark"/>' if b64
            else '<div class="t">MTRAX</div>')
    st.markdown(f'<div class="mx-brand">{mark}</div>', unsafe_allow_html=True)


_KPI_ICO = {
    "booking": '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M9 3v4M15 3v4M9 15l2 2 4-4"/>',
    "request": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
    "hotel": '<path d="M4 21V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v16M15 9h4a1 1 0 0 1 1 1v11M3 21h18M8 8h3M8 12h3M8 16h3"/>',
    "night": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "rate": '<circle cx="12" cy="12" r="8.5"/><path d="M14.8 9.2A2.6 2.6 0 0 0 12.4 8h-1a2 2 0 0 0 0 4h1.2a2 2 0 0 1 0 4h-1.1a2.6 2.6 0 0 1-2.4-1.3M12 6.5V8M12 16v1.5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
    "saving": '<path d="M19 7h-9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H5M12 3v18"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "scale": '<path d="M12 4v16M7 20h10M5 8h14M5 8l-2.5 6a3 3 0 0 0 5 0zM19 8l-2.5 6a3 3 0 0 0 5 0z"/>',
}


def delta_html(cur, prev, vs="", higher_is_good=True, tip="", eps=0.05) -> str:
    """Pil naik/turun vs periode pembanding. Kosong jika pembanding tidak ada."""
    def _na(v):
        try:
            return v is None or v != v
        except Exception:
            return True
    if _na(cur) or _na(prev) or prev == 0:
        return '<div class="dl"><span class="pill flat">–</span><span>tidak ada pembanding</span></div>'
    ch = (cur - prev) / abs(prev) * 100
    if abs(ch) < eps:
        cls, arrow = "flat", '<svg viewBox="0 0 10 10"><rect x="1" y="4" width="8" height="2" rx="1"/></svg>'
    else:
        up = ch > 0
        cls = "good" if up == higher_is_good else "bad"
        arrow = ('<svg viewBox="0 0 10 10"><path d="M5 1l4 6H1z"/></svg>' if up
                 else '<svg viewBox="0 0 10 10"><path d="M5 9L1 3h8z"/></svg>')
    txt = f"{abs(ch):.1f}".replace(".", ",") + "%"
    t = f' title="{esc(tip)}"' if tip else ""
    return f'<div class="dl"{t}><span class="pill {cls}">{arrow}{txt}</span><span>{esc(vs)}</span></div>'


def kpi(label, value, sub="", dark=False, icon=None, unit="", sub_html=False, delta="") -> str:
    ico = (f'<span class="ic"><svg viewBox="0 0 24 24">{_KPI_ICO[icon]}</svg></span>'
           if icon in _KPI_ICO else "")
    u = f"<small>{esc(unit)}</small>" if unit else ""
    sub_s = sub if sub_html else esc(sub)
    return (f'<div class="mx-kpi{" dark" if dark else ""}"><div class="hd">{ico}<div class="l">{esc(label)}</div></div>'
            f'<div class="v">{esc(value)}{u}</div>{delta}<div class="s">{sub_s}</div></div>')


def kpi_row(items, min_width=180):
    st.markdown(f'<div class="mx-kpis" style="--mx-min:{min_width}px">{"".join(items)}</div>', unsafe_allow_html=True)


def mini_row(pairs):
    """pairs: (label, nilai) atau (label, nilai, persen 0–100 untuk bar)."""
    cells = []
    for p in pairs:
        a, b = p[0], p[1]
        bar = (f'<div class="bar"><i style="width:{max(0, min(100, float(p[2]))):.1f}%"></i></div>'
               if len(p) > 2 and p[2] is not None else '<div class="bar e"></div>')
        cells.append(f'<div><div class="r"><span>{esc(a)}</span><strong>{esc(b)}</strong></div>{bar}</div>')
    st.markdown(f'<div class="mx-mini">{"".join(cells)}</div>', unsafe_allow_html=True)


def page_title(title, sub=""):
    st.markdown(f'<div class="mx-page">{esc(title)}</div>' + (f'<div class="mx-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="mx-h2">{esc(title)}</div>' + (f'<div class="mx-h2sub">{esc(sub)}</div>' if sub else
                '<div style="height:10px"></div>'), unsafe_allow_html=True)


def note(text_html, warn=False):
    st.markdown(f'<div class="{"mx-warn" if warn else "mx-note"}">{text_html}</div>', unsafe_allow_html=True)


def scope_bar(items, footnote_html=""):
    """Satu kartu cakupan data. items: (label, nilai, keterangan[, persen 0–100]).
    Nilai 'a / b' ditampilkan dengan b lebih kecil; persen tampil di kanan atas + bar tipis."""
    cells = []
    for it in items:
        a, b, c = it[0], str(it[1]), it[2]
        pv = it[3] if len(it) > 3 else None
        if " / " in b:
            x, y = b.split(" / ", 1)
            val = f"{esc(x)}<small>/ {esc(y)}</small>"
        else:
            val = esc(b)
        ptxt = f'<span class="p">{pv:.0f}%</span>' if pv is not None else ""
        bar = (f'<div class="bar"><i style="width:{max(0, min(100, pv)):.1f}%"></i></div>' if pv is not None else "")
        cells.append(f'<div class="it"><div class="hd"><span class="l">{esc(a)}</span>{ptxt}</div>'
                     f'<div class="v">{val}</div><div class="s">{esc(c)}</div>{bar}</div>')
    ft = f'<div class="ft"><span>{footnote_html}</span></div>' if footnote_html else ""
    st.markdown(f'<div class="mx-scope"><div class="row">{"".join(cells)}</div>{ft}</div>', unsafe_allow_html=True)


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


def _count_html(n: int) -> str:
    """Angka beranimasi (hitung naik) dengan pemisah ribuan titik, tanpa JavaScript."""
    n = max(int(n), 0)
    groups = []
    while True:
        groups.insert(0, n % 1000)
        n //= 1000
        if n == 0:
            break
    parts = [f'<span class="mx-n" style="--v:{groups[0]}"></span>']
    parts += [f'.<span class="mx-n p3" style="--v:{g}"></span>' for g in groups[1:]]
    return "".join(parts)


def load_card(records: int, period: str = "", secs: float | None = None) -> str:
    """Kartu ringkas 'data siap': jumlah transaksi (menghitung naik) + periode & waktu muat."""
    meta = []
    if period:
        meta.append(f"Periode {esc(period)}")
    if secs is not None and secs >= 0.1:
        whole, dec = int(secs), int(round((secs - int(secs)) * 10))
        if dec >= 10:
            whole, dec = whole + 1, 0
        meta.append(f'dimuat <b class="sec"><span class="mx-n" style="--v:{whole}"></span>,'
                    f'<span class="mx-n" style="--v:{dec}"></span></b> detik')
    return (f'<div class="mx-load"><div class="top"><i></i>Data siap</div>'
            f'<div class="big">{_count_html(records)}<small>transaksi</small></div>'
            f'<div class="sub">{" · ".join(meta)}</div></div>')


# ---------------------------------------------------------------------------
# KARTU STATUS KONEKSI (sidebar) — diukur di browser pengguna
# ---------------------------------------------------------------------------
NET_CARD_HEIGHT = 150


def net_card_html(server_ip: str = "", drive: dict | None = None) -> str:
    """Kartu minimalis status koneksi: kestabilan (ping ke server MTRAX), kecepatan unduh, IP & Google Drive.
    Diukur di browser; hasil disimpan 5 menit di sessionStorage agar tidak mengulang tes."""
    import json as _json
    ip = server_ip if isinstance(server_ip, str) else ""
    d = drive if isinstance(drive, dict) else {}
    d = {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in d.items()}
    cfg = _json.dumps({"ip": ip, "drive": d}).replace("</", "<\\/")
    return """<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,"Source Sans Pro",system-ui,sans-serif;color:#434343;background:transparent}
.card{border:1px solid #DEDEDE;border-radius:10px;background:#fff;padding:10px 12px}
.hd{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;background:#BDBDBD;flex:0 0 7px}
.ok .dot{background:#3B7A0B}.mid .dot{background:#E0A800}.bad .dot{background:#D61B54}
.run .dot{animation:blink 1s infinite}@keyframes blink{50%{opacity:.25}}
#st{flex:1}
.re{appearance:none;border:0;background:none;color:#6B6B6B;cursor:pointer;width:20px;height:20px;border-radius:6px;
 display:inline-flex;align-items:center;justify-content:center}
.re:hover{background:#F0F0F0;color:#0C79C2}.re:disabled{cursor:wait}
.re svg{width:13px;height:13px;stroke:currentColor;fill:none;stroke-width:2.2;stroke-linecap:round}
.run .re svg{animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.nums{display:grid;grid-template-columns:repeat(3,1fr);margin-top:8px}
.n{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1;white-space:nowrap}
.n span{font-size:10px;font-weight:400;color:#6B6B6B;margin-left:2px}
.l{font-size:10px;color:#6B6B6B;margin-top:2px}
.bars{display:flex;align-items:flex-end;gap:2px;height:14px;margin-top:8px}
.bars i{flex:1;border-radius:1px;background:#EDEDED;height:20%;transition:height .3s,background .3s}
.meta{margin-top:8px;font-size:10.5px;color:#6B6B6B;line-height:1.55}
.meta b{font-weight:500;color:#434343}
.ip{font-family:ui-monospace,Menlo,Consolas,monospace;cursor:pointer;border-bottom:1px dotted #BDBDBD}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="card">
 <div class="hd run" id="hd"><span class="dot"></span><span id="st">Mengukur koneksi…</span>
  <button class="re" id="re" title="Tes ulang"><svg viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/></svg></button></div>
 <div class="nums">
  <div><div class="n" id="lat">–</div><div class="l">Latensi</div></div>
  <div><div class="n" id="spd">–</div><div class="l">Unduh</div></div>
  <div><div class="n" id="jit">–</div><div class="l">Jitter</div></div>
 </div>
 <div class="bars" id="bars"></div>
 <div class="meta">IP <b class="ip" id="ip" title="Klik untuk tampilkan/sembunyikan">–</b><span id="net"></span><br>
  Cloud Drive <b id="drv">–</b></div>
</div>
<script>
const CFG = __CFG__;
const N = 10, KEY = "mtrax_net_v1", TTL = 5*60*1000;
const $ = id => document.getElementById(id);
const bars = $("bars"); for (let i=0;i<N;i++) bars.appendChild(document.createElement("i"));
let ipFull = CFG.ip || "", ipShown = false;
function mask(ip){ if(!ip) return "–"; if(ip.includes(":")) return ip.split(":").slice(0,2).join(":")+":••••";
  const p=ip.split("."); return p.length===4? p[0]+"."+p[1]+".•••.•••" : ip; }
function drawIp(){ $("ip").textContent = ipShown? (ipFull||"–") : mask(ipFull); }
$("ip").onclick = () => { ipShown=!ipShown; drawIp(); };
function drive(){ const d=CFG.drive||{};
  $("drv").textContent = !d.state ? "tidak diatur" :
    d.state==="ok" ? "tersambung · "+String(d.secs).replace(".",",")+" detik" : "gagal";
  if(d.msg) $("drv").title=d.msg; }
function base(){ try{ const L=window.parent.location; return L.origin + L.pathname.replace(/[^\\/]*$/,""); }catch(e){ return null; } }
async function ping(url){ const t=performance.now();
  const r=await fetch(url+(url.includes("?")?"&":"?")+"t="+Math.random(),{cache:"no-store"});
  if(!r.ok) throw 0; await r.text(); return performance.now()-t; }
function colorFor(ms){ return ms<100?"#9CC9E8":ms<250?"#E0A800":"#D61B54"; }
function setBar(i,ms){ const b=bars.children[i];
  if(ms==null){ b.style.height="100%"; b.style.background="#D61B54"; return; }
  b.style.height=Math.max(20,Math.min(100,ms/4))+"%"; b.style.background=colorFor(ms); }
function setSpeed(m){ $("spd").innerHTML = m==null? "–" :
  (m>=10? Math.round(m) : m.toFixed(1).replace(".",",")) + "<span>Mbps</span>"; }
function status(lat,jit,loss){ let cls="ok", txt="stabil";
  if(lat==null || loss>=3 || lat>400 || jit>120){cls="bad";txt="tidak stabil";}
  else if(loss>=1 || lat>=150 || jit>=40){cls="mid";txt="kurang stabil";}
  $("hd").className="hd "+cls; $("st").textContent="Koneksi "+txt; }
function render(r){ (r.rtts||[]).forEach((v,i)=>setBar(i,v));
  $("lat").innerHTML = r.lat==null? "–" : Math.round(r.lat)+"<span>ms</span>";
  $("jit").innerHTML = (r.jit==null? "–" : Math.round(r.jit)+"<span>ms</span>") + (r.loss? "<span>· "+r.loss+" gagal</span>" : "");
  setSpeed(r.mbps); status(r.lat,r.jit,r.loss);
  if(r.ip){ ipFull=r.ip; drawIp(); }
  const c=navigator.connection, parts=[];
  if(c && c.effectiveType) parts.push(c.effectiveType.toUpperCase());
  if(r.colo) parts.push("CDN "+r.colo);
  $("net").textContent = parts.length? " · "+parts.join(" · ") : ""; }
async function run(force){
  if(!force){ try{ const s=JSON.parse(window.parent.sessionStorage.getItem(KEY)||"null");
    if(s && Date.now()-s.at<TTL){ render(s); return; } }catch(e){} }
  $("re").disabled=true; $("hd").className="hd run"; $("st").textContent="Mengukur koneksi…";
  for(const b of bars.children){ b.style.height="20%"; b.style.background="#EDEDED"; }
  setSpeed(null);
  const b=base(), urls=[]; if(b) urls.push(b+"_stcore/health"); urls.push("/_stcore/health");
  let url=null; for(const u of urls){ try{ await ping(u); url=u; break; }catch(e){} }
  const rtts=[]; let loss=0;
  for(let i=0;i<N;i++){ let v=null; if(url){ try{ v=await ping(url); }catch(e){ v=null; } }
    if(v==null) loss++; rtts.push(v); setBar(i,v); await new Promise(r=>setTimeout(r,120)); }
  const ok=rtts.filter(v=>v!=null).sort((a,b)=>a-b);
  const lat= ok.length? ok[Math.floor(ok.length/2)] : null;
  let jit=null; const seq=rtts.filter(v=>v!=null);
  if(seq.length>1){ const d=[]; for(let i=1;i<seq.length;i++) d.push(Math.abs(seq[i]-seq[i-1]));
    d.sort((a,b)=>a-b); jit=d[Math.floor(d.length/2)]; }   // median: satu lonjakan tidak langsung dianggap tidak stabil
  let ip=null, colo=null;
  try{ const t=await (await fetch("https://speed.cloudflare.com/cdn-cgi/trace",{cache:"no-store"})).text();
    const m=Object.fromEntries(t.trim().split("\\n").map(l=>l.split("="))); ip=m.ip||null; colo=m.colo||null; }catch(e){}
  let mbps=null;
  try{ const t0=performance.now();
    const r=await fetch("https://speed.cloudflare.com/__down?bytes=2000000&r="+Math.random(),{cache:"no-store"});
    const buf=await r.arrayBuffer(); const sec=(performance.now()-t0)/1000;
    mbps=(buf.byteLength*8/1e6)/Math.max(sec,0.05);
  }catch(e){ const c=navigator.connection; if(c && c.downlink) mbps=c.downlink; }
  const res={at:Date.now(),rtts,lat,jit,loss,mbps,ip,colo};
  try{ window.parent.sessionStorage.setItem(KEY, JSON.stringify(res)); }catch(e){}
  render(res); }
$("re").onclick=()=>run(true).finally(()=>{$("re").disabled=false;});
drawIp(); drive(); run(false).finally(()=>{$("re").disabled=false;});
</script></body></html>""".replace("__CFG__", cfg)


# ---------------------------------------------------------------------------
# CHIP STATUS KONEKSI DI HEADER (+ popover detail)
# Iframe komponen diletakkan (position:fixed) tepat di atas placeholder .mx-net-ph di header.
# ---------------------------------------------------------------------------
NET_CHIP_PLACEHOLDER = '<span class="mx-chip mx-net-ph" style="visibility:hidden">● Mengukur…</span>'


def net_chip_html(server_ip: str = "", drive: dict | None = None) -> str:
    import json as _json
    ip = server_ip if isinstance(server_ip, str) else ""
    d = drive if isinstance(drive, dict) else {}
    d = {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in d.items()}
    cfg = _json.dumps({"ip": ip, "drive": d}).replace("</", "<\\/")
    return """<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:transparent;overflow:hidden;font-family:"Source Sans Pro","Source Sans 3",Inter,system-ui,sans-serif;color:#434343}
.chip{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:999px;font-size:12.5px;font-weight:600;
 white-space:nowrap;cursor:pointer;border:1px solid #E6EBEF;background:#F5F8FA;color:#434343;user-select:none;
 transition:background .2s,border-color .2s,color .2s}
.chip .d{width:7px;height:7px;border-radius:50%;background:#BDBDBD}
.chip .c{font-weight:400;color:#6B6B6B}
.chip.ok{background:#EEF7E6;border-color:#CFE6BB;color:#3B7A0B}.chip.ok .d{background:#3B7A0B;box-shadow:0 0 0 3px rgba(59,122,11,.15)}
.chip.mid{background:#FFF8E0;border-color:#F2DE9B;color:#8A6400}.chip.mid .d{background:#E0A800}
.chip.bad{background:#FDEBF1;border-color:#F3C1D1;color:#B0123F}.chip.bad .d{background:#D61B54}
.chip.run .d{animation:blink 1s infinite}@keyframes blink{50%{opacity:.25}}
.pop{position:absolute;top:42px;left:0;width:300px;background:#fff;border:1px solid #DEDEDE;border-radius:12px;
 box-shadow:0 10px 30px rgba(0,0,0,.10);padding:14px;font-size:12.5px;display:none}
.pop.on{display:block;animation:in .15s ease-out}@keyframes in{from{opacity:0;transform:translateY(-4px)}}
.pop:before{content:"";position:absolute;top:-6px;left:24px;width:10px;height:10px;background:#fff;border-left:1px solid #DEDEDE;
 border-top:1px solid #DEDEDE;transform:rotate(45deg)}
.h{display:flex;align-items:center;gap:7px;font-weight:600;font-size:13px}.h .d{width:7px;height:7px;border-radius:50%;background:#BDBDBD}
.ok2 .d{background:#3B7A0B}.mid2 .d{background:#E0A800}.bad2 .d{background:#D61B54}
.h .r{margin-left:auto;color:#6B6B6B;font-weight:400;font-size:11px}
.g{display:grid;grid-template-columns:repeat(3,1fr);margin-top:10px}
.n{font-size:16px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}.n small{font-size:10.5px;font-weight:400;color:#6B6B6B;margin-left:2px}
.l{font-size:10.5px;color:#6B6B6B}
.sp{margin-top:10px}.sp .t{display:flex;justify-content:space-between;font-size:10.5px;color:#6B6B6B}
.sp svg{display:block;width:100%;height:38px;margin-top:4px}
.m{margin-top:10px;padding-top:10px;border-top:1px solid #EFEFEF;display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:11.5px}
.m .k{color:#6B6B6B}.m b{font-weight:500}
.ip{font-family:ui-monospace,Menlo,Consolas,monospace;border-bottom:1px dotted #BDBDBD;cursor:pointer}
.bt{margin-top:12px;display:flex;justify-content:space-between;align-items:center;font-size:10.5px;color:#9A9A9A}
button{border:1px solid #DEDEDE;background:#fff;border-radius:8px;padding:4px 10px;font:500 11.5px inherit;color:#0C79C2;cursor:pointer}
button:hover{background:#F5F8FA}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<span class="chip run" id="chip"><span class="d"></span><span id="ct">Mengukur…</span><span class="c" id="cm"></span></span>
<div class="pop" id="pop">
 <div class="h" id="ph"><span class="d"></span><span id="pt">Mengukur koneksi…</span><span class="r" id="at"></span></div>
 <div class="g"><div><div class="n" id="lat">–</div><div class="l">Latensi</div></div>
  <div><div class="n" id="spd">–</div><div class="l">Unduh</div></div>
  <div><div class="n" id="jit">–</div><div class="l">Jitter</div></div></div>
 <div class="sp"><div class="t"><span>Latensi 10 menit terakhir</span><span id="mx"></span></div><svg id="sv" viewBox="0 0 270 38" preserveAspectRatio="none"></svg></div>
 <div class="m"><span class="k">IP publik</span><b><span class="ip" id="ip" title="Klik untuk tampilkan/sembunyikan">–</span></b>
  <span class="k">Jaringan</span><b id="net">–</b>
  <span class="k">Cloud Drive</span><b id="drv">–</b></div>
 <div class="bt"><span>IP tidak disimpan di server</span><button id="re">↻ Tes ulang</button></div>
</div>
<script>
const CFG=__CFG__, N=10, KEY="mtrax_net_v2", HKEY="mtrax_net_hist", TTL=5*60*1000;
const $=id=>document.getElementById(id), fe=window.frameElement, P=window.parent;
let D=null; try{ D=P.document; }catch(e){}
let open=false, ipFull=CFG.ip||"", ipShown=false;
const POP_H=336;
function ph(){ return D? D.querySelector(".mx-net-ph") : null; }
function place(){ if(!fe||!D) return; const p=ph();
  if(!p){ fe.style.display="none"; return; }
  const w=$("chip").offsetWidth+1; p.style.width=w+"px"; p.style.display="inline-flex";
  const b=p.getBoundingClientRect();
  Object.assign(fe.style,{display:"block",position:"fixed",left:b.left+"px",top:b.top+"px",border:"0",zIndex:"999",
    width:(open?Math.max(302,w):w)+"px",height:(open?POP_H:b.height)+"px",colorScheme:"normal",background:"transparent"});
  const box=fe.closest('[data-testid="stElementContainer"]')||fe.parentElement;   // keluarkan dari alur layout
  if(box){ box.style.position="absolute"; box.style.height="0px"; box.style.width="0px"; box.style.overflow="visible"; } }
if(D){ D.addEventListener("scroll",place,true); P.addEventListener("resize",place);
  D.addEventListener("mousedown",()=>{ if(open){ setOpen(false); } }); }
setInterval(place,400);
function setOpen(v){ open=v; $("pop").classList.toggle("on",v); place(); }
$("chip").onclick=()=>setOpen(!open);
function mask(ip){ if(!ip) return "–"; if(ip.includes(":")) return ip.split(":").slice(0,2).join(":")+":••••";
  const p=ip.split("."); return p.length===4? p[0]+"."+p[1]+".•••.•••" : ip; }
function drawIp(){ $("ip").textContent=ipShown?(ipFull||"–"):mask(ipFull); }
$("ip").onclick=()=>{ ipShown=!ipShown; drawIp(); };
(function(){ const d=CFG.drive||{}; const el=$("drv");
  if(!d.state){ el.textContent="tidak diatur"; el.style.color="#6B6B6B"; }
  else if(d.state==="ok"){ el.textContent="● tersambung · "+String(d.secs).replace(".",",")+" detik"; el.style.color="#3B7A0B"; }
  else { el.textContent="● gagal"; el.style.color="#D61B54"; if(d.msg) el.title=d.msg; } })();
function base(){ try{ const L=P.location; return L.origin+L.pathname.replace(/[^\\/]*$/,""); }catch(e){ return null; } }
let URL_=null;
async function ping(url){ const t=performance.now();
  const r=await fetch(url+(url.includes("?")?"&":"?")+"t="+Math.random(),{cache:"no-store"}); if(!r.ok) throw 0; await r.text();
  return performance.now()-t; }
async function findUrl(){ if(URL_) return URL_; const b=base(), us=[]; if(b) us.push(b+"_stcore/health"); us.push("/_stcore/health");
  for(const u of us){ try{ await ping(u); URL_=u; return u; }catch(e){} } return null; }
function ss(){ try{ return P.sessionStorage; }catch(e){ return null; } }
function hist(){ try{ return JSON.parse((ss()&&ss().getItem(HKEY))||"[]"); }catch(e){ return []; } }
function pushHist(ms){ const h=hist().filter(p=>Date.now()-p.t<10*60*1000); h.push({t:Date.now(),ms}); try{ ss()&&ss().setItem(HKEY,JSON.stringify(h.slice(-40))); }catch(e){} return h; }
function spark(){ const h=hist().filter(p=>Date.now()-p.t<10*60*1000), sv=$("sv");
  if(h.length<2){ sv.innerHTML='<text x="135" y="23" text-anchor="middle" font-size="10" fill="#9A9A9A">grafik terisi otomatis tiap 30 detik</text>'; $("mx").textContent=""; return; }
  const vals=h.map(p=>p.ms==null?null:p.ms), ok=vals.filter(v=>v!=null), mx=Math.max(50,...ok);
  const x=i=>i*270/(h.length-1), y=v=>36-(v/mx)*32;
  let pts=[]; vals.forEach((v,i)=>{ if(v!=null) pts.push(x(i).toFixed(1)+","+y(v).toFixed(1)); });
  const fails=vals.map((v,i)=>v==null?'<line x1="'+x(i)+'" x2="'+x(i)+'" y1="2" y2="38" stroke="#D61B54" stroke-width="1.5" stroke-dasharray="2 2"/>':"").join("");
  const last=pts[pts.length-1]||"0,36";
  sv.innerHTML='<polygon points="'+pts.join(" ")+' '+last.split(",")[0]+',38 '+(pts[0]||"0,36").split(",")[0]+',38" fill="rgba(27,160,226,.08)"/>'+
    '<polyline points="'+pts.join(" ")+'" fill="none" stroke="#1BA0E2" stroke-width="1.8" stroke-linejoin="round"/>'+fails+
    '<circle cx="'+last.split(",")[0]+'" cy="'+last.split(",")[1]+'" r="3" fill="#1BA0E2"/>';
  $("mx").textContent= ok.length? "maks "+Math.round(Math.max(...ok))+" ms" : ""; }
function status(lat,jit,loss){ let c="ok",t="Stabil";
  if(lat==null||loss>=3||lat>400||jit>120){c="bad";t="Tidak stabil";} else if(loss>=1||lat>=150||jit>=40){c="mid";t="Kurang stabil";}
  return [c,t]; }
function render(r){ const [c,t]=status(r.lat,r.jit,r.loss);
  $("chip").className="chip "+c; $("ct").textContent=t; $("cm").textContent=r.lat==null?"":"· "+Math.round(r.lat)+" ms";
  $("ph").className="h "+c+"2"; $("pt").textContent="Koneksi "+t.toLowerCase();
  $("at").textContent="diukur "+new Date(r.at).toLocaleTimeString("id-ID",{hour:"2-digit",minute:"2-digit"});
  $("lat").innerHTML=r.lat==null?"–":Math.round(r.lat)+"<small>ms</small>";
  $("jit").innerHTML=(r.jit==null?"–":Math.round(r.jit)+"<small>ms</small>")+(r.loss?"<small>· "+r.loss+" gagal</small>":"");
  $("spd").innerHTML=r.mbps==null?"–":(r.mbps>=10?Math.round(r.mbps):r.mbps.toFixed(1).replace(".",","))+"<small>Mbps</small>";
  if(r.ip){ ipFull=r.ip; } drawIp();
  const cn=navigator.connection, parts=[]; if(cn&&cn.effectiveType) parts.push(cn.effectiveType.toUpperCase()); if(r.colo) parts.push("CDN "+r.colo);
  $("net").textContent=parts.length?parts.join(" · "):"–"; spark(); place(); }
async function run(force){
  if(!force){ try{ const s=JSON.parse((ss()&&ss().getItem(KEY))||"null"); if(s&&Date.now()-s.at<TTL){ render(s); return; } }catch(e){} }
  $("chip").className="chip run"; $("ct").textContent="Mengukur…"; $("cm").textContent=""; $("re").disabled=true; place();
  const u=await findUrl(), rtts=[]; let loss=0;
  for(let i=0;i<N;i++){ let v=null; if(u){ try{ v=await ping(u); }catch(e){} } if(v==null) loss++; rtts.push(v); await new Promise(r=>setTimeout(r,120)); }
  const ok=rtts.filter(v=>v!=null).sort((a,b)=>a-b), lat=ok.length?ok[Math.floor(ok.length/2)]:null;
  let jit=null; const seq=rtts.filter(v=>v!=null); if(seq.length>1){ const d=[]; for(let i=1;i<seq.length;i++) d.push(Math.abs(seq[i]-seq[i-1])); d.sort((a,b)=>a-b); jit=d[Math.floor(d.length/2)]; }
  let ip=null,colo=null; try{ const t=await (await fetch("https://speed.cloudflare.com/cdn-cgi/trace",{cache:"no-store"})).text();
    const m=Object.fromEntries(t.trim().split("\\n").map(l=>l.split("="))); ip=m.ip||null; colo=m.colo||null; }catch(e){}
  let mbps=null; try{ const t0=performance.now(); const r=await fetch("https://speed.cloudflare.com/__down?bytes=2000000&r="+Math.random(),{cache:"no-store"});
    const buf=await r.arrayBuffer(); mbps=(buf.byteLength*8/1e6)/Math.max((performance.now()-t0)/1000,0.05);
  }catch(e){ const cn=navigator.connection; if(cn&&cn.downlink) mbps=cn.downlink; }
  if(lat!=null) pushHist(lat);
  const res={at:Date.now(),lat,jit,loss,mbps,ip,colo}; try{ ss()&&ss().setItem(KEY,JSON.stringify(res)); }catch(e){}
  render(res); $("re").disabled=false; }
$("re").onclick=()=>run(true);
setInterval(async()=>{ const u=await findUrl(); let v=null; if(u){ try{ v=await ping(u); }catch(e){} } pushHist(v); spark(); },30000);
drawIp(); place(); run(false);
</script></body></html>""".replace("__CFG__", cfg)


# ---------------------------------------------------------------------------
# KARTU LAPORAN (halaman Reports)
# ---------------------------------------------------------------------------
_REP_ICO = {
    "hotel": '<path d="M4 21V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v16M15 9h4a1 1 0 0 1 1 1v11M3 21h18M8 8h3M8 12h3M8 16h3"/>',
    "pin": '<path d="M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    "coin": '<circle cx="12" cy="12" r="8.5"/><path d="M14.8 9.2A2.6 2.6 0 0 0 12.4 8h-1a2 2 0 0 0 0 4h1.2a2 2 0 0 1 0 4h-1.1a2.6 2.6 0 0 1-2.4-1.3M12 6.5V8M12 16v1.5"/>',
    "leak": '<path d="M14 4h6v6M20 4l-8 8M10 5H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-4"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
}


def report_card(title: str, desc: str, rows: int, icon: str = "file") -> str:
    """Isi kartu laporan: ikon, judul, deskripsi (2 baris), jumlah baris & format."""
    ico = _REP_ICO.get(icon, _REP_ICO["file"])
    return (f'<div class="mx-rep"><div class="hd"><span class="ic"><svg viewBox="0 0 24 24">{ico}</svg></span>'
            f'<span class="t">{esc(title)}</span></div>'
            f'<div class="d">{esc(desc)}</div>'
            f'<div class="m"><span>{fmt_int(rows)} baris</span><span>Excel · .xlsx</span></div></div>')
