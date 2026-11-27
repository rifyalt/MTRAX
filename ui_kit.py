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

UI_VERSION = "2026.11.27-b"   # tampil di footer; cek ini bila tampilan terasa belum berubah

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
  margin:6px 0 18px; }}
.mx-scope .row {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); }}
.mx-scope .it {{ padding:14px 18px; border-right:1px solid #EDEDED; }}
.mx-scope .it:last-child {{ border-right:none; }}
.mx-scope .l {{ font-size:12.5px; color:var(--muted); }}
.mx-scope .v {{ font-size:18px; font-weight:600; color:var(--ink); margin-top:4px; font-variant-numeric:tabular-nums; }}
.mx-scope .s {{ font-size:12.5px; color:var(--muted); margin-top:2px; }}
.mx-scope .ft {{ border-top:1px solid #EDEDED; padding:9px 18px; font-size:12.5px; color:var(--muted); background:#FAFAFA;
  border-radius:0 0 var(--r) var(--r); }}
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
.mx-load {{ position: relative; overflow: hidden; border-radius: 10px; padding: 12px 14px 14px;
  background: linear-gradient(135deg, #EEF9E4 0%, #E6F4FB 100%); border: 1px solid #D5EBC3;
  animation: mxLoadIn .45s ease-out both; }}
.mx-load .top {{ display: flex; align-items: center; gap: 8px; color: #3B7A0B; font-size: 12px;
  font-weight: 600; letter-spacing: .02em; text-transform: uppercase; }}
.mx-load .tick {{ width: 18px; height: 18px; border-radius: 50%; background: #3B7A0B; display: inline-flex;
  align-items: center; justify-content: center; animation: mxPop .5s .9s cubic-bezier(.2,1.6,.4,1) both; }}
.mx-load .tick svg {{ width: 11px; height: 11px; stroke: #fff; stroke-width: 3; fill: none;
  stroke-dasharray: 20; stroke-dashoffset: 20; animation: mxDraw .35s 1.15s ease-out forwards; }}
.mx-load .big {{ font-size: 26px; font-weight: 700; color: #434343; line-height: 1.15; margin-top: 6px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
.mx-load .big small {{ font-size: 13px; font-weight: 500; color: #6B6B6B; margin-left: 4px; }}
.mx-load .sub {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }}
.mx-load .pill {{ font-size: 12px; color: #0A5D96; background: rgba(255,255,255,.75); border: 1px solid #D6EEFA;
  border-radius: 999px; padding: 2px 9px; white-space: nowrap; font-variant-numeric: tabular-nums;
  animation: mxLoadIn .4s ease-out both; }}
.mx-load .pill:nth-child(2) {{ animation-delay: .25s; }}
.mx-load .bar {{ position: absolute; left: 0; bottom: 0; height: 3px; width: 100%;
  background: linear-gradient(90deg, #1BA0E2, #98EA16); transform-origin: left;
  animation: mxBar 1.3s ease-out both; }}
.mx-n {{ counter-reset: c var(--v); animation: mxCount 1.3s cubic-bezier(.15,.8,.25,1) both; }}
.mx-n::after {{ content: counter(c); }}
.mx-n.p3::after {{ content: counter(c, mxpad3); }}
@keyframes mxCount {{ from {{ --v: 0; }} }}
@keyframes mxBar {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes mxLoadIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes mxPop {{ from {{ transform: scale(0); }} to {{ transform: scale(1); }} }}
@keyframes mxDraw {{ to {{ stroke-dashoffset: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .mx-load, .mx-load *, .mx-n {{ animation: none !important; }}
  .mx-load .tick svg {{ stroke-dashoffset: 0; }} }}
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


def scope_bar(items, footnote_html=""):
    """Satu kartu berisi beberapa angka cakupan data, dipisah garis tipis, plus catatan kecil di bawah."""
    cells = "".join(f'<div class="it"><div class="l">{esc(a)}</div><div class="v">{esc(b)}</div>'
                    f'<div class="s">{esc(c)}</div></div>' for a, b, c in items)
    ft = f'<div class="ft">{footnote_html}</div>' if footnote_html else ""
    st.markdown(f'<div class="mx-scope"><div class="row">{cells}</div>{ft}</div>', unsafe_allow_html=True)


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
    """Kartu 'data dimuat': jumlah record & waktu proses menghitung naik saat pertama tampil."""
    pills = []
    if period:
        pills.append(f'<span class="pill">Periode {esc(period)}</span>')
    if secs is not None and secs >= 0.1:
        whole, dec = int(secs), int(round((secs - int(secs)) * 10)) % 10
        if round((secs - int(secs)) * 10) >= 10:
            whole += 1
        pills.append(f'<span class="pill">dimuat <span class="mx-n" style="--v:{whole}"></span>,'
                     f'<span class="mx-n" style="--v:{dec}"></span> detik</span>')
    return (f'<div class="mx-load"><div class="top"><span class="tick"><svg viewBox="0 0 24 24">'
            f'<path d="M5 12.5l4.5 4.5L19 7.5"/></svg></span>Data siap</div>'
            f'<div class="big">{_count_html(records)}<small>transaksi</small></div>'
            f'<div class="sub">{"".join(pills)}</div><div class="bar"></div></div>')


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
