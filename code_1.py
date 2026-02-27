# =====================================
# CORE IMPORTS
# =====================================
import streamlit as st
import pandas as pd
import os
import hashlib
import base64
import json
import requests
import numpy as np
import shutil
import time
import re
import hmac

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gdown
import xml.etree.ElementTree as ET

# =====================================
# OPTIONAL LSTM SUPPORT
# =====================================
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# =====================================
# 2FA / TOTP SUPPORT
# ── Ditambahkan: Pure-Python TOTP (RFC 6238) — tidak butuh library tambahan ──
# Jika pyotp & qrcode terinstall, QR code akan ditampilkan.
# Jika tidak, user cukup input manual key ke Google Authenticator.
# =====================================
import struct as _struct
import hmac   as _hmac_mod
import base64 as _b64_mod

try:
    import pyotp
    import qrcode
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

def _hotp(key_b32: str, counter: int) -> str:
    """HMAC-based OTP — RFC 4226."""
    key  = _b64_mod.b32decode(key_b32.upper().replace(" ", ""), casefold=True)
    msg  = _struct.pack(">Q", counter)
    h    = _hmac_mod.new(key, msg, "sha1").digest()
    off  = h[-1] & 0x0F
    code = (_struct.unpack(">I", h[off:off+4])[0] & 0x7FFFFFFF) % 1_000_000
    return str(code).zfill(6)

def _totp_now(key_b32: str, window: int = 30) -> str:
    counter = int(time.time()) // window
    return _hotp(key_b32, counter)

def _totp_valid(key_b32: str, code: str, window: int = 30, drift: int = 1) -> bool:
    """Verify TOTP allowing ±drift windows (toleransi clock skew)."""
    counter = int(time.time()) // window
    for d in range(-drift, drift + 1):
        if _hmac_mod.compare_digest(_hotp(key_b32, counter + d), code.strip()):
            return True
    return False

def _generate_totp_secret() -> str:
    """Generate random 20-byte base32 TOTP secret."""
    return _b64_mod.b32encode(os.urandom(20)).decode()

def _provisioning_uri(secret: str, username: str, issuer: str = "MTRAX") -> str:
    from urllib.parse import quote
    return (
        f"otpauth://totp/{quote(issuer)}:{quote(username)}"
        f"?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30"
    )

def _make_qr_b64(uri: str) -> str:
    """Render QR code -> base64 PNG. Tries qrcode -> segno -> pyqrcode in order."""
    import io

    # attempt 1: qrcode (pil)
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#9c5789", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # attempt 2: segno
    try:
        import segno
        qr = segno.make(uri, error="M")
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=6, border=2,
                dark="#9c5789", light="white")
        buf.seek(0)
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # attempt 3: pyqrcode
    try:
        import pyqrcode
        qr = pyqrcode.create(uri)
        buf = io.BytesIO()
        qr.png(buf, scale=6, module_color=(156, 87, 137), background=(255, 255, 255))
        buf.seek(0)
        return _b64_mod.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    return ""

def _load_totp_secrets() -> dict:
    """
    Load TOTP secrets — semua user berbagi secret milik 'admin'.
    Hanya satu QR / kode Authenticator yang perlu di-setup oleh admin.
    Priority: secrets.toml [totp][admin] -> session_state (runtime-generated).
    """
    # Ambil / generate satu shared secret dari admin
    shared_secret = None
    try:
        val = st.secrets["totp"]["admin"]
        if val:
            shared_secret = val
    except Exception:
        pass

    if shared_secret is None:
        key = "_totp_secret_admin"
        if key not in st.session_state:
            st.session_state[key] = _generate_totp_secret()
        shared_secret = st.session_state[key]

    # Semua user memakai secret yang sama (shared admin TOTP)
    return {uname: shared_secret for uname in USERS.keys()}

# ======================================
# ADVANCED LOGIN SECURITY CONFIG
# ======================================

# Jika pakai secrets.toml (recommended)
MAX_LOGIN_ATTEMPTS = st.secrets.get("security", {}).get("max_login_attempts", 5)
LOCKOUT_SECONDS = st.secrets.get("security", {}).get("lockout_seconds", 300)  # 5 menit
SESSION_TIMEOUT = st.secrets.get("security", {}).get("session_timeout_minutes", 30) * 60

def secure_compare(plain_password, stored_hash):
    """
    Timing attack safe comparison
    """
    hashed_input = hashlib.sha256(plain_password.encode()).hexdigest()
    return hmac.compare_digest(hashed_input, stored_hash)

def validate_username(username):
    """
    Mencegah injection / karakter aneh
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,30}", username))

def init_login_security():
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "lockout_until" not in st.session_state:
        st.session_state.lockout_until = 0
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    # ── 2FA states ──────────────────────────────────────────
    if "pending_2fa" not in st.session_state:
        st.session_state.pending_2fa = False
    if "pending_user" not in st.session_state:
        st.session_state.pending_user = ""
    if "pending_role" not in st.session_state:
        st.session_state.pending_role = ""
    if "totp_enrolled" not in st.session_state:
        st.session_state.totp_enrolled = {}   # {username: True/False}

def check_session_timeout():
    if st.session_state.get("login_time"):
        if time.time() - st.session_state.login_time > SESSION_TIMEOUT:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.warning("Session expired. Please login again.")
            st.stop()

init_login_security()


# =====================================
# HELPER FUNCTIONS
# =====================================
def get_greeting():
    """Mendapatkan greeting sesuai waktu Jakarta"""
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    hour = now.hour

    if hour < 11:
        greet = "Good Morning"
    elif hour < 15:
        greet = "Good Afternoon"
    elif hour < 18:
        greet = "Good Evening"
    else:
        greet = "Good Night"

    return greet, now


@st.cache_data(show_spinner=False)
def load_drive_data(folder_id, drop_cols):
    shutil.rmtree("data_temp", ignore_errors=True)
    os.makedirs("data_temp", exist_ok=True)

    gdown.download_folder(
        id=folder_id,
        output="data_temp",
        quiet=True,
        use_cookies=False
    )

    files = [
        f for f in os.listdir("data_temp")
        if f.endswith((".xlsx", ".xls"))
    ]

    dfs = []

    for f in files:
        df = pd.read_excel(os.path.join("data_temp", f))

        # drop kolom tidak perlu
        df = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors="ignore"
        )

        # 🔹 TRIM & CLEAN STRING DATA
        df = trim_string_columns(df)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

def build_employee_cohort(df):
    required_cols = ["Employee Id", "Issue Time", "Travel Request Number"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df = df.copy()
    df["Issue Time"] = pd.to_datetime(df["Issue Time"], errors="coerce")
    df = df.dropna(subset=["Issue Time", "Employee Id"])

    df["OrderMonth"] = df["Issue Time"].dt.to_period("M")
    
    # Cohort = first booking month per employee
    df["CohortMonth"] = (
        df.groupby("Employee Id")["OrderMonth"]
        .transform("min")
    )

    # Cohort index (bulan ke-n sejak first booking)
    df["CohortIndex"] = (
        df["OrderMonth"].astype(int) -
        df["CohortMonth"].astype(int)
    )

    cohort = (
        df.groupby(["CohortMonth", "CohortIndex"])["Travel Request Number"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort.pivot(
        index="CohortMonth",
        columns="CohortIndex",
        values="Travel Request Number"
    ).fillna(0)

    cohort_pivot.index = cohort_pivot.index.astype(str)

    return cohort_pivot


#==========================#
# FUNGSI AUTO-CANONICAL MAPPING
#==========================#
def auto_canonical_hotel_mapping(
    df,
    hotel_col="Hotel Name",
    threshold=0.88
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    hotel_series = (
        df[hotel_col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
        .drop_duplicates()
    )

    hotel_names = hotel_series.tolist()

    if len(hotel_names) < 2:
        df["Canonical Hotel Name"] = df[hotel_col]
        return df, pd.DataFrame()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    tfidf = vectorizer.fit_transform(hotel_names)
    similarity = cosine_similarity(tfidf)

    clusters = {}
    visited = set()

    for i, name in enumerate(hotel_names):
        if i in visited:
            continue

        group = [name]
        visited.add(i)

        for j in range(i + 1, len(hotel_names)):
            if similarity[i, j] >= threshold:
                group.append(hotel_names[j])
                visited.add(j)

        # canonical = nama terpanjang
        canonical = max(group, key=len)
        for g in group:
            clusters[g] = canonical

    mapping_df = pd.DataFrame(
        clusters.items(),
        columns=["Hotel Name Clean", "Canonical Hotel Name"]
    )

    # merge ke df awal
    df_out = df.copy()
    df_out["_hotel_clean"] = (
        df_out[hotel_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )

    df_out = df_out.merge(
        mapping_df,
        left_on="_hotel_clean",
        right_on="Hotel Name Clean",
        how="left"
    )

    df_out["Canonical Hotel Name"] = (
        df_out["Canonical Hotel Name"]
        .fillna(df_out[hotel_col])
    )

    df_out.drop(columns=["_hotel_clean", "Hotel Name Clean"], inplace=True)

    return df_out, mapping_df



#==========================#    
# TEXT SIMILARITY
#==========================#
def hotel_name_similarity(df, text_col="Hotel Name", threshold=0.75):
    df_text = (
        df[[text_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    df_text[text_col] = (
        df_text[text_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    tfidf_matrix = vectorizer.fit_transform(df_text[text_col])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []
    names = df_text[text_col].tolist()

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = similarity_matrix[i, j]
            if score >= threshold:
                results.append({
                    "Hotel Name A": names[i],
                    "Hotel Name B": names[j],
                    "Similarity Score": round(score, 3)
                })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df["Level"] = pd.cut(
            result_df["Similarity Score"],
            bins=[0.7, 0.8, 0.9, 1.0],
            labels=["Medium", "High", "Very High"]
        ).astype(str)

        result_df = result_df.sort_values(
            by="Similarity Score",
            ascending=False
        )

    return result_df

def trim_string_columns(df):
    """
    Membersihkan semua kolom bertipe object (string):
    - strip spasi depan & belakang
    - hapus spasi ganda di tengah
    """
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        # kembalikan NaN asli (bukan string 'nan')
        df_clean[col] = df_clean[col].replace("nan", np.nan)

    return df_clean

# ===============================
# SESSION STATE INIT
# ===============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "df_all" not in st.session_state:
    st.session_state.df_all = pd.DataFrame()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "data_period" not in st.session_state:
    st.session_state.data_period = ""

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ======================================
# USER LOGIN
# ======================================
def hash_password(password: str) -> str:
    """Hash password menggunakan SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {
        "password": st.secrets["auth"]["admin_password"],
        "role": "Admin"
    },
    "ssc": {
        "password": st.secrets["auth"]["ssc_password"],
        "role": "Analyst"
    },
    "dtm": {
        "password": st.secrets["auth"]["dtm_password"],
        "role": "Viewer"
    },
}

# ======================================
# BMKG FUNCTIONS
# ======================================
def get_bmkg_realtime_quake():
    """Mengambil data gempa real-time dari BMKG"""
    url = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "Infogempa" in data and "gempa" in data["Infogempa"]:
            return data["Infogempa"]["gempa"]

        return None

    except Exception as e:
        st.sidebar.error(f"BMKG Error: {e}")
        return None

# ======================================
# NEWS TICKER
# ======================================
def fetch_rss_news(limit=10):
    urls = [
        "https://news.google.com/rss/search?q=danantara&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=pertamina&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=bmkg&hl=id&gl=ID&ceid=ID:id"
    ]

    news = []

    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            root = ET.fromstring(r.content)

            for item in root.findall(".//item")[:limit]:
                title = item.find("title").text
                link = item.find("link").text
                news.append({"title": title, "link": link})
        except Exception:
            pass

    return news[:limit]

def render_news_ticker(news):
    items = "".join([
        f'<a class="news-item" href="{n["link"]}" target="_blank">{n["title"]}</a>'
        for n in news
    ])
    st.markdown(f"""
    <div class="news-ticker">
        <div class="ticker-content">{items}</div>
    </div>
    """, unsafe_allow_html=True)


# ======================================
# ── 2FA PAGES (DITAMBAHKAN) ──
# ======================================

def _2fa_css():
    return """
    <style>
    .fa-card {
        background: white;
        border-radius: 8px;
        padding: 36px 40px;
        max-width: 440px;
        margin: 3rem auto 0 auto;
        box-shadow: 0 2px 16px rgba(0,0,0,0.09);
        border-top: 3px solid #9c5789;
    }
    .fa-title {
        font-size: 1.25em;
        font-weight: 700;
        color: #9c5789;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .fa-sub {
        font-size: 0.83em;
        color: #888;
        margin-bottom: 24px;
        line-height: 1.6;
    }
    .fa-qr-wrap {
        background: #fdf7fc;
        border: 1px solid #e8d5e4;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-bottom: 18px;
    }
    .fa-secret-box {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: 'Courier New', monospace;
        font-size: 1.05em;
        letter-spacing: 0.14em;
        color: #333;
        text-align: center;
        margin: 10px 0 18px 0;
        word-break: break-all;
    }
    .fa-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }
    .fa-step-num {
        background: #9c5789;
        color: white;
        border-radius: 50%;
        width: 22px; height: 22px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75em; font-weight: 700;
        flex-shrink: 0; margin-top: 2px;
    }
    .fa-step-text { font-size: 0.85em; color: #444; line-height: 1.55; }
    .fa-timer {
        display: inline-flex; align-items: center; gap: 8px;
        background: #fdf7fc; border: 1px solid #e8d5e4;
        border-radius: 20px; padding: 5px 16px;
        font-size: 0.78em; color: #9c5789;
        margin-top: 16px;
    }
    </style>
    """

def twofa_setup_page(username: str):
    """Halaman setup 2FA — redesign v2: editorial split-panel."""
    import streamlit.components.v1 as _components
    st.set_page_config(page_title="Setup 2FA | MTRAX", layout="centered")

    totp_secrets = _load_totp_secrets()
    secret       = totp_secrets[username]
    uri          = _provisioning_uri(secret, "admin")   # akun selalu MTRAX:admin
    qr_b64       = _make_qr_b64(uri)
    fmt_secret   = " ".join([secret[i:i+4] for i in range(0, len(secret), 4)])

    if qr_b64:
        qr_html = f'<img id="qrimg" src="data:image/png;base64,{qr_b64}" alt="QR Code" />'
        qr_note = ''
    else:
        qr_html = ''
        qr_note = '<div class="qr-missing">📱 Install <code>qrcode[pil]</code> atau <code>segno</code> untuk QR code</div>'

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box;}}

:root {{
  --ink:    #0f0a12;
  --ink2:   #3a2a3a;
  --muted:  #8a7a8a;
  --line:   #e8dde8;
  --bg:     #f9f6fb;
  --white:  #ffffff;
  --accent: #9c5789;
  --acc-l:  #f3eaf1;
  --acc-d:  #6a1a5a;
  --green:  #22c55e;
  --mono:   'JetBrains Mono', monospace;
  --sans:   'Sora', sans-serif;
}}

body{{font-family:var(--sans);background:var(--bg);padding:20px 12px 28px;}}

/* ── wrapper ── */
.w{{max-width:500px;margin:0 auto;animation:up .5s cubic-bezier(.16,1,.3,1) both;}}
@keyframes up{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:none}}}}

/* ── topbar ── */
.topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}}
.logo{{font-size:.72em;font-weight:700;letter-spacing:.28em;color:var(--accent);}}
.badge{{display:inline-flex;align-items:center;gap:5px;font-size:.65em;font-weight:600;
  letter-spacing:.1em;color:var(--accent);background:var(--acc-l);
  border:1px solid #ddc8d8;border-radius:100px;padding:3px 10px;}}
.badge-dot{{width:5px;height:5px;border-radius:50%;background:var(--green);
  box-shadow:0 0 5px var(--green);animation:blink 2s ease infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}

/* ── card ── */
.card{{background:var(--white);border-radius:18px;
  box-shadow:0 2px 24px rgba(60,10,55,.08),0 1px 3px rgba(60,10,55,.04);
  overflow:hidden;}}

/* ── panel split ── */
.split{{display:grid;grid-template-columns:1fr 1fr;}}

/* ── left: QR ── */
.left{{
  background:linear-gradient(160deg,#1a0820 0%,#3a1035 50%,#5c1a50 100%);
  padding:28px 22px 24px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:14px;position:relative;overflow:hidden;
}}
.left::before{{content:'';position:absolute;top:-60px;left:-60px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(156,87,137,.35),transparent 70%);}}
.left-tag{{font-size:.62em;font-weight:600;letter-spacing:.14em;
  color:rgba(255,255,255,.45);text-transform:uppercase;}}
#qrimg{{
  width:160px;height:160px;border-radius:10px;
  box-shadow:0 4px 20px rgba(0,0,0,.5),0 0 0 1px rgba(255,255,255,.08);
  display:block;
}}
.qr-missing{{
  width:160px;height:160px;border-radius:10px;
  background:rgba(255,255,255,.06);border:1px dashed rgba(255,255,255,.15);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-size:.75em;color:rgba(255,255,255,.45);text-align:center;line-height:1.6;gap:8px;
}}
.scan-label{{font-size:.62em;color:rgba(255,255,255,.38);text-align:center;line-height:1.5;}}

/* ── right: steps ── */
.right{{padding:24px 22px;display:flex;flex-direction:column;justify-content:center;gap:0;}}
.right-head{{margin-bottom:16px;}}
.right-title{{font-size:.98em;font-weight:700;color:var(--ink);line-height:1.3;}}
.right-sub{{font-size:.72em;color:var(--muted);margin-top:4px;line-height:1.5;}}
.right-sub strong{{color:var(--accent);font-weight:600;}}

.step{{display:flex;align-items:flex-start;gap:9px;padding:7px 0;
  border-bottom:1px solid var(--line);}}
.step:last-child{{border-bottom:none;}}
.sn{{width:20px;height:20px;border-radius:6px;background:var(--accent);
  color:#fff;font-size:.66em;font-weight:700;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px;}}
.st{{font-size:.72em;color:var(--ink2);line-height:1.55;}}
.st strong{{color:var(--ink);font-weight:600;}}
.st em{{color:var(--accent);font-style:normal;font-weight:500;}}

/* ── bottom section ── */
.bottom{{padding:20px 24px;border-top:1px solid var(--line);}}

.key-label{{display:flex;align-items:center;gap:8px;
  font-size:.65em;font-weight:600;letter-spacing:.12em;
  color:var(--muted);text-transform:uppercase;margin-bottom:10px;}}
.key-label span{{flex:1;height:1px;background:var(--line);}}

.key-box{{
  background:#12071a;border-radius:10px;
  padding:13px 16px;
  font-family:var(--mono);font-size:.82em;
  letter-spacing:.15em;color:#c8a8d8;
  text-align:center;line-height:1.8;
  word-break:break-all;cursor:copy;user-select:all;
  border:1px solid rgba(156,87,137,.18);
  transition:background .2s,border-color .2s;
  position:relative;
}}
.key-box:hover{{background:#1a0a26;border-color:rgba(156,87,137,.38);}}
.key-box::after{{
  content:'COPIED!';position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;
  border-radius:10px;background:#9c5789;
  color:#fff;font-family:var(--sans);font-size:.8em;font-weight:700;letter-spacing:.12em;
  opacity:0;transition:opacity .15s;pointer-events:none;
}}
.key-box.copied::after{{opacity:1;}}

.key-hint{{text-align:center;font-size:.62em;color:var(--muted);margin-top:6px;}}

/* ── save-hint collapsible ── */
.save-details{{margin-top:14px;}}
.save-summary{{
  display:flex;align-items:center;gap:7px;
  font-size:.67em;font-weight:500;color:var(--muted);
  cursor:pointer;list-style:none;user-select:none;
  padding:6px 0;
}}
.save-summary::-webkit-details-marker{{display:none;}}
.save-summary::before{{
  content:'▶';font-size:.7em;color:var(--accent);
  transition:transform .2s;display:inline-block;
}}
details[open] .save-summary::before{{transform:rotate(90deg);}}
.save-body{{
  margin-top:8px;background:#0f0714;border-radius:8px;
  padding:12px 14px;font-family:var(--mono);font-size:.68em;
  color:#a890b8;line-height:1.8;border:1px solid rgba(156,87,137,.15);
  animation:fadeIn .2s ease;
}}
.save-body .cm{{color:#5a8a5a;}}
.save-body .key{{color:#9c5789;}}
.save-body .val{{color:#c8a8d8;}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(-4px)}}to{{opacity:1;transform:none}}}}

</style>
</head>
<body>
<div class="w">

  <!-- topbar -->
  <div class="topbar">
    <span class="logo">MTRAX</span>
    <span class="badge"><span class="badge-dot"></span>SETUP 2FA</span>
  </div>

  <div class="card">

    <!-- split panel -->
    <div class="split">

      <!-- LEFT — QR -->
      <div class="left">
        <span class="left-tag">Scan QR Code</span>
        {qr_html}
        {qr_note}
        <p class="scan-label">Google Authenticator<br>Authy · Microsoft Auth</p>
      </div>

      <!-- RIGHT — Steps -->
      <div class="right">
        <div class="right-head">
          <div class="right-title">Aktivasi 2FA</div>
          <div class="right-sub">Halo <strong>{username}</strong>,<br>ikuti langkah berikut.</div>
        </div>

        <div class="step">
          <div class="sn">1</div>
          <div class="st">Buka <strong>Google Authenticator</strong> atau Authy.</div>
        </div>
        <div class="step">
          <div class="sn">2</div>
          <div class="st">Ketuk <strong>"+"</strong> → <em>Scan QR</em> atau <em>Enter key</em>.</div>
        </div>
        <div class="step">
          <div class="sn">3</div>
          <div class="st">Nama akun: <strong>admin@MTRAX</strong>, tipe: <em>Time-based</em>.</div>
        </div>
        <div class="step">
          <div class="sn">4</div>
          <div class="st">Masukkan <strong>6 digit kode</strong> dari app di bawah.</div>
        </div>
      </div>

    </div>

    <!-- BOTTOM — secret key + hidden save hint -->
    <div class="bottom">

      <div class="key-label"><span></span>atau gunakan kunci manual<span></span></div>
      <div class="key-box" id="kbox" onclick="copyKey(this)" title="Klik untuk copy">{fmt_secret}</div>
      <div class="key-hint" id="khint">Klik untuk menyalin kunci</div>

      <!-- Hidden: cara simpan secret -->
      <details class="save-details">
        <summary class="save-summary">Cara menyimpan secret agar tidak reset saat restart</summary>
        <div class="save-body">
          <span class="cm"># Tambahkan ke .streamlit/secrets.toml:</span><br>
          <span class="key">[totp]</span><br>
          <span class="key">{username}</span> = <span class="val">"{secret}"</span>
        </div>
      </details>

    </div>

  </div>
</div>

<script>
function copyKey(el) {{
  const text = el.textContent.replace(/\\s+/g,'').trim();
  navigator.clipboard.writeText(text).then(function() {{
    el.classList.add('copied');
    document.getElementById('khint').textContent = '✓ Tersalin ke clipboard!';
    setTimeout(function() {{
      el.classList.remove('copied');
      document.getElementById('khint').textContent = 'Klik untuk menyalin kunci';
    }}, 1800);
  }}).catch(function() {{
    document.getElementById('khint').textContent = 'Pilih semua teks lalu Ctrl+C';
  }});
}}
</script>
</body>
</html>"""

    _components.html(html_content, height=540, scrolling=False)

    # ── OTP input styling ──────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    div[data-testid="stTextInput"] > label {
        font-family: 'Sora', sans-serif !important;
        font-size: 0.75em !important;
        font-weight: 600 !important;
        color: #8a7a8a !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #e0d0e0 !important;
        font-size: 1.4em !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.40em !important;
        text-align: center !important;
        padding: 14px 12px !important;
        background: #fdf7fc !important;
        color: #3a1a3a !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #9c5789 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
        background: #ffffff !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: #d0b8d0 !important; letter-spacing: 0.30em !important; }
    </style>
    """, unsafe_allow_html=True)

    otp_input = st.text_input(
        "Kode OTP dari Google Authenticator",
        placeholder="● ● ● ● ● ●",
        max_chars=6,
        key="setup_otp_field"
    )

    col_v, col_c = st.columns([3, 2])
    with col_v:
        if st.button("✅  Verifikasi & Masuk", use_container_width=True, type="primary"):
            if _totp_valid(secret, otp_input):
                st.session_state.totp_enrolled[username] = True
                _finish_login(username)
            else:
                st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")
    with col_c:
        if st.button("← Kembali", use_container_width=True):
            st.session_state.pending_2fa  = False
            st.session_state.pending_user = ""
            st.rerun()


def twofa_verify_page(username: str):
    """Halaman verifikasi OTP untuk login berikutnya — premium redesign."""
    import streamlit.components.v1 as _components
    st.set_page_config(page_title="Verifikasi 2FA | MTRAX", layout="centered")

    totp_secrets = _load_totp_secrets()
    secret       = totp_secrets[username]
    remaining    = 30 - (int(time.time()) % 30)
    progress_pct = int((remaining / 30) * 100)

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'DM Sans', sans-serif;
  background: #f7f4f9;
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 24px 16px;
}}

.shell {{
  width: 100%;
  max-width: 440px;
  animation: fadeUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
}}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(18px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

.brand-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}}
.brand-name {{
  font-size: 0.78em; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: #9c5789;
}}
.step-pill {{
  background: #f0e6ee; color: #9c5789;
  font-size: 0.70em; font-weight: 600;
  letter-spacing: 0.08em;
  padding: 4px 12px; border-radius: 100px;
}}

.card {{
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 4px 32px rgba(100,40,90,0.10), 0 1px 4px rgba(100,40,90,0.06);
  overflow: hidden;
}}

.card-header {{
  background: linear-gradient(135deg, #1e0a1a 0%, #5a1a4a 55%, #9c5789 100%);
  padding: 28px 32px 24px;
  position: relative;
  overflow: hidden;
}}
.card-header::before {{
  content: '';
  position: absolute; top: -50px; right: -50px;
  width: 180px; height: 180px; border-radius: 50%;
  background: rgba(255,255,255,0.06);
}}
.header-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 100px; padding: 5px 14px;
  font-size: 0.72em; font-weight: 600;
  color: rgba(255,255,255,0.88);
  letter-spacing: 0.06em; margin-bottom: 14px;
}}
.header-badge .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: #a8ff78;
  box-shadow: 0 0 6px #a8ff78;
  animation: blink 2s ease infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}

.header-title {{
  font-size: 1.25em; font-weight: 700; color: #fff;
  margin-bottom: 6px; position: relative; z-index: 1;
}}
.header-sub {{
  font-size: 0.80em; color: rgba(255,255,255,0.68);
  line-height: 1.6; position: relative; z-index: 1;
}}
.header-sub strong {{ color: rgba(255,255,255,0.95); }}
.header-sub em {{ color: #e0b8d8; font-style: normal; font-weight: 500; }}

/* ── Icon lock ─────────────────────────────── */
.lock-wrap {{
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 14px;
}}
.lock-circle {{
  width: 64px; height: 64px; border-radius: 20px;
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.8em;
}}

.card-body {{
  padding: 28px 32px;
}}

/* ── Timer ─────────────────────────────────── */
.timer-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fdf7fc;
  border: 1px solid #edd9e8;
  border-radius: 12px;
  padding: 12px 18px;
  margin-bottom: 20px;
  gap: 12px;
}}
.timer-label {{
  font-size: 0.76em;
  color: #9c8fa0;
  font-weight: 500;
}}
.timer-val {{
  font-family: 'DM Mono', monospace;
  font-size: 1.0em;
  font-weight: 600;
  color: #9c5789;
  min-width: 28px;
  text-align: right;
}}
.timer-bar-wrap {{
  flex: 1;
  height: 4px;
  background: #edd9e8;
  border-radius: 2px;
  overflow: hidden;
}}
.timer-bar {{
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, #9c5789, #c983af);
  width: {progress_pct}%;
  transition: width 1s linear;
}}

/* ── Hint ──────────────────────────────────── */
.hint-box {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: #f7f4f9;
  border-radius: 10px;
  padding: 12px 16px;
}}
.hint-icon {{
  font-size: 1.1em;
  flex-shrink: 0;
  margin-top: 1px;
}}
.hint-text {{
  font-size: 0.78em;
  color: #6a5a6a;
  line-height: 1.6;
}}
.hint-text strong {{ color: #3a1a3a; }}
.hint-text em {{ color: #9c5789; font-style: normal; font-weight: 500; }}

</style>
</head>
<body>
<div class="shell">

  <div class="brand-bar">
    <span class="brand-name">MTRAX</span>
    <span class="step-pill">VERIFIKASI 2FA</span>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="lock-wrap">
        <div class="lock-circle">🔐</div>
      </div>
      <div class="header-badge"><span class="dot"></span>Autentikasi Dua Langkah</div>
      <div class="header-title">Masukkan Kode OTP</div>
      <div class="header-sub">
        Halo <strong>{username}</strong>, password sudah benar!<br>
        Buka <strong>Google Authenticator</strong> dan masukkan kode 6 digit untuk akun <em>MTRAX:{username}</em>.
      </div>
    </div>

    <div class="card-body">
      <!-- Timer -->
      <div class="timer-row">
        <span class="timer-label">⏱ Kode berikutnya dalam</span>
        <div class="timer-bar-wrap"><div class="timer-bar"></div></div>
        <span class="timer-val">{remaining}s</span>
      </div>

      <!-- Hint -->
      <div class="hint-box">
        <span class="hint-icon">💡</span>
        <div class="hint-text">
          Buka <strong>Google Authenticator</strong> &rarr; cari akun <em>MTRAX:{username}</em> &rarr; masukkan <strong>6 digit kode</strong> yang ditampilkan.
        </div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""

    _components.html(html_content, height=420, scrolling=False)

    st.markdown("""
    <style>
    div[data-testid="stTextInput"] label {
        font-size: 0.82em !important;
        font-weight: 600 !important;
        color: #5a4a5a !important;
        letter-spacing: 0.04em;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #e0d0e0 !important;
        font-size: 1.3em !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.35em !important;
        text-align: center !important;
        padding: 14px !important;
        background: #fdf7fc !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #9c5789 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    otp_input = st.text_input(
        "Kode OTP (6 digit)",
        placeholder="● ● ● ● ● ●",
        max_chars=6,
        key="verify_otp_field"
    )

    col_v, col_c = st.columns([3, 2])
    with col_v:
        if st.button("✅  Verifikasi & Masuk", use_container_width=True, type="primary"):
            if _totp_valid(secret, otp_input):
                _finish_login(username)
            else:
                st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")
                time.sleep(1)
    with col_c:
        if st.button("← Kembali", use_container_width=True):
            st.session_state.pending_2fa  = False
            st.session_state.pending_user = ""
            st.rerun()


def _finish_login(username: str):
    """Selesaikan proses login setelah OTP berhasil diverifikasi."""
    st.session_state.login_attempts = 0
    st.session_state.lockout_until  = 0
    st.session_state.authenticated  = True
    st.session_state.logged_in      = True
    st.session_state.username       = username
    st.session_state.role           = USERS[username]["role"]
    st.session_state.login_time     = time.time()
    st.session_state.pending_2fa    = False
    st.session_state.pending_user   = ""
    st.success("✅ Login berhasil! Selamat datang.")
    time.sleep(0.5)
    st.rerun()


# ======================================
# LOGIN FUNCTION (SECURE + ANTI BRUTE FORCE)
# ── DIMODIFIKASI: setelah password OK → redirect ke 2FA ──
# ======================================
def login_page():
    """Halaman login dengan desain minimalis + advanced security"""
    st.set_page_config(page_title="Login | MTRAX", layout="centered")

    # Cek apakah sedang dalam masa lockout
    current_time = time.time()
    if current_time < st.session_state.lockout_until:
        remaining = int(st.session_state.lockout_until - current_time)
        st.error(f"Too many failed attempts. Try again in {remaining} seconds.")
        st.stop()

    st.markdown(
        """
        <style>
        * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif; }
        .stApp { background: #fafafa; }
        #MainMenu, footer, header { visibility: hidden; }
        .login-container {
            background: white;
            padding: 40px 40px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            max-width: 300px;
            margin: 6rem auto;
            border-top: 3px solid #9c5789;
        }
        .login-logo { text-align: center; margin-bottom: 40px; }
        .login-logo img { width: 70px; opacity: 0.9; }
        .app-name { font-size: 2em; font-weight: 600; color: #9c5789; margin: 20px 0 8px 0; }
        .app-subtitle { color: #888888; font-size: 0.9em; }
        .stTextInput > div > div > input {
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            padding: 12px;
            font-size: 0.95em;
        }
        .stTextInput > div > div > input:focus {
            border-color: #9c5789;
            box-shadow: 0 0 0 1px #9c5789;
        }
        .stButton > button {
            background: #9c5789;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 12px;
            font-weight: 500;
            width: 100%;
        }
        .stButton > button:hover { background: #8a4d78; }
        </style>
        """,
        unsafe_allow_html=True
    )

    logo_path = os.path.join("assets", "LOGO M FIX.png")

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    logo_base64 = get_base64_image(logo_path)

    st.markdown(f"""
        <div class="login-container">
            <div class="login-logo">
                <img src="data:image/png;base64,{logo_base64}" />
                <div class="app-name">MTRAX</div>
                <div class="app-subtitle">Travel Analytics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    username = st.text_input("", placeholder="Username", label_visibility="collapsed")
    password = st.text_input("", type="password", placeholder="Password", label_visibility="collapsed")

    if st.button("Sign In", use_container_width=True):

        # Validasi format username
        if not validate_username(username):
            st.error("Invalid username format.")
            return

        if username in USERS:

            if secure_compare(password, USERS[username]["password"]):

                # ── PASSWORD BENAR → Lanjut ke 2FA ─────────────────────────
                st.session_state.login_attempts = 0
                st.session_state.lockout_until  = 0
                st.session_state.pending_2fa    = True
                st.session_state.pending_user   = username
                st.session_state.pending_role   = USERS[username]["role"]
                st.rerun()

            else:
                st.session_state.login_attempts += 1

        else:
            st.session_state.login_attempts += 1

        # Jika gagal
        if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
            st.session_state.lockout_until = time.time() + LOCKOUT_SECONDS
            st.session_state.login_attempts = 0
            st.error("Too many failed attempts. Account temporarily locked.")
        else:
            remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
            st.error(f"Login failed. {remaining} attempt(s) remaining.")



# ======================================
# MAIN APP
# ======================================
def main_app():
    """Main application"""
    st.set_page_config(
        page_title="MTRAX | Travel Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS - Minimalist Theme with #9c5789
    st.markdown("""
        <style>
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif;
        }
        
        .stApp { 
            background: #fafafa;
        }
        
        #MainMenu, footer, header { visibility: hidden; }
        
        /* NEWS TICKER */
        .news-ticker {
            background: #9c5789;
            color: white;
            padding: 8px 0;
            font-size: 0.85em;
            overflow: hidden;
            white-space: nowrap;
            position: relative;
            margin: -60px -60px 0 -60px;
        }
        
        .news-ticker:hover .ticker-content {
            animation-play-state: paused;
        }
        
        .ticker-content {
            display: inline-block;
            padding-left: 100%;
            animation: ticker 70s linear infinite;
        }
        
        .news-item {
            color: #ffffff !important;
            text-decoration: none;
            margin-right: 40px;
            cursor: pointer;
            transition: opacity 0.2s ease;
        }
        
        .news-item:hover {
            opacity: 0.8;
            text-decoration: underline;
        }
        
        @keyframes ticker {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        
        /* HEADER */
        .header {
            background: white;
            padding: 25px 40px;
            border-bottom: 1px solid #e0e0e0;
            margin: 0 -60px 30px -60px;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-title {
            font-size: 1.8em;
            font-weight: 600;
            color: #9c5789;
        }
        
        .header-subtitle {
            font-size: 0.9em;
            color: #888888;
            margin-top: 4px;
        }
        
        .header-user {
            text-align: right;
        }
        
        .user-name {
            font-size: 0.95em;
            color: #1a1a1a;
            font-weight: 500;
        }
        
        .user-role {
            font-size: 0.85em;
            color: #888888;
            margin-top: 2px;
        }
        
        /* METRICS */
        .metric-box {
            background: white;
            padding: 20px;
            border-radius: 4px;
            border-left: 3px solid #9c5789;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        /* STATS CARDS */
        .stats-card {
            background: white;
            padding: 25px;
            border-radius: 4px;
            text-align: center;
            height: 100%;
        }
        
        .stats-number {
            font-size: 2.5em;
            font-weight: 600;
            color: #9c5789;
            margin: 15px 0;
        }
        
        .stats-label {
            font-size: 0.85em;
            color: #888888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stats-detail {
            font-size: 0.9em;
            color: #888888;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #f0f0f0;
        }
        
        /* SECTION */
        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #1a1a1a;
            margin: 30px 0 20px 0;
        }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: white;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            font-weight: 500;
            color: #888888;
            border-bottom: 2px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #1a1a1a;
        }
        
        .stTabs [aria-selected="true"] {
            color: #9c5789;
            border-bottom-color: #9c5789;
            background: transparent;
        }
        
        /* BUTTONS */
        .stButton > button {
            border-radius: 4px;
            font-weight: 500;
            padding: 10px 20px;
        }
        
        .stButton > button[kind="primary"] {
            background: #9c5789;
            color: white;
            border: none;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: #8a4d78;
        }
        
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: white;
            border-right: 1px solid #e0e0e0;
        }
        
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #9c5789;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* METRIC FROM STREAMLIT */
        .stMetric {
            background: white;
            padding: 18px;
            border-radius: 4px;
            border-left: 3px solid #9c5789;
        }
        
        .stMetric label {
            color: #888888 !important;
            font-size: 0.8em !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            font-size: 1.8em !important;
        }
        
        /* PROGRESS */
        .stProgress > div > div {
            background: #9c5789;
        }
        
        /* DIVIDER */
        .divider {
            height: 1px;
            background: #e0e0e0;
            margin: 30px 0;
        }
        
        /* SELECTBOX */
        .stSelectbox > div > div {
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # NEWS TICKER
    news_data = fetch_rss_news()
    render_news_ticker(news_data)

    # HEADER
    greet, now = get_greeting()
    role_display = st.session_state.get("role", "User")
    username_display = st.session_state.get("username", "User")

    st.markdown(f"""
        <div class='header'>
            <div class='header-content'>
                <div>
                    <div class='header-title'>MTRAX</div>
                    <div class='header-subtitle'>Corporate Travel Analytics</div>
                </div>
                <div class='header-user'>
                    <div class='user-name'>{greet}, {username_display.title()}</div>
                    <div class='user-role'>{role_display} · {now.strftime('%d %b %Y')}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ======================================
    # SIDEBAR
    # ======================================
    def load_logo_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    logo_base64 = load_logo_base64("assets/LOGO M FIX.png")

    with st.sidebar:
        st.markdown(f"""
            <div style='text-align:center;padding:20px 0;'>
                <img src='data:image/png;base64,{logo_base64}' width='60'/>
            </div>
        """, unsafe_allow_html=True)

        # ── 2FA Status Badge di Sidebar ──────────────────────────────
        totp_secrets  = _load_totp_secrets()
        _uname_side   = st.session_state.get("username", "")
        _enrolled     = st.session_state.totp_enrolled.get(_uname_side, False)
        try:
            _from_toml = bool(st.secrets["totp"][_uname_side])
        except Exception:
            _from_toml = False
        _2fa_active = _enrolled or _from_toml
        _badge_color = "#3dab7a" if _2fa_active else "#e05a2b"
        _badge_text  = "Active ✅" if _2fa_active else "Setup Required"
        st.markdown(f"""
        <div style='background:white;border-radius:6px;padding:10px 14px;
                    border-left:3px solid {_badge_color};margin-bottom:4px;
                    font-size:0.78em;'>
            <div style='color:#888;font-size:0.85em;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:3px;'>🔐 2FA Status</div>
            <div style='color:{_badge_color};font-weight:600;'>{_badge_text}</div>
            <div style='color:#bbb;font-size:0.88em;'>Google Authenticator · TOTP</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Drive options
        drive_options = {
            "2023": "1xDFRdGLDiiScIwW9gTucRyeFCmuqNyq_",
            "2024": "16ZMZ42BLN4GPbYKAd5h75ocbxFuyc85V",
            "2025": "1chxbGHfk9hHNPZ8vlU6AqRVUKH1jEnxF",
            "2026": "14CbafYeVrKUXWBE1LPUFlRXHeXGXAaO4",
        }
        
        selected_period = st.selectbox(
            "Select Data Period",
            options=list(drive_options.keys()),
            help="Choose which year's data to load"
        )
        
        if st.button("Get Data", use_container_width=True, type="primary"):
            with st.spinner(f"Loading {selected_period} data..."):
                progress_bar = st.progress(0)
                try:
                    folder_id = drive_options[selected_period]
                    drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                    
                    progress_bar.progress(20)
                    loaded_data = load_drive_data(folder_id, drop_cols)
                    
                    progress_bar.progress(100)
                    st.session_state.df_all = loaded_data
                    st.session_state.data_loaded = True
                    st.session_state.data_period = selected_period
                    
                    st.success(f"✅ Loaded {len(loaded_data):,} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        uploaded_files = st.file_uploader(
            "Upload Excel Files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            help="Select multiple Excel files"
        )

        if uploaded_files:
            if st.button("Process Uploaded Files", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        dfs = []
                        drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                        
                        for file in uploaded_files:
                            df = pd.read_excel(file)
                            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                            dfs.append(df)
                        
                        st.session_state.df_all = pd.concat(dfs, ignore_index=True)
                        st.session_state.data_loaded = True
                        st.session_state.data_period = "Manual Upload"
                        
                        st.success(f"✅ Processed {len(uploaded_files)} files")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Data status
        if st.session_state.data_loaded and not st.session_state.df_all.empty:
            period_info = f" ({st.session_state.data_period})" if st.session_state.data_period else ""
            st.success(f"✅ {len(st.session_state.df_all):,} records{period_info}")
        else:
            st.info("ℹ️ No data loaded")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # BMKG Info
        st.markdown("### 🌏 BMKG INFO")
        
        quake_data = get_bmkg_realtime_quake()
        
        if quake_data:
            st.markdown(f"""
                <div style='background:white;padding:12px;border-radius:4px;border-left:3px solid #9c5789;'>
                    <div style='font-weight:600;color:#9c5789;font-size:0.85em;'>Latest Earthquake</div>
                    <div style='font-size:0.8em;color:#666;margin-top:8px;'>
                        📍 {quake_data.get('Wilayah', 'N/A')}<br>
                        📊 M {quake_data.get('Magnitude', 'N/A')}<br>
                        🕐 {quake_data.get('Tanggal', 'N/A')} {quake_data.get('Jam', 'N/A')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No recent data")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Logout
        if st.button("Logout", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ======================================
    # MAIN CONTENT
    # ======================================
    df_all = st.session_state.df_all

    if not df_all.empty:
        # Helper functions
        def prepare_monthly_trend(df, date_col, value_col, agg="count"):
            if date_col not in df.columns:
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            df_copy = df_copy.dropna(subset=[date_col])
            df_copy["YearMonth"] = df_copy[date_col].dt.to_period("M").astype(str)

            if agg == "count":
                trend = df_copy.groupby("YearMonth")[value_col].count().reset_index(name="Value")
            elif agg == "nunique":
                trend = df_copy.groupby("YearMonth")[value_col].nunique().reset_index(name="Value")
            elif agg == "sum":
                trend = df_copy.groupby("YearMonth")[value_col].sum().reset_index(name="Value")
            else:
                trend = df_copy.groupby("YearMonth").size().reset_index(name="Value")

            return trend

    # ======================================
    # DATA CLEANING
    # ======================================
    if not df_all.empty:
        # Bersihkan Invoice Amount
        if "Invoice Amount" in df_all.columns:
            df_all["Invoice Amount"] = pd.to_numeric(
                df_all["Invoice Amount"].astype(str).str.replace("$", "").str.replace(",", ""),
                errors="coerce"
            )
        
        # Bersihkan Number of Rooms Night
        if "Number of Rooms Night" in df_all.columns:
            df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors="coerce")
        
        # Convert dates
        if "Check in Date" in df_all.columns:
            df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce")
        
        if "Check out Date" in df_all.columns:
            df_all["Check out Date"] = pd.to_datetime(df_all["Check out Date"], errors="coerce")
        
        if "Issue Time" in df_all.columns:
            df_all["Issue Time"] = pd.to_datetime(df_all["Issue Time"], errors="coerce")

            # Mapping Company Code -> Nama Perusahaan
        company_map = {
            "1010": "PT Pertamina (Persero)",
            "2022": "PT Pertamina Geothermal Energy",
            "2033": "PT Pertamina Trans Kontinental",
            "2034": "PT Pelita Air Service",
            "2042": "PT Pertamina Retail",
            "2059": "PT Pertamina Port And Logistics",
            "2061": "PT Pertamina Energy Terminal",
            "2119": "PT Nusantara Regas",
            "2138": "PT Pertamina Lubricants",
            "2147": "PT Pertamina International Shipping",
            "2151": "PT Pertamina International EP",
            "2183": "PT Pertamina Power Indonesia",
            "2186": "PT Kilang Pertamina International",
            "2205": "PT Kilang Pertamina Balikpapan",
            "2222": "PT Pertamina Patra Niaga",
            "5000": "PT Pertamina Hulu Energi",
            "2060": "PT Pertamina Marine Solutions",
            "2062": "PT Pertamina Marine Engineering",
            "2110": "PT Pertamina Drilling Services Indonesia ",
            "2052": "PT Pertamina Maintenance and Construction",
        }

        if "Company Code" in df_all.columns:
            df_all["Company Code"] = df_all["Company Code"].astype(str).str.strip()
            df_all["Nama Perusahaan"] = df_all["Company Code"].map(company_map).fillna("Lainnya / Unknown")

        # ======================================
        # AUTO-NORMALISASI
        # ======================================

            df_all, hotel_mapping = auto_canonical_hotel_mapping(
                df_all,
                hotel_col="Hotel Name",
                threshold=0.88
            )

        # ======================================
        # GLOBAL CITY STANDARDIZATION
        # ======================================

        if "City" in df_all.columns:
            df_all["City"] = (
                df_all["City"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .str.title()
            )
            df_all.loc[df_all["City"] == "Nan", "City"] = np.nan

        if "City Destination" in df_all.columns:
            df_all["City Destination"] = (
                df_all["City Destination"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
                .str.title()
            )
            df_all.loc[df_all["City Destination"] == "Nan", "City Destination"] = np.nan

        # ======================================
        # TAB SEMUA
        # ======================================

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Value Creation",
            "Dashboard",
            "Explorer",
            "CRM",
            "Network",
            "Price Intelligence",
            "Sankey Flow",
            "Top Hotel/City",
            "Dendrogram",
            "Export"
        ])

                # TAB 1: STRATEGIC VALUE CREATION
        # ======================================
        with tab1:

            import streamlit.components.v1 as components

            components.html("""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    background: #fafafa;
    color: #1a1a1a;
    padding: 8px 4px 40px;
    -webkit-font-smoothing: antialiased;
  }

  :root {
    --purple : #9c5789;
    --purp-l : rgba(156,87,137,0.08);
    --purp-b : rgba(156,87,137,0.22);
    --black  : #1a1a1a;
    --grey   : #888888;
    --grey-l : #e0e0e0;
    --white  : #fafafa;
    --surf   : #ffffff;
    --bdr    : #e0e0e0;
  }

  /* ── Page Header ── */
  .hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 18px; border-bottom: 1px solid var(--bdr);
    margin-bottom: 28px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .mark {
    width: 34px; height: 34px; background: var(--purple);
    border-radius: 6px; display: inline-flex;
    align-items: center; justify-content: center;
    font-family: 'Geist Mono', monospace;
    font-size: 11px; font-weight: 600; color: #fff; letter-spacing: .04em;
    flex-shrink: 0;
  }
  .brand-title {
    font-size: 17px; font-weight: 600; color: var(--black);
    letter-spacing: -.02em; line-height: 1.2; margin: 0;
  }
  .brand-sub { font-size: 12px; color: var(--grey); margin: 2px 0 0; }
  .hdr-right { display: flex; align-items: center; gap: 8px; }
  .mod-tag {
    font-family: 'Geist Mono', monospace; font-size: 10px;
    color: var(--grey); background: var(--grey-l);
    padding: 4px 10px; border-radius: 4px; letter-spacing: .08em;
  }
  .live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid var(--bdr); border-radius: 100px;
    padding: 4px 12px; font-size: 11px; color: var(--grey);
    background: var(--surf);
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--purple); display: inline-block;
    animation: blink 2.4s ease infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ── Eyebrow ── */
  .eyebrow {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Geist Mono', monospace; font-size: 10px; font-weight: 500;
    color: var(--grey); letter-spacing: .18em;
    text-transform: uppercase; margin-bottom: 14px;
  }
  .eyebrow-ln { flex: 1; height: 1px; background: var(--bdr); }

  /* ── 2×2 Pillar Grid ── */
  .pgrid {
    display: grid; grid-template-columns: 1fr 1fr;
    border: 1px solid var(--bdr); border-radius: 10px;
    overflow: hidden; background: var(--bdr); gap: 1px;
    margin-bottom: 1px;
  }
  .pcard {
    background: var(--surf); padding: 26px 24px;
    position: relative; transition: background .18s ease;
  }
  .pcard:hover { background: #fdfdfd; }
  .pcard::before {
    content: ''; position: absolute;
    top: 26px; bottom: 26px; left: 0; width: 2px;
    background: var(--purple); opacity: 0; transition: opacity .2s ease;
  }
  .pcard:hover::before { opacity: 1; }
  .card-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; margin-bottom: 14px;
  }
  .card-icon { font-size: 18px; line-height: 1; }
  .badge {
    font-family: 'Geist Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: .12em;
    color: var(--purple); background: var(--purp-l);
    border: 1px solid var(--purp-b);
    padding: 3px 8px; border-radius: 3px;
  }
  .pcard h3 {
    font-size: 14px; font-weight: 600; color: var(--black);
    letter-spacing: -.015em; margin-bottom: 4px; line-height: 1.3;
  }
  .obj { font-size: 12px; color: var(--grey); margin-bottom: 18px; line-height: 1.55; }
  .fl {
    font-family: 'Geist Mono', monospace; font-size: 9px; font-weight: 500;
    letter-spacing: .15em; text-transform: uppercase;
    color: var(--purple); margin-bottom: 8px;
  }
  .dlist { display: flex; flex-direction: column; gap: 5px; margin-bottom: 14px; }
  .drow  { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--black); }
  .ddot  {
    width: 4px; height: 4px; border-radius: 50%;
    background: var(--purple); flex-shrink: 0; display: inline-block;
  }
  .sep  { height: 1px; background: var(--grey-l); margin: 12px 0; }
  .ilist { display: flex; flex-direction: column; gap: 5px; }
  .irow {
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 11.5px; color: var(--grey); line-height: 1.5;
  }
  .iarr { font-size: 8px; margin-top: 5px; flex-shrink: 0; color: var(--purple); opacity: .7; }

  /* ── Card 5 ── */
  .c5wrap {
    border: 1px solid var(--bdr); border-top: none;
    border-radius: 0 0 10px 10px; overflow: hidden; margin-bottom: 32px;
  }
  .c5hd {
    background: #f5f5f5; border-bottom: 1px solid var(--bdr);
    padding: 16px 24px; display: flex;
    align-items: center; justify-content: space-between;
  }
  .c5hd-l { display: flex; align-items: center; gap: 10px; }
  .c5hd h3 { font-size: 13px; font-weight: 600; color: var(--black); margin: 0; }
  .c5hd .c5sub { font-size: 11px; color: var(--grey); }
  .c5body {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    background: var(--bdr); gap: 1px;
  }
  .c5col { background: var(--surf); padding: 20px 24px; }

  /* ── Executive Summary ── */
  .exec { border: 1px solid var(--bdr); border-radius: 10px; overflow: hidden; }
  .exec-hd {
    background: var(--black); padding: 22px 26px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .exec-hd h2 {
    font-size: 16px; font-weight: 600; color: var(--white);
    letter-spacing: -.02em; margin: 0;
  }
  .exec-hd .exec-sub { font-size: 11.5px; color: #888; margin: 3px 0 0; }
  .exec-tag {
    font-family: 'Geist Mono', monospace; font-size: 9.5px;
    letter-spacing: .12em; color: var(--purple);
    background: rgba(156,87,137,.15); border: 1px solid rgba(156,87,137,.3);
    padding: 4px 11px; border-radius: 100px; text-transform: uppercase;
    white-space: nowrap;
  }
  .pillars {
    display: grid; grid-template-columns: repeat(4,1fr);
    background: var(--bdr); gap: 1px;
    border-bottom: 1px solid var(--bdr);
  }
  .pillar {
    background: var(--surf); padding: 18px 18px 16px; position: relative;
  }
  .pillar::after {
    content: ''; position: absolute;
    bottom: 0; left: 18px; right: 18px; height: 1px;
    background: var(--purple); opacity: 0; transition: opacity .2s;
  }
  .pillar:hover::after { opacity: .4; }
  .p-icon { font-size: 16px; margin-bottom: 8px; display: block; }
  .pillar p { font-size: 12px; color: var(--black); font-weight: 500; line-height: 1.45; margin: 0; }
  .flow-bar {
    background: var(--surf); padding: 18px 24px;
    display: flex; align-items: center; justify-content: center;
    border-top: 1px solid var(--bdr);
  }
  .fn { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .fn-lbl { font-size: 12px; font-weight: 600; color: var(--black); letter-spacing: -.01em; }
  .fn-lbl.hi { color: var(--purple); }
  .fn-sub {
    font-family: 'Geist Mono', monospace; font-size: 9px;
    color: var(--grey); letter-spacing: .06em;
  }
  .fsep { display: flex; align-items: center; margin: 0 18px; }
  .fsep-ln { width: 28px; height: 1px; background: var(--grey-l); }
  .fsep-arr { font-size: 9px; color: var(--purple); opacity: .6; }
</style>
</head>
<body>

  <!-- ── Page Header ── -->
  <div class="hdr">
    <div class="brand">
      <div class="mark">MTX</div>
      <div>
        <p class="brand-title">Strategic Value Creation Framework</p>
        <p class="brand-sub">Travel Analytics &amp; Procurement Intelligence</p>
      </div>
    </div>
    <div class="hdr-right">
      <span class="mod-tag">MODULE 08</span>
      <span class="live-badge"><span class="live-dot"></span>5 Value Pillars</span>
    </div>
  </div>

  <!-- ── Eyebrow ── -->
  <div class="eyebrow"><span>Value Pillars</span><div class="eyebrow-ln"></div></div>

  <!-- ── 2×2 Grid ── -->
  <div class="pgrid">

    <!-- 01 Financial -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">💰</span>
        <span class="badge">01 · FINANCIAL</span>
      </div>
      <h3>Financial Optimization</h3>
      <p class="obj">Mengurangi total travel spend dan meningkatkan efisiensi biaya operasional secara terukur.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Rate benchmarking antar hotel</div>
        <div class="drow"><span class="ddot"></span>Price per night analysis</div>
        <div class="drow"><span class="ddot"></span>Negotiation leverage berbasis volume room nights</div>
        <div class="drow"><span class="ddot"></span>Last-minute booking cost impact</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Estimasi saving 5–15% dari negotiated rate</div>
        <div class="irow"><span class="iarr">▶</span>Pengurangan overpricing hotel tidak terstandarisasi</div>
        <div class="irow"><span class="iarr">▶</span>Kontrol budget lintas perusahaan</div>
      </div>
    </div>

    <!-- 02 Operational -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">⚙️</span>
        <span class="badge">02 · OPERATIONAL</span>
      </div>
      <h3>Operational Efficiency</h3>
      <p class="obj">Meningkatkan kecepatan dan kualitas proses booking secara end-to-end.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Lead time monitoring</div>
        <div class="drow"><span class="ddot"></span>Multi-booking behavior analysis</div>
        <div class="drow"><span class="ddot"></span>Travel request pattern heatmap</div>
        <div class="drow"><span class="ddot"></span>Automation &amp; canonical hotel mapping</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Mengurangi booking mendadak (≤2 hari)</div>
        <div class="irow"><span class="iarr">▶</span>Mengurangi duplikasi nama hotel</div>
        <div class="irow"><span class="iarr">▶</span>Meningkatkan data reliability untuk reporting</div>
      </div>
    </div>

    <!-- 03 Procurement -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">🎯</span>
        <span class="badge">03 · PROCUREMENT</span>
      </div>
      <h3>Strategic Procurement Intelligence</h3>
      <p class="obj">Meningkatkan posisi tawar terhadap hotel dan vendor strategis.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Top 10 hotel concentration</div>
        <div class="drow"><span class="ddot"></span>Volume aggregation per city</div>
        <div class="drow"><span class="ddot"></span>Corporate usage clustering</div>
        <div class="drow"><span class="ddot"></span>Canonical hotel normalization</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Centralized negotiation strategy</div>
        <div class="irow"><span class="iarr">▶</span>Volume-based discount leverage</div>
        <div class="irow"><span class="iarr">▶</span>Preferred hotel program optimization</div>
      </div>
    </div>

    <!-- 04 Governance -->
    <div class="pcard">
      <div class="card-top">
        <span class="card-icon">🛡️</span>
        <span class="badge">04 · GOVERNANCE</span>
      </div>
      <h3>Risk &amp; Governance Control</h3>
      <p class="obj">Menjamin kontrol dan keamanan data travel perusahaan secara sistemik.</p>
      <div class="fl">Value Drivers</div>
      <div class="dlist">
        <div class="drow"><span class="ddot"></span>Role-based download restriction</div>
        <div class="drow"><span class="ddot"></span>Admin-only data export</div>
        <div class="drow"><span class="ddot"></span>Real-time monitoring dashboard</div>
        <div class="drow"><span class="ddot"></span>2FA Google Authenticator (TOTP)</div>
      </div>
      <div class="sep"></div>
      <div class="fl">Business Impact</div>
      <div class="ilist">
        <div class="irow"><span class="iarr">▶</span>Mencegah data leakage</div>
        <div class="irow"><span class="iarr">▶</span>Meningkatkan compliance standar</div>
        <div class="irow"><span class="iarr">▶</span>Governance berbasis sistem</div>
      </div>
    </div>

  </div>

  <!-- ── Card 5 — Predictive ── -->
  <div class="c5wrap">
    <div class="c5hd">
      <div class="c5hd-l">
        <span style="font-size:17px;">🔮</span>
        <div>
          <h3>Predictive &amp; Future Intelligence</h3>
          <span class="c5sub">Next Phase Development</span>
        </div>
      </div>
      <span class="badge">05 · PREDICTIVE</span>
    </div>
    <div class="c5body">
      <div class="c5col">
        <div class="fl">Potential Development</div>
        <div class="dlist" style="margin-top:10px;">
          <div class="drow"><span class="ddot"></span>LSTM-based demand forecasting</div>
          <div class="drow"><span class="ddot"></span>Hotel price anomaly detection</div>
          <div class="drow"><span class="ddot"></span>Traveler segmentation (KMeans)</div>
          <div class="drow"><span class="ddot"></span>Automated negotiation simulator</div>
        </div>
      </div>
      <div class="c5col" style="border-left:1px solid #e0e0e0;border-right:1px solid #e0e0e0;">
        <div class="fl">Future Business Value</div>
        <div class="ilist" style="margin-top:10px;">
          <div class="irow"><span class="iarr">▶</span>Predictive budget planning yang akurat</div>
          <div class="irow"><span class="iarr">▶</span>Early warning overpricing otomatis</div>
          <div class="irow"><span class="iarr">▶</span>Smart hotel contract recommendation</div>
        </div>
      </div>
      <div class="c5col">
        <div class="fl">Technology Stack</div>
        <div class="dlist" style="margin-top:10px;">
          <div class="drow"><span class="ddot"></span>Deep Learning / LSTM</div>
          <div class="drow"><span class="ddot"></span>Unsupervised Clustering</div>
          <div class="drow"><span class="ddot"></span>Anomaly Detection Models</div>
          <div class="drow"><span class="ddot"></span>Simulation Engine</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Executive Summary eyebrow ── -->
  <div class="eyebrow" style="margin-top:4px;">
    <span>Executive Summary</span><div class="eyebrow-ln"></div>
  </div>

  <!-- ── Executive Summary ── -->
  <div class="exec">
    <div class="exec-hd">
      <div>
        <h2>MTRAX Platform Overview</h2>
        <p class="exec-sub">Lebih dari sekadar dashboard — sistem intelijen strategis untuk travel spend.</p>
      </div>
      <span class="exec-tag">Strategic Intelligence</span>
    </div>

    <div class="pillars">
      <div class="pillar">
        <span class="p-icon">🧠</span>
        <p>Strategic Decision<br>Support System</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">⚡</span>
        <p>Negotiation<br>Intelligence Engine</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">💡</span>
        <p>Corporate Cost<br>Optimization Platform</p>
      </div>
      <div class="pillar" style="border-left:1px solid #e0e0e0;">
        <span class="p-icon">🔐</span>
        <p>Governance-Controlled<br>Analytics Ecosystem</p>
      </div>
    </div>

    <div class="flow-bar">
      <div class="fn">
        <span class="fn-lbl">Insight</span>
        <span class="fn-sub">Data → Analytics</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl">Strategy</span>
        <span class="fn-sub">Pattern → Direction</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl">Negotiation Leverage</span>
        <span class="fn-sub">Volume → Power</span>
      </div>
      <div class="fsep"><div class="fsep-ln"></div><span class="fsep-arr">›</span></div>
      <div class="fn">
        <span class="fn-lbl hi">Financial Impact</span>
        <span class="fn-sub">Cost → Savings</span>
      </div>
    </div>
  </div>

</body>
</html>
""", height=1600, scrolling=True)


        # ======================================
        # TAB 2: DASHBOARD
        # ======================================
        with tab2:

        # ======================================
        # FILTER OVERVIEW — NAMA PERUSAHAAN
        # ======================================
            company_col = "Nama Perusahaan"

            if company_col in df_all.columns:
                company_list = (
                    df_all[company_col]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                st.markdown("""
                <style>
                div[data-testid="stMultiSelect"] > label {
                    font-size: 0.85em !important;
                    color: #555 !important;
                    font-weight: 400 !important;
                    margin-bottom: 4px !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
                    border: 1px solid #e0e0e0 !important;
                    border-radius: 6px !important;
                    background: white !important;
                    min-height: 38px !important;
                    font-size: 0.875em !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
                    border-color: #9c5789 !important;
                    box-shadow: 0 0 0 2px rgba(156,87,137,0.15) !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
                    background: #9c5789 !important;
                    border-radius: 50px !important;
                    padding: 2px 10px !important;
                    font-size: 0.78em !important;
                    font-weight: 500 !important;
                    color: white !important;
                    border: none !important;
                }
                div[data-testid="stMultiSelect"] [data-baseweb="tag"] span[role="presentation"] {
                    color: rgba(255,255,255,0.75) !important;
                    font-size: 1.1em !important;
                }
                div[data-testid="stMultiSelect"] [role="option"]:hover {
                    background: rgba(156,87,137,0.08) !important;
                }
                div[data-testid="stMultiSelect"] [aria-selected="true"] {
                    background: rgba(156,87,137,0.12) !important;
                    color: #9c5789 !important;
                    font-weight: 500 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                selected_companies = st.multiselect(
                    "Filter Overview berdasarkan Nama Perusahaan",
                    options=company_list,
                    default=[],
                    placeholder="Semua perusahaan (pilih untuk filter spesifik)…"
                )

                if selected_companies:
                    df_overview = df_all[df_all[company_col].isin(selected_companies)]
                else:
                    df_overview = df_all.copy()
            else:
                st.warning("Kolom 'Nama Perusahaan' tidak ditemukan.")
                df_overview = df_all.copy()


            st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
            
            # Primary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_rows = len(df_overview)
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>Bookings</div>
                        <div class='metric-value'>{total_rows:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                if "Travel Request Number" in df_overview.columns:
                    unique_tr = df_overview["Travel Request Number"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Trvl Requests</div>
                            <div class='metric-value'>{unique_tr:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col3:
                if "Employee Id" in df_overview.columns:
                    unique_travelers = df_overview["Employee Id"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Travelers</div>
                            <div class='metric-value'>{unique_travelers:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col4:
                if "Hotel Name" in df_overview.columns:
                    unique_hotels = df_overview["Hotel Name"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Hotels</div>
                            <div class='metric-value'>{unique_hotels:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col5:
                if "Number of Rooms Night" in df_overview.columns:
                    total_nights = df_overview["Number of Rooms Night"].sum()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Room Nights</div>
                            <div class='metric-value'>{total_nights:,.0f}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Secondary Metrics
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if "Company Code" in df_overview.columns:
                    unique_company = df_overview["Company Code"].nunique()
                    st.metric("Companies", f"{unique_company:,}")

            with col2:
                if "Cost Center Pekerja" in df_overview.columns:
                    unique_cc = df_overview["Cost Center Pekerja"].nunique()
                    st.metric("Cost Centers", f"{unique_cc:,}")

            with col3:
                if "City" in df_overview.columns:
                    unique_cities = df_overview["City"].nunique()
                    st.metric("Cities", f"{unique_cities:,}")

            with col4:
                if "Country" in df_overview.columns:
                    unique_countries = df_overview["Country"].nunique()
                    st.metric("Countries", f"{unique_countries:,}")

            # Travel Request Analysis
            if "Travel Request Number" in df_overview.columns:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Travel Request Analysis</div>", unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    tr_bookings = df_overview.groupby("Travel Request Number").size()
                    avg_booking = tr_bookings.mean()
                    max_booking = tr_bookings.max()
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div class='stats-label'>Avg Bookings</div>
                            <div class='stats-number'>{avg_booking:.1f}</div>
                            <div class='stats-detail'>Max: {max_booking}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if "Number of Rooms Night" in df_overview.columns:
                        tr_nights = df_overview.groupby("Travel Request Number")["Number of Rooms Night"].sum()
                        avg_nights = tr_nights.mean()
                        max_nights = tr_nights.max()
                        
                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Avg Nights</div>
                                <div class='stats-number'>{avg_nights:.1f}</div>
                                <div class='stats-detail'>Max: {max_nights:.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    tr_with_multiple = (tr_bookings > 1).sum()
                    tr_single = (tr_bookings == 1).sum()
                    multi_percentage = (tr_with_multiple / len(tr_bookings) * 100)
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div class='stats-label'>Multi-Booking</div>
                            <div class='stats-number'>{multi_percentage:.0f}%</div>
                            <div class='stats-detail'>{tr_with_multiple:,} / {tr_single:,}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    if "Invoice Amount" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:
                        valid_rows = (df_overview["Invoice Amount"].notna()) & \
                                    (df_overview["Number of Rooms Night"].notna()) & \
                                    (df_overview["Number of Rooms Night"] > 0)
                        df_valid = df_overview[valid_rows].copy()
                        df_valid["Price Per Night"] = df_valid["Invoice Amount"] / df_valid["Number of Rooms Night"]
                        
                        avg_price = df_valid["Price Per Night"].mean()
                        
                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Avg Rate</div>
                                <div class='stats-number'>{avg_price/1000:.0f}k</div>
                                <div class='stats-detail'>Per Night</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col5:
                    date_cols = ["Issue Time", "Check in Date"]
                    for col in date_cols:
                        if col in df_overview.columns:
                            df_overview[col] = pd.to_datetime(df_overview[col], errors="coerce")

                    if "Issue Time" in df_all.columns and "Check in Date" in df_all.columns:
                        df_lead = df_overview[
                            df_overview["Issue Time"].notna() &
                            df_overview["Check in Date"].notna()
                        ].copy()

                        df_lead["Lead Time (Days)"] = (
                            df_lead["Check in Date"].dt.normalize() -
                            df_lead["Issue Time"].dt.normalize()
                        ).dt.days

                        lead_valid = df_lead[df_lead["Lead Time (Days)"] >= 0]
                        avg_lead = lead_valid["Lead Time (Days)"].mean()
                        last_minute_pct = (
                            (lead_valid["Lead Time (Days)"] <= 2).sum() / len(lead_valid) * 100
                            if len(lead_valid) > 0 else 0
                        )

                        st.markdown(f"""
                            <div class='stats-card'>
                                <div class='stats-label'>Lead Time</div>
                                <div class='stats-number'>{avg_lead:.0f}</div>
                                <div class='stats-detail'>{last_minute_pct:.0f}% last-minute</div>
                            </div>
                        """, unsafe_allow_html=True)

            if "Issue Time" in df_overview.columns and "Travel Request Number" in df_overview.columns:
                df_heat = df_overview[
                    df_overview["Issue Time"].notna() &
                    df_overview["Travel Request Number"].notna()
                ].copy()

                df_heat["Issue Hour"] = df_heat["Issue Time"].dt.hour
                df_heat["Issue Day"] = df_heat["Issue Time"].dt.day_name()

                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                pivot_core = (
                    df_heat
                    .groupby(["Issue Day", "Issue Hour"])["Travel Request Number"]
                    .nunique()
                    .reset_index(name="TR_Count")
                    .pivot(index="Issue Day", columns="Issue Hour", values="TR_Count")
                    .reindex(day_order)
                    .fillna(0)
                )

                total_day = pivot_core.sum(axis=1)
                total_hour = pivot_core.sum(axis=0)
                grand_total = total_day.sum()

                fig = px.imshow(
                    pivot_core,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#ddd", "#9c5789"]
                )

                annotations = []

                for i, day in enumerate(pivot_core.index):
                    annotations.append(dict(
                        x=len(pivot_core.columns),
                        y=i,
                        text=f"<b>{int(total_day.loc[day])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                for j, hour in enumerate(pivot_core.columns):
                    annotations.append(dict(
                        x=j,
                        y=len(pivot_core.index),
                        text=f"<b>{int(total_hour.loc[hour])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                annotations.append(dict(
                    x=len(pivot_core.columns),
                    y=len(pivot_core.index),
                    text=f"<b>{int(grand_total)}</b>",
                    showarrow=False,
                    font=dict(color="black", size=13)
                ))

                fig.update_layout(
                    title="Travel Request Heatmap (Issue Time)",
                    xaxis_title="Issue Hour",
                    yaxis_title="Issue Day",
                    annotations=annotations,
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=40, r=60, t=60, b=60),
                    font=dict(size=11)
                )

                fig.update_xaxes(range=[-0.5, len(pivot_core.columns) + 0.5])
                fig.update_yaxes(range=[len(pivot_core.index) + 0.5, -0.5])

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if "City Destination" in df_overview.columns:
                    top_cities = df_overview["City Destination"].value_counts().head(10)

                    fig_cities = px.bar(
                        x=top_cities.values,
                        y=top_cities.index,
                        orientation="h",
                        text=top_cities.values
                    )

                    fig_cities.update_traces(
                        marker_color="#9c5789",
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11)
                    )

                    fig_cities.update_layout(
                        height=380,
                        title="Top 10 Cities",
                        xaxis_title="",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=40, t=50, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig_cities, use_container_width=True)

            with col2:
                if "Country Destination" in df_overview.columns:
                    top_countries = df_overview["Country Destination"].value_counts().head(10)

                    fig_countries = px.pie(
                        values=top_countries.values,
                        names=top_countries.index,
                        hole=0.4,
                        color_discrete_sequence=["#9c5789", "#b36d9c", "#c983af", "#df99c2", "#f5afd5"]
                    )

                    fig_countries.update_layout(
                        height=380,
                        title="Top 10 Countries",
                        plot_bgcolor="white",
                        paper_bgcolor="white"
                    )

                    st.plotly_chart(fig_countries, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                if "Nama Perusahaan" in df_overview.columns and "Travel Request Number" in df_overview.columns:
                    top_dir = (
                        df_overview.groupby("Nama Perusahaan")["Travel Request Number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index(name="Travel Requests")
                    )

                    fig_dir = px.bar(
                        top_dir,
                        x="Travel Requests",
                        y="Nama Perusahaan",
                        orientation="h",
                        text="Travel Requests"
                    )

                    fig_dir.update_traces(
                        marker_color="#9c5789",
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11)
                    )

                    fig_dir.update_layout(
                        height=380,
                        title="Top 10 Directorates by Travel Requests",
                        xaxis_title="Number of Travel Requests",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=40, t=50, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig_dir, use_container_width=True)

                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                if "Issue Time" in df_overview.columns and "Travel Request Number" in df_overview.columns:

                    trend_df = prepare_monthly_trend(
                        df_overview,
                        date_col="Issue Time",
                        value_col="Travel Request Number",
                        agg="nunique"
                    )

                    fig_trend = px.line(
                        trend_df,
                        x="YearMonth",
                        y="Value",
                        markers=True,
                        labels={"Value": "Unique Travel Requests", "YearMonth": "Month"}
                    )

                    fig_trend.update_traces(
                        line=dict(color="#9c5789", width=3), 
                        marker=dict(size=8, color="#9c5789")
                    )
                    
                    fig_trend.update_layout(
                        height=380,
                        title="Monthly Travel Request Trend",
                        xaxis_title="",
                        yaxis_title="Travel Requests",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=40, r=20, t=50, b=40),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    output_tr_trend = BytesIO()
                    trend_df.to_excel(
                        output_tr_trend,
                        index=False,
                        sheet_name="Monthly Travel Request Trend"
                    )
                    output_tr_trend.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_tr_trend,
                            file_name="monthly_travel_request_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #e8d5e4;border-left:3px solid #9c5789;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)

            with col2:
                if "Country" in df_overview.columns:
                    df_all["Country"] = df_overview["Country"].astype(str).str.strip().str.upper()

                    indo = df_overview[df_overview["Country"] == "INDONESIA"].shape[0]
                    intl = df_overview[df_overview["Country"] != "INDONESIA"].shape[0]

                    pie_df = pd.DataFrame({
                        "Market": ["Domestic", "International"],
                        "Bookings": [indo, intl]
                    })

                    fig_pie = px.pie(
                        pie_df,
                        names="Market",
                        values="Bookings",
                        hole=0.5,
                        color_discrete_sequence=["#9c5789", "#cccccc"]
                    )

                    fig_pie.update_traces(
                        textinfo="percent+label", 
                        textfont=dict(size=12),
                        marker=dict(line=dict(color='white', width=2))
                    )

                    total = pie_df["Bookings"].sum()

                    fig_pie.update_layout(
                        height=380,
                        title="Market Distribution (Dom vs Int)",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                        margin=dict(l=10, r=30, t=50, b=10),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        annotations=[dict(
                            text=f"<b>{total:,}</b><br>Total", 
                            x=0.5, y=0.5, font=dict(size=16), showarrow=False
                        )]
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                if "Issue Time" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:

                    trend_df = prepare_monthly_trend(
                        df_overview,
                        date_col="Issue Time",
                        value_col="Number of Rooms Night",
                        agg="sum"
                    )

                    fig_trend = px.line(
                        trend_df,
                        x="YearMonth",
                        y="Value",
                        markers=True,
                        labels={"Value": "Total Room Nights", "YearMonth": "Month"}
                    )

                    fig_trend.update_traces(
                        line=dict(color="#9c5789", width=3), 
                        marker=dict(size=8, color="#9c5789")
                    )
                    
                    fig_trend.update_layout(
                        height=380,
                        title="Monthly Room Nights Trend",
                        xaxis_title="",
                        yaxis_title="Room Nights",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=40, r=20, t=50, b=40),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_trend, use_container_width=True)

                    output_rn_trend = BytesIO()
                    trend_df.to_excel(
                        output_rn_trend,
                        index=False,
                        sheet_name="Monthly Room Nights Trend"
                    )
                    output_rn_trend.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_rn_trend,
                            file_name="monthly_room_nights_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #e8d5e4;border-left:3px solid #9c5789;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Column 'Issue Time' or 'Number of Rooms Night' not available.")

            for city_col in ["City", "City Destination"]:
                if city_col in df_overview.columns:
                    df_overview[city_col] = (
                        df_overview[city_col]
                        .astype(str).str.strip().str.lower().str.title()
                    )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ======================================
        # TAB 3: EXPLORER
        # ======================================
        with tab3:
            st.markdown("<div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                search_term = st.text_input("🔍 Search", placeholder="Search in any column...")

            with col2:
                if "City Destination" in df_all.columns:
                    cities = ["All"] + sorted(df_all["City Destination"].dropna().unique().tolist())
                    selected_city = st.selectbox("Filter by City", cities)
                else:
                    selected_city = "All"

            df_filtered = df_all.copy()

            if search_term:
                mask = df_filtered.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df_filtered = df_filtered[mask]

            if selected_city != "All" and "City Destination" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["City Destination"] == selected_city]

            st.markdown(f"**Showing {len(df_filtered):,} of {len(df_all):,} records**")
            st.dataframe(df_filtered, use_container_width=True, height=500)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Dataset Information</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Columns", len(df_all.columns))

            with col2:
                st.metric("Total Rows", f"{len(df_all):,}")

            with col3:
                missing_pct = (df_all.isnull().sum().sum() / (len(df_all) * len(df_all.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")

            st.markdown("<div class='section-title'>Hotel Name Text Similarity Analysis</div>", unsafe_allow_html=True)

            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.70,
                max_value=0.95,
                value=0.85,
                step=0.01,
                help="Semakin tinggi, semakin ketat kemiripan"
            )

            if "Hotel Name" in df_all.columns:
                with st.spinner("Analyzing hotel name similarity..."):
                    sim_df = hotel_name_similarity(
                        df_all,
                        text_col="Hotel Name",
                        threshold=threshold
                    )

                if not sim_df.empty:
                    st.dataframe(sim_df, use_container_width=True, height=450)
                    st.caption("🔎 Digunakan untuk mendeteksi potensi duplikasi nama hotel akibat perbedaan penulisan.")

                    output = BytesIO()
                    sim_df.to_excel(output, index=False, sheet_name="Hotel Name Similarity")
                    output.seek(0)

                    if st.session_state.get('role') == 'Admin':
                        st.download_button(
                            label="⬇️ Download Similarity Result (Excel)",
                            data=output,
                            file_name="hotel_name_similarity.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #e8d5e4;border-left:3px solid #9c5789;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Tidak ditemukan hotel dengan tingkat kemiripan sesuai threshold.")
            else:
                st.warning("Kolom 'Hotel Name' tidak tersedia.")

            st.markdown("### Canonical Hotel Mapping")

            if not hotel_mapping.empty:
                st.dataframe(
                    hotel_mapping.sort_values("Canonical Hotel Name"),
                    use_container_width=True,
                    height=350
                )

                output = BytesIO()
                hotel_mapping.to_excel(output, index=False, sheet_name="Hotel Canonical Mapping")
                output.seek(0)

                if st.session_state.get('role') == 'Admin':
                    st.download_button(
                        "⬇️ Download Canonical Mapping (Excel)",
                        data=output,
                        file_name="hotel_canonical_mapping.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.markdown("""
                    <div style='background:#f9f9f9;border:1px solid #e8d5e4;border-left:3px solid #9c5789;
                    border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;
                    display:flex;align-items:center;gap:8px;'>
                        <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Tidak ada mapping canonical yang terbentuk.")

        # ======================================
        # TAB 3: ANALYTICS — CRM
        # ======================================
        with tab4:

            st.markdown("<div class='section-title'>CRM Analytics</div>", unsafe_allow_html=True)

            df_crm = df_all.copy()

            required_cols = ["Employee Id", "Travel Request Number", "Issue Time"]
            if not all(col in df_crm.columns for col in required_cols):
                st.warning("Data belum cukup untuk analisa CRM")
            else:
                df_crm["Issue Time"] = pd.to_datetime(df_crm["Issue Time"], errors="coerce")
                df_crm = df_crm.dropna(subset=["Employee Id", "Issue Time"])

                traveler_stats = (
                    df_crm
                    .groupby("Employee Id")
                    .agg(
                        total_tr=("Travel Request Number", "nunique"),
                        total_booking=("Travel Request Number", "count"),
                        last_booking=("Issue Time", "max"),
                        first_booking=("Issue Time", "min")
                    )
                    .reset_index()
                )

                total_travelers = len(traveler_stats)
                repeat_travelers = (traveler_stats["total_tr"] > 1).sum()
                repeat_rate = repeat_travelers / total_travelers * 100
                avg_booking = traveler_stats["total_booking"].mean()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Active Travelers", f"{total_travelers:,}")
                col2.metric("Repeat Traveler Rate", f"{repeat_rate:.1f}%")
                col3.metric("Avg Booking / Traveler", f"{avg_booking:.1f}")
                col4.metric("Repeat Travelers", f"{repeat_travelers:,}")

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("<div class='section-title'>Employee Booking Cohort Analysis</div>", unsafe_allow_html=True)

            cohort_df = build_employee_cohort(df_all)

            if cohort_df.empty:
                st.warning("Data tidak cukup untuk Cohort Analysis (butuh Employee Id & Issue Time).")
            else:
                cohort_pct = cohort_df.copy()
                if 0 in cohort_pct.columns:
                    base = cohort_pct[0].replace(0, np.nan)
                    cohort_pct = cohort_pct.div(base, axis=0) * 100
                else:
                    base = cohort_pct.iloc[:, 0].replace(0, np.nan)
                    cohort_pct = cohort_pct.div(base, axis=0) * 100

                cohort_pct = cohort_pct.round(1)

                text_matrix = cohort_pct.applymap(
                    lambda v: f"{v:.1f}%" if not np.isnan(v) and v > 0 else ""
                )

                fig = px.imshow(
                    cohort_pct,
                    text_auto=False,
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#e0c7d8", "#9c5789"],
                    zmin=0, zmax=100
                )

                fig.update_traces(
                    text=text_matrix.values,
                    texttemplate="%{text}",
                    textfont=dict(size=10)
                )

                fig.update_layout(
                    title=dict(
                        text="Employee Booking Cohort Heatmap  "
                             "<span style='color:#9c8fa0;font-size:11px;'>"
                             "Retensi relatif terhadap bulan pertama booking (Bulan ke-0 = 100%)"
                             "</span>",
                        font=dict(size=13, color="#2a1a2a"),
                        x=0, xanchor="left"
                    ),
                    xaxis_title="Bulan ke-n sejak booking pertama",
                    yaxis_title="Cohort (Bulan Pertama Booking)",
                    coloraxis_colorbar=dict(
                        title=dict(text="%", font=dict(size=10, color="#9c8fa0")),
                        ticksuffix="%",
                        tickfont=dict(size=9, color="#9c8fa0"),
                        len=0.8
                    ),
                    height=max(400, len(cohort_pct) * 36 + 120),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=60, r=40, t=70, b=60),
                    font=dict(size=11, color="#2a1a2a")
                )

                st.plotly_chart(fig, use_container_width=True)

                output = BytesIO()
                cohort_df.reset_index().to_excel(output, index=False, sheet_name="Employee Cohort")
                output.seek(0)

                if st.session_state.get('role') == 'Admin':
                    st.download_button(
                        label="⬇️ Download Data",
                        data=output,
                        file_name="employee_booking_cohort.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.markdown("""
                    <div style='background:#f9f9f9;border:1px solid #e8d5e4;border-left:3px solid #9c5789;
                    border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;
                    display:flex;align-items:center;gap:8px;'>
                        <span>🔒</span><span>Download hanya tersedia untuk <strong>Admin</strong></span>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📖 Panduan Membaca Cohort Heatmap", expanded=False):
                    _cohort_narasi = (
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:4px;'>"
                        "<div style='background:#fdf7fc;border-radius:9px;padding:14px 16px;border:1px solid #e8d5e4;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#128269; Apa itu Cohort Heatmap?</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.7;'>Cohort Heatmap mengelompokkan karyawan berdasarkan <b>bulan pertama kali mereka melakukan booking</b> (cohort). Nilai pada setiap sel adalah <b>persentase retensi</b>.</div></div>"
                        "<div style='background:#fdf7fc;border-radius:9px;padding:14px 16px;border:1px solid #e8d5e4;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#127919; Kegunaan Analisis Ini</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.8;'>&#8226; <b>Pantau loyalitas traveler</b><br>&#8226; <b>Deteksi penurunan aktivitas</b><br>&#8226; <b>Evaluasi kebijakan travel</b><br>&#8226; <b>Benchmark antar periode</b></div></div>"
                        "<div style='background:#fdf7fc;border-radius:9px;padding:14px 16px;border:1px solid #e8d5e4;'>"
                        "<div style='font-weight:700;color:#6a1a5a;font-size:0.83em;margin-bottom:8px;'>&#128202; Cara Membaca</div>"
                        "<div style='font-size:0.78em;color:#4a3a4a;line-height:1.8;'>&#8226; <b>Kolom 0</b> = bulan pertama &#8594; selalu <b>100%</b><br>&#8226; <b>Warna gelap</b> = retensi tinggi &#9989;<br>&#8226; <b>Warna terang</b> = retensi rendah &#9888;&#65039;</div></div>"
                        "</div>"
                    )
                    st.markdown(_cohort_narasi, unsafe_allow_html=True)

                today = df_crm["Issue Time"].max()

                traveler_stats["Recency (Days)"] = (today - traveler_stats["last_booking"]).dt.days

                if "Invoice Amount" in df_crm.columns:
                    spend = (df_crm.groupby("Employee Id")["Invoice Amount"].sum().reset_index(name="Total Spend"))
                    traveler_stats = traveler_stats.merge(spend, on="Employee Id", how="left")
                else:
                    traveler_stats["Total Spend"] = 0

                def segment(row):
                    if row["total_tr"] >= 10: return "High Value"
                    elif row["total_tr"] >= 3: return "Medium Value"
                    else: return "Low Value"

                traveler_stats["Segment"] = traveler_stats.apply(segment, axis=1)

                st.markdown("<div class='section-title'>Top Valuable Travelers</div>", unsafe_allow_html=True)

                top_travelers = traveler_stats.sort_values(by=["total_tr", "Total Spend"], ascending=False).head(10)
                display_cols = ["Employee Id", "total_tr", "total_booking", "Total Spend", "Segment"]
                numeric_cols = ["total_tr", "total_booking", "Total Spend"]

                st.dataframe(
                    top_travelers[display_cols]
                        .style
                        .format({"Total Spend": lambda x: f"Rp{x:,.0f}" if pd.notnull(x) else "Rp 0"})
                        .set_properties(subset=numeric_cols, **{"text-align": "right"}),
                    use_container_width=True
                )

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                with tab4:
                    st.markdown("### Behavioral Persona Clustering")

                    df_behavior = df_all.copy()

                    required_cols = ["Travel Request Number","Employee Id","Issue Time","Check in Date","Check out Date","Number of Rooms Night"]

                    if all(col in df_behavior.columns for col in required_cols):

                        df_behavior["Issue Time"] = pd.to_datetime(df_behavior["Issue Time"], errors="coerce")
                        df_behavior["Check in Date"] = pd.to_datetime(df_behavior["Check in Date"], errors="coerce")
                        df_behavior["Check out Date"] = pd.to_datetime(df_behavior["Check out Date"], errors="coerce")

                        df_behavior["Lead_Time"] = (df_behavior["Check in Date"] - df_behavior["Issue Time"]).dt.days
                        df_behavior["Last_Minute"] = df_behavior["Lead_Time"].apply(lambda x: 1 if pd.notnull(x) and x <= 2 else 0)
                        df_behavior["Weekend_Stay"] = df_behavior["Check in Date"].dt.weekday.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)

                        employee_features = df_behavior.groupby("Employee Id").agg(
                            Booking_Frequency=("Travel Request Number", "nunique"),
                            Avg_Lead_Time=("Lead_Time", "mean"),
                            Last_Minute_Ratio=("Last_Minute", "mean"),
                            Avg_Stay=("Number of Rooms Night", "mean"),
                            Weekend_Ratio=("Weekend_Stay", "mean")
                        ).reset_index()

                        employee_features = employee_features.fillna(0)

                        feature_cols = ["Booking_Frequency","Avg_Lead_Time","Last_Minute_Ratio","Avg_Stay","Weekend_Ratio"]

                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(employee_features[feature_cols])

                        n_employee = len(employee_features)
                        if n_employee >= 4: n_cluster = 4
                        elif n_employee >= 2: n_cluster = 2
                        else: n_cluster = 1

                        kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
                        employee_features["Cluster"] = kmeans.fit_predict(X_scaled)

                        cluster_profile = employee_features.groupby("Cluster")[feature_cols].mean()

                        persona_map = {}
                        for cluster_id, row in cluster_profile.iterrows():
                            if row["Last_Minute_Ratio"] > 0.5: persona = "Last Minute Traveler"
                            elif row["Avg_Lead_Time"] > 14: persona = "Strategic Planner"
                            elif row["Weekend_Ratio"] > 0.4: persona = "Weekend Traveler"
                            elif row["Booking_Frequency"] > employee_features["Booking_Frequency"].median(): persona = "Frequent Traveler"
                            else: persona = "Regular Business Traveler"
                            persona_map[cluster_id] = persona

                        employee_features["Persona"] = employee_features["Cluster"].map(persona_map)

                        selected_employee = st.selectbox("Select Employee Id", employee_features["Employee Id"])

                        selected_data = employee_features[employee_features["Employee Id"] == selected_employee]
                        selected_cluster = selected_data["Cluster"].values[0]
                        selected_persona = selected_data["Persona"].values[0]

                        st.success(f"Persona: {selected_persona}")

                if not selected_data.empty:
                    st.markdown("""
                    <style>
                    .metric-card{background:white;padding:20px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center;transition:all 0.3s ease;margin-bottom:12px;border:1px solid #f0f0f0;border-top:3px solid #9c5789;}
                    .metric-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(156,87,137,0.12);}
                    .metric-value{font-size:32px;font-weight:700;margin:8px 0;color:#9c5789;}
                    .metric-label{font-size:11px;color:#888888;text-transform:uppercase;letter-spacing:1px;font-weight:500;}
                    .insight-card{background:white;border-radius:6px;padding:16px;margin-bottom:10px;border-left:4px solid;box-shadow:0 1px 4px rgba(0,0,0,0.05);font-size:14px;line-height:1.6;}
                    .insight-success{border-left-color:#10b981;background:linear-gradient(to right,#ecfdf5,white);}
                    .insight-warning{border-left-color:#f59e0b;background:linear-gradient(to right,#fffbeb,white);}
                    .insight-info{border-left-color:#9c5789;background:linear-gradient(to right,#f8f4f7,white);}
                    .insight-error{border-left-color:#ef4444;background:linear-gradient(to right,#fef2f2,white);}
                    .persona-badge{background:linear-gradient(135deg,#9c5789 0%,#c983af 100%);padding:24px;border-radius:8px;text-align:center;color:white;font-size:19px;font-weight:600;box-shadow:0 4px 16px rgba(156,87,137,0.2);margin:15px 0;}
                    .section-header{font-size:18px;font-weight:600;color:#1a1a1a;margin:30px 0 18px 0;padding-bottom:8px;border-bottom:2px solid #9c5789;}
                    .progress-container{background:#f0f0f0;height:6px;border-radius:3px;overflow:hidden;margin-top:6px;}
                    .progress-bar{height:100%;border-radius:3px;transition:width 0.4s ease;}
                    .breakdown-card{background:white;padding:12px 14px;border-radius:6px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.04);border:1px solid #f0f0f0;}
                    </style>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1.4], gap="large")

                    with col1:
                        st.markdown('<div class="section-header">📊 Behavioral Overview</div>', unsafe_allow_html=True)
                        bf = selected_data["Booking_Frequency"].values[0]
                        lead = selected_data["Avg_Lead_Time"].values[0]
                        lf = selected_data["Last_Minute_Ratio"].values[0]
                        weekend = selected_data["Weekend_Ratio"].values[0]
                        stay = selected_data["Avg_Stay"].values[0]

                        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Booking Frequency</div><div class="metric-value">{round(bf,1)}</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Lead Time (Days)</div><div class="metric-value">{round(lead,1)}</div></div>', unsafe_allow_html=True)
                        m3, m4 = st.columns(2)
                        with m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Last Minute Ratio</div><div class="metric-value">{round(lf*100,1)}%</div></div>', unsafe_allow_html=True)
                        with m4:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Weekend Ratio</div><div class="metric-value">{round(weekend*100,1)}%</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Average Stay Duration</div><div class="metric-value">{round(stay,1)} <span style="font-size:18px;font-weight:500;">nights</span></div></div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header">🎯 Behavioral Radar Profile</div>', unsafe_allow_html=True)

                        from sklearn.preprocessing import MinMaxScaler
                        cluster_profile = employee_features.groupby("Cluster")[feature_cols].mean()
                        minmax_scaler = MinMaxScaler()
                        cluster_scaled = minmax_scaler.fit_transform(cluster_profile)

                        profile_row = cluster_scaled[selected_cluster]
                        radar_values = list(profile_row) + [profile_row[0]]
                        radar_labels = feature_cols + [feature_cols[0]]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=radar_values, theta=radar_labels, fill='toself',
                            line=dict(width=4, color="#9c5789"), fillcolor="rgba(156,87,137,0.25)", name='Profile'))
                        fig.add_trace(go.Scatterpolar(r=[0.5]*len(radar_labels), theta=radar_labels,
                            line=dict(width=2, color="rgba(138,77,120,0.4)", dash='dash'), name='Benchmark'))

                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            showlegend=True, height=580,
                            margin=dict(l=50,r=50,t=50,b=50),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if not selected_data.empty:
                    col1, col2 = st.columns([1, 1.4], gap="large")

                    with col1:
                        st.markdown('<div class="section-header">💡 Behavioral Insights</div>', unsafe_allow_html=True)
                        if lf > 0.5:
                            st.markdown('<div class="insight-card insight-error"><strong>⚠️ Reactive Traveler</strong><br>High last-minute booking ratio detected.</div>', unsafe_allow_html=True)
                        elif lead > 7:
                            st.markdown('<div class="insight-card insight-success"><strong>✅ Strategic Planner</strong><br>Excellent advance planning.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-info"><strong>ℹ️ Balanced Approach</strong><br>Shows balanced booking behavior.</div>', unsafe_allow_html=True)

                        avg_bf = employee_features["Booking_Frequency"].mean()
                        if bf > avg_bf:
                            intensity_pct = ((bf - avg_bf) / avg_bf * 100)
                            st.markdown(f'<div class="insight-card insight-warning"><strong>📊 High Activity</strong><br>Travel intensity <strong>{round(intensity_pct,1)}%</strong> above peer average.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-success"><strong>📊 Normal Activity</strong><br>Travel intensity aligns with baseline.</div>', unsafe_allow_html=True)

                        if weekend > 0.4:
                            st.markdown('<div class="insight-card insight-info"><strong>🌅 Weekend Preference</strong><br>Strong weekend tendency.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-info"><strong>💼 Weekday Focus</strong><br>Primarily weekday travel.</div>', unsafe_allow_html=True)

                        avg_stay_val = employee_features["Avg_Stay"].mean()
                        if stay > avg_stay_val:
                            st.markdown(f'<div class="insight-card insight-warning"><strong>🏨 Extended Stays</strong><br><strong>{round(stay-avg_stay_val,1)}</strong> nights above average.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="insight-card insight-success"><strong>🏨 Quick Visits</strong><br>Efficient short trips.</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header">📋 Metric Breakdown</div>', unsafe_allow_html=True)
                        for i, feature in enumerate(feature_cols):
                            score = profile_row[i]
                            if score > 0.7: status, status_icon, color = "High", "🔴", "#9c5789"
                            elif score > 0.4: status, status_icon, color = "Medium", "🟡", "#c983af"
                            else: status, status_icon, color = "Low", "🟢", "#e7c3d9"
                            progress_width = int(score * 100)
                            st.markdown(f"""
                            <div class="breakdown-card">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                    <strong style="color:#1a1a1a;font-size:13px;">{feature.replace("_"," ")}</strong>
                                    <span style="color:#888;font-weight:500;font-size:11px;margin:0 10px;">{score:.2f}</span>
                                    <span style="color:{color};font-weight:600;font-size:11px;">{status_icon} {status}</span>
                                </div>
                                <div class="progress-container">
                                    <div class="progress-bar" style="background:{color};width:{progress_width}%;"></div>
                                </div>
                            </div>""", unsafe_allow_html=True)

        # ======================================
        # TAB 5: SOCIAL NETWORK ANALYSIS
        # ======================================
        with tab5:
            st.markdown("""
                <div style="background:linear-gradient(135deg,#1a1a2e 0%,#2d1b3d 50%,#1a1a2e 100%);
                border-radius:12px;padding:32px 36px;margin-bottom:28px;position:relative;overflow:hidden;">
                    <div style="position:relative;z-index:1;">
                        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                            <span style="font-size:2em;">🕸️</span>
                            <div>
                                <div style="font-size:1.35em;font-weight:700;color:#ffffff;">Social Network Analysis</div>
                                <div style="font-size:0.85em;color:rgba(255,255,255,0.55);margin-top:3px;">Employee ↔ Hotel Interaction Network</div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            required_cols = ["Employee Id", "Hotel Name"]

            if all(col in df_all.columns for col in required_cols):

                df_sna = (df_all.dropna(subset=required_cols)
                          .groupby(required_cols).size().reset_index(name="weight"))

                total_emp_count = df_sna["Employee Id"].nunique()
                total_htl_count = df_sna["Hotel Name"].nunique()
                total_edges     = len(df_sna)
                avg_connections = df_sna.groupby("Employee Id")["weight"].sum().mean()

                stat_cols = st.columns(4)
                stat_data = [
                    ("👤","Total Employees",f"{total_emp_count:,}","#9c5789"),
                    ("🏨","Total Hotels",f"{total_htl_count:,}","#5879c0"),
                    ("🔗","Total Interactions",f"{total_edges:,}","#5a9c7e"),
                    ("📊","Avg Trips / Emp",f"{avg_connections:.1f}","#c07840"),
                ]
                for col, (icon,label,val,color) in zip(stat_cols, stat_data):
                    with col:
                        st.markdown(f"""
                        <div style="background:white;border-radius:10px;padding:16px 18px;
                                    border-top:3px solid {color};box-shadow:0 1px 6px rgba(0,0,0,0.07);
                                    text-align:center;margin-bottom:20px;">
                            <div style="font-size:1.4em;margin-bottom:5px;">{icon}</div>
                            <div style="font-size:0.7em;font-weight:600;color:#888;text-transform:uppercase;">{label}</div>
                            <div style="font-size:1.3em;font-weight:700;color:#1a1a1a;margin-top:6px;">{val}</div>
                        </div>""", unsafe_allow_html=True)

                fcol1, fcol2, fcol3 = st.columns([2,2,1])
                with fcol1:
                    top_emp = st.slider("👤 Top Employees", 5, min(100,total_emp_count), min(50,total_emp_count))
                with fcol2:
                    top_htl = st.slider("🏨 Top Hotels", 5, min(50,total_htl_count), min(15,total_htl_count))
                with fcol3:
                    layout_algo = st.selectbox("📐 Layout", ["Spring","Kamada-Kawai","Circular"])

                top_employees = (df_sna.groupby("Employee Id")["weight"].sum()
                                 .sort_values(ascending=False).head(top_emp).index)
                top_hotels = (df_sna.groupby("Hotel Name")["weight"].sum()
                              .sort_values(ascending=False).head(top_htl).index)
                df_filtered_sna = df_sna[df_sna["Employee Id"].isin(top_employees) & df_sna["Hotel Name"].isin(top_hotels)]

                G = nx.Graph()
                for _, row in df_filtered_sna.iterrows():
                    G.add_edge(row["Employee Id"], row["Hotel Name"], weight=row["weight"])

                seed = 42
                if layout_algo == "Spring":
                    pos = nx.spring_layout(G, seed=seed, k=0.7)
                elif layout_algo == "Kamada-Kawai":
                    try: pos = nx.kamada_kawai_layout(G)
                    except: pos = nx.spring_layout(G, seed=seed)
                else:
                    pos = nx.circular_layout(G)

                degree      = dict(G.degree())
                betweenness = nx.betweenness_centrality(G)
                max_weight  = max((G[u][v]["weight"] for u,v in G.edges()), default=1)
                max_degree  = max(degree.values(), default=1)

                edge_traces = []
                for u, v in G.edges():
                    x0,y0 = pos[u]; x1,y1 = pos[v]
                    w = G[u][v]["weight"]
                    opacity = 0.15 + 0.65*(w/max_weight)
                    width   = 0.5  + 3.5*(w/max_weight)
                    edge_traces.append(go.Scatter(
                        x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                        line=dict(width=width, color=f"rgba(156,87,137,{opacity:.2f})"),
                        hoverinfo="none", showlegend=False
                    ))

                emp_x,emp_y,emp_text,emp_size,emp_mc = [],[],[],[],[]
                for node in G.nodes():
                    if node not in top_employees: continue
                    x,y = pos[node]; deg = degree[node]; bet = betweenness.get(node,0)
                    size = 14+(deg/max_degree)*30
                    total_trips   = df_filtered_sna[df_filtered_sna["Employee Id"]==node]["weight"].sum()
                    hotels_visited= df_filtered_sna[df_filtered_sna["Employee Id"]==node]["Hotel Name"].nunique()
                    emp_x.append(x); emp_y.append(y); emp_size.append(size); emp_mc.append(deg)
                    emp_text.append(f"<b>👤 {node}</b><br>Hotel Connections: <b>{deg}</b><br>Total Stays: <b>{int(total_trips):,}</b><br>Unique Hotels: <b>{hotels_visited}</b><br>Betweenness: <b>{bet:.3f}</b>")

                employee_trace = go.Scatter(x=emp_x,y=emp_y,mode="markers",name="Employee",
                    hoverinfo="text",text=emp_text,
                    marker=dict(size=emp_size,color=emp_mc,
                        colorscale=[[0.0,"#d4a0c8"],[0.5,"#9c5789"],[1.0,"#5c1f4a"]],
                        showscale=True,colorbar=dict(title=dict(text="Degree<br>(Employee)"),thickness=10,len=0.45,y=0.75,x=1.01),
                        line=dict(width=2,color="white"),symbol="circle"))

                htl_x,htl_y,htl_text,htl_size,htl_mc = [],[],[],[],[]
                for node in G.nodes():
                    if node not in top_hotels: continue
                    x,y = pos[node]; deg = degree[node]; bet = betweenness.get(node,0)
                    size = 18+(deg/max_degree)*28
                    total_stays = df_filtered_sna[df_filtered_sna["Hotel Name"]==node]["weight"].sum()
                    unique_emps = df_filtered_sna[df_filtered_sna["Hotel Name"]==node]["Employee Id"].nunique()
                    htl_x.append(x); htl_y.append(y); htl_size.append(size); htl_mc.append(deg)
                    htl_text.append(f"<b>🏨 {node}</b><br>Employee Connections: <b>{deg}</b><br>Total Stays: <b>{int(total_stays):,}</b><br>Unique Travelers: <b>{unique_emps}</b><br>Betweenness: <b>{bet:.3f}</b>")

                hotel_trace = go.Scatter(x=htl_x,y=htl_y,mode="markers",name="Hotel",
                    hoverinfo="text",text=htl_text,
                    marker=dict(size=htl_size,color=htl_mc,
                        colorscale=[[0.0,"#a0b8e8"],[0.5,"#5879c0"],[1.0,"#1a3a7a"]],
                        showscale=True,colorbar=dict(title=dict(text="Degree<br>(Hotel)"),thickness=10,len=0.45,y=0.28,x=1.01),
                        line=dict(width=2,color="white"),symbol="diamond"))

                fig = go.Figure(
                    data=edge_traces+[employee_trace,hotel_trace],
                    layout=go.Layout(
                        title=dict(text=f"<b>Employee ↔ Hotel Network</b>",font=dict(size=15,color="#1a1a1a"),x=0.0,xanchor="left"),
                        showlegend=True,hovermode="closest",
                        margin=dict(b=60,l=10,r=80,t=60),
                        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                        plot_bgcolor="#fafafa",paper_bgcolor="white",height=640
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                rank_col1, rank_col2 = st.columns(2)
                emp_rank = []
                for node in top_employees:
                    if node not in G.nodes(): continue
                    deg = degree.get(node,0); bet = betweenness.get(node,0)
                    total_stays = df_filtered_sna[df_filtered_sna["Employee Id"]==node]["weight"].sum()
                    emp_rank.append({"Employee ID":str(node),"Connections":deg,"Total Stays":int(total_stays),"Centrality":round(bet,4)})
                emp_rank_df = pd.DataFrame(emp_rank).sort_values("Connections",ascending=False).head(10).reset_index(drop=True)
                emp_rank_df.index = emp_rank_df.index+1; emp_rank_df.index.name = "Rank"

                htl_rank = []
                for node in top_hotels:
                    if node not in G.nodes(): continue
                    deg = degree.get(node,0); bet = betweenness.get(node,0)
                    total_stays = df_filtered_sna[df_filtered_sna["Hotel Name"]==node]["weight"].sum()
                    unique_emps = df_filtered_sna[df_filtered_sna["Hotel Name"]==node]["Employee Id"].nunique()
                    htl_rank.append({"Hotel Name":str(node),"Travelers":deg,"Total Stays":int(total_stays),"Centrality":round(bet,4)})
                htl_rank_df = pd.DataFrame(htl_rank).sort_values("Travelers",ascending=False).head(10).reset_index(drop=True)
                htl_rank_df.index = htl_rank_df.index+1; htl_rank_df.index.name = "Rank"

                with rank_col1:
                    st.markdown("""<div style="background:linear-gradient(135deg,#f5eef3 0%,#ffffff 100%);border-radius:10px;padding:14px 18px 4px 18px;border-left:3px solid #9c5789;margin-bottom:8px;">
                    <div style="font-size:0.78em;font-weight:700;color:#9c5789;text-transform:uppercase;letter-spacing:0.7px;">👤 Top Employees by Connectivity</div></div>""", unsafe_allow_html=True)
                    st.dataframe(emp_rank_df, use_container_width=True)

                with rank_col2:
                    st.markdown("""<div style="background:linear-gradient(135deg,#eef3f5 0%,#ffffff 100%);border-radius:10px;padding:14px 18px 4px 18px;border-left:3px solid #5879c0;margin-bottom:8px;">
                    <div style="font-size:0.78em;font-weight:700;color:#5879c0;text-transform:uppercase;letter-spacing:0.7px;">🏨 Top Hotels by Dependency Risk</div></div>""", unsafe_allow_html=True)
                    st.dataframe(htl_rank_df, use_container_width=True)

            else:
                st.warning("⚠️ Kolom Employee Id atau Hotel Name tidak tersedia.")

        # ======================================
        # TAB 6: SPEND CONCENTRATION (PARETO 80/20)
        # ======================================
        with tab6:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#9c5789 0%,#b07a9e 100%);
            padding:25px 30px;border-radius:8px;margin-bottom:25px;">
                <h2 style="color:white;margin:0;font-weight:500;">Spend Concentration Analysis</h2>
                <p style="color:rgba(255,255,255,0.85);margin:8px 0 0 0;font-size:0.95em;">Pareto 80/20 Analysis</p>
            </div>""", unsafe_allow_html=True)

            required_cols = ["Invoice Amount"]

            if not all(col in df_all.columns for col in required_cols):
                st.warning("⚠️ Kolom Invoice Amount tidak tersedia.")
            else:
                df_sc = df_all.copy()
                df_sc = df_sc.dropna(subset=["Invoice Amount"])

                dimension_options = []
                if "Hotel Name" in df_sc.columns: dimension_options.append("Hotel Name")
                if "City" in df_sc.columns: dimension_options.append("City")
                if "Supplier Name" in df_sc.columns: dimension_options.append("Supplier Name")

                if len(dimension_options) == 0:
                    st.warning("⚠️ Tidak ada dimensi yang tersedia.")
                else:
                    dimension = st.selectbox("Analisa berdasarkan:", dimension_options)

                    if "Country" in df_sc.columns:
                        st.markdown("""
                        <style>
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > div[role="radiogroup"] {
                            display:inline-flex!important;background:#9c5789;border-radius:50px;padding:3px;gap:0;
                        }
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > div[role="radiogroup"] > label {
                            cursor:pointer;padding:4px 16px!important;border-radius:50px!important;
                            font-size:0.78em!important;font-weight:500!important;color:rgba(255,255,255,0.80)!important;margin:0!important;
                        }
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > div[role="radiogroup"] > label > div:first-child{display:none!important;}
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){background:white!important;color:#9c5789!important;}
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > div[role="radiogroup"] > label:has(input:checked) > div:last-child p{color:#9c5789!important;}
                        div[data-testid="stRadio"][data-key="pareto_country_radio"] > label{display:none!important;}
                        </style>""", unsafe_allow_html=True)
                        country_filter = st.radio(label="filter_wilayah",options=["Indonesia","Non-Indonesia"],
                                                  index=0,horizontal=True,label_visibility="collapsed",key="pareto_country_radio")

                        df_sc_country = df_sc.copy()
                        df_sc_country["_country_upper"] = df_sc_country["Country"].astype(str).str.strip().str.upper()
                        if country_filter == "Indonesia":
                            df_sc_filtered = df_sc_country[df_sc_country["_country_upper"]=="INDONESIA"]
                            filter_label = "🇮🇩 Indonesia"
                        else:
                            df_sc_filtered = df_sc_country[df_sc_country["_country_upper"]!="INDONESIA"]
                            filter_label = "🌐 Non-Indonesia"
                        df_sc_filtered = df_sc_filtered.drop(columns=["_country_upper"])
                        if df_sc_filtered.empty:
                            st.warning(f"⚠️ Tidak ada data untuk filter: {filter_label}")
                            st.stop()
                    else:
                        df_sc_filtered = df_sc.copy()
                        filter_label = "🌏 Semua Wilayah"

                    pareto_df = (df_sc_filtered.groupby(dimension)["Invoice Amount"].sum()
                                 .reset_index().sort_values("Invoice Amount",ascending=False))
                    total_spend = pareto_df["Invoice Amount"].sum()
                    pareto_df["Spend %"] = pareto_df["Invoice Amount"]/total_spend*100
                    pareto_df["Cumulative %"] = pareto_df["Spend %"].cumsum()
                    pareto_df["Rank"] = range(1,len(pareto_df)+1)

                    top_20_percent_count = max(1,int(len(pareto_df)*0.2))
                    top_contributors = pareto_df.head(top_20_percent_count)
                    top_spend = top_contributors["Invoice Amount"].sum()
                    top_spend_pct = top_spend/total_spend*100

                    col1,col2,col3,col4 = st.columns(4)
                    for c,(lbl,val) in zip([col1,col2,col3,col4],[
                        ("Total Spend",f"Rp{total_spend:,.0f}"),
                        ("Top 20% Count",str(top_20_percent_count)),
                        ("Top 20% Contribution",f"{top_spend_pct:.1f}%"),
                        ("Bottom 80% Spend",f"Rp{(total_spend-top_spend):,.0f}"),
                    ]):
                        with c:
                            st.markdown(f"""<div style="background:white;padding:18px;border-radius:6px;border-left:3px solid #9c5789;">
                            <div style="color:#999;font-size:0.8em;margin-bottom:6px;">{lbl} · {filter_label}</div>
                            <div style="color:#9c5789;font-size:1.6em;font-weight:500;">{val}</div></div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    colors = ['#9c5789' if i < top_20_percent_count else '#d4d4d4' for i in range(len(pareto_df))]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=pareto_df[dimension],y=pareto_df["Invoice Amount"],name="Spend",
                                        marker=dict(color=colors),hovertemplate="<b>%{x}</b><br>Rp%{y:,.0f}<extra></extra>"))
                    fig.add_trace(go.Scatter(x=pareto_df[dimension],y=pareto_df["Cumulative %"],name="Cumulative %",
                                            yaxis="y2",mode="lines+markers",line=dict(color='#9c5789',width=2.5),
                                            marker=dict(size=5)))
                    fig.add_hline(y=80,yref='y2',line_dash="dash",line_color="#9c5789",opacity=0.4,
                                  annotation_text="80%",annotation_position="right")
                    fig.update_layout(template="plotly_white",
                                      yaxis=dict(title="Spend (Rp)"),
                                      yaxis2=dict(title="Cumulative %",overlaying="y",side="right",range=[0,100],showgrid=False),
                                      height=500,plot_bgcolor="white",paper_bgcolor="white",
                                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                                      margin=dict(l=60,r=60,t=40,b=100),
                                      xaxis=dict(tickangle=-45,tickfont=dict(size=9)),hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

                    tab_sim, tab_detail, tab_insight = st.tabs(["Saving Simulation","Detail Data","Insight"])

                    with tab_sim:
                        st.markdown("### Saving Simulation")
                        renegotiation_rate = st.slider("Target diskon pada Top 20% contributors (%)",0,25,5,1)
                        potential_saving = top_spend*(renegotiation_rate/100)
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown(f"""<div style="background:#9c5789;padding:20px;border-radius:6px;color:white;">
                            <div style="font-size:0.85em;margin-bottom:6px;opacity:0.9;">Potensi Saving</div>
                            <div style="font-size:2em;font-weight:500;">Rp{potential_saving:,.0f}</div>
                            <div style="font-size:0.8em;opacity:0.8;">dengan diskon {renegotiation_rate}%</div></div>""", unsafe_allow_html=True)
                        with col_s2:
                            st.markdown(f"""<div style="background:#f5f5f5;padding:20px;border-radius:6px;border-left:3px solid #9c5789;">
                            <div style="font-size:0.9em;color:#333;line-height:1.6;">Top 20% Spend: <strong>Rp{top_spend:,.0f}</strong><br>
                            Diskon: <strong>{renegotiation_rate}%</strong><br>Saving: <strong>Rp{potential_saving:,.0f}</strong></div></div>""", unsafe_allow_html=True)

                    with tab_detail:
                        st.markdown("### Top Contributors")
                        display_cols = [dimension,"Invoice Amount","Spend %","Cumulative %","Rank"]
                        st.dataframe(
                            top_contributors[display_cols].style
                            .format({"Invoice Amount":"Rp{:,.0f}","Spend %":"{:.2f}%","Cumulative %":"{:.2f}%"})
                            .background_gradient(subset=["Spend %"],cmap="BuPu"),
                            use_container_width=True
                        )
                        output_excel = BytesIO()
                        top_contributors.to_excel(output_excel,index=False,sheet_name="Top Contributors")
                        output_excel.seek(0)
                        if st.session_state.get('role') == 'Admin':
                            st.download_button(label="Download Excel",data=output_excel,
                                file_name=f"pareto_{dimension.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        else:
                            st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #9c5789;border-radius:6px;
                            padding:10px 16px;font-size:0.82em;color:#9c5789;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                            unsafe_allow_html=True)

                    with tab_insight:
                        st.markdown("### Key Insights")
                        st.markdown(f"""<div style="background:#f9f9f9;padding:20px;border-radius:6px;border-left:3px solid #9c5789;">
                        <p style="margin:0 0 12px 0;color:#333;"><strong>Konsentrasi Spending ({filter_label}):</strong><br>
                        Top 20% ({top_20_percent_count} {dimension}) menyumbang <strong>{top_spend_pct:.1f}%</strong> dari total pengeluaran.</p>
                        <p style="margin:0;color:#666;font-size:0.9em;">Fokus renegosiasi pada kelompok ini memberikan dampak finansial terbesar.</p></div>""",
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                # ── Hotel Price Intelligence ──────────────────────────────────
                st.markdown("""
                <div style="background:white;border-radius:8px;padding:28px 32px;margin-bottom:24px;
                            border-left:4px solid #9c5789;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                    <div style="font-size:1.25em;font-weight:700;color:#1a1a1a;">🏨 Hotel Price Intelligence</div>
                    <div style="font-size:0.88em;color:#888888;margin-top:4px;">Analisis harga historis & simulasi negosiasi kontrak hotel</div>
                </div>""", unsafe_allow_html=True)

                if df_all.empty:
                    st.warning("No data loaded.")
                else:
                    if "Canonical Hotel Name" in df_all.columns:
                        df_all["Hotel Display"] = df_all["Canonical Hotel Name"].astype(str).str.strip().str.replace(r"\s+"," ",regex=True).str.title()
                    else:
                        df_all["Hotel Display"] = df_all["Hotel Name"].astype(str).str.strip().str.replace(r"\s+"," ",regex=True).str.title()

                    hotel_list = sorted(df_all["Hotel Display"].dropna().unique())
                    selected_hotel = st.selectbox("Select Hotel", hotel_list, label_visibility="collapsed")
                    df_hotel = df_all[df_all["Hotel Display"]==selected_hotel].copy()

                    if not df_hotel.empty:
                        if "Invoice Amount" in df_hotel.columns and "Number of Rooms Night" in df_hotel.columns:
                            df_valid = df_hotel[(df_hotel["Invoice Amount"].notna()) & (df_hotel["Number of Rooms Night"]>0)].copy()
                            df_valid["Price Per Night"] = df_valid["Invoice Amount"]/df_valid["Number of Rooms Night"]
                            avg_rate    = df_valid["Price Per Night"].mean()
                            median_rate = df_valid["Price Per Night"].median()
                            max_rate    = df_valid["Price Per Night"].max()
                            min_rate    = df_valid["Price Per Night"].min()
                            total_room_nights = df_valid["Number of Rooms Night"].sum()

                            col1,col2,col3,col4,col5 = st.columns(5)
                            kpi_data = [
                                ("📊","Avg Rate",f"Rp {avg_rate:,.0f}","#9c5789"),
                                ("📍","Median Rate",f"Rp {median_rate:,.0f}","#7a4a6e"),
                                ("🔺","Max Rate",f"Rp {max_rate:,.0f}","#c0556a"),
                                ("🔻","Min Rate",f"Rp {min_rate:,.0f}","#5a9c7e"),
                                ("🌙","Total Room Nights",f"{total_room_nights:,.0f}","#5879c0"),
                            ]
                            for col,(icon,label,value,color) in zip([col1,col2,col3,col4,col5],kpi_data):
                                with col:
                                    st.markdown(f"""<div style="background:white;border-radius:8px;padding:18px 16px 16px;
                                    border-top:3px solid {color};box-shadow:0 1px 4px rgba(0,0,0,0.06);text-align:center;">
                                    <div style="font-size:1.5em;">{icon}</div>
                                    <div style="font-size:0.72em;font-weight:600;color:#888;text-transform:uppercase;margin:6px 0 8px;">{label}</div>
                                    <div style="font-size:1.05em;font-weight:700;color:#1a1a1a;">{value}</div></div>""", unsafe_allow_html=True)

                            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                            target_discount = st.slider("🎯 Target Discount (%)",0,30,10)
                            negotiated_rate  = avg_rate*(1-target_discount/100)
                            estimated_saving = (avg_rate-negotiated_rate)*total_room_nights

                            res_col1,res_col2,res_col3 = st.columns(3)
                            with res_col1:
                                st.markdown(f"""<div style="background:#f5eef3;border-radius:8px;padding:20px;text-align:center;border:1px solid #e8d5e3;">
                                <div style="font-size:0.75em;font-weight:600;color:#9c5789;text-transform:uppercase;">Current Avg Rate</div>
                                <div style="font-size:1.4em;font-weight:700;color:#1a1a1a;margin-top:8px;">Rp {avg_rate:,.0f}</div>
                                <div style="font-size:0.78em;color:#888;margin-top:4px;">per malam</div></div>""", unsafe_allow_html=True)
                            with res_col2:
                                st.markdown(f"""<div style="background:#eef3f5;border-radius:8px;padding:20px;text-align:center;border:1px solid #d5e3e8;">
                                <div style="font-size:0.75em;font-weight:600;color:#5879c0;text-transform:uppercase;">Negotiated Rate (-{target_discount}%)</div>
                                <div style="font-size:1.4em;font-weight:700;color:#1a1a1a;margin-top:8px;">Rp {negotiated_rate:,.0f}</div>
                                <div style="font-size:0.78em;color:#888;margin-top:4px;">per malam</div></div>""", unsafe_allow_html=True)
                            with res_col3:
                                st.markdown(f"""<div style="background:linear-gradient(135deg,#9c5789 0%,#7a4a6e 100%);border-radius:8px;padding:20px;text-align:center;">
                                <div style="font-size:0.75em;font-weight:600;color:rgba(255,255,255,0.75);text-transform:uppercase;">💰 Estimated Saving</div>
                                <div style="font-size:1.4em;font-weight:700;color:#fff;margin-top:8px;">Rp {estimated_saving:,.0f}</div>
                                <div style="font-size:0.78em;color:rgba(255,255,255,0.65);margin-top:4px;">total potensi hemat</div></div>""", unsafe_allow_html=True)

                            with st.expander("📘 Cara Perhitungan Estimated Saving", expanded=False):
                                st.markdown(f"""Saving = (Rp {avg_rate:,.0f} − Rp {negotiated_rate:,.0f}) × {total_room_nights:,.0f} malam
                                = **Rp {estimated_saving:,.0f}**""")

        # ======================================
        # TAB 7: SANKEY FLOW
        # ======================================
        with tab7:
            st.markdown("""<div style="background:#ffffff;border:1px solid #EBEBEB;border-left:4px solid #9c5789;
            border-radius:10px;padding:20px 28px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="color:#111111;font-size:1.30em;font-weight:700;letter-spacing:-0.02em;margin-bottom:4px;">
                    Perusahaan &rarr; Kota &rarr; Hotel</div>
                <div style="color:#999999;font-size:0.82em;">Visualisasi alur pengeluaran travel berdasarkan volume Invoice Amount</div>
            </div>""", unsafe_allow_html=True)

            required_sankey = ["Nama Perusahaan","Hotel Name","Invoice Amount"]
            city_sankey_col = next((c for c in ["City","City Destination"] if c in df_overview.columns), None)

            if not all(c in df_overview.columns for c in required_sankey) or city_sankey_col is None:
                st.warning("Kolom yang dibutuhkan tidak lengkap.")
            else:
                row_ctrl = st.columns([1.2,1,1,1,1])
                with row_ctrl[0]:
                    st.markdown("""
                    <style>
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > div[role="radiogroup"]{display:inline-flex!important;background:#9c5789;border-radius:50px;padding:3px;}
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > div[role="radiogroup"] > label{cursor:pointer;padding:4px 16px!important;border-radius:50px!important;font-size:0.78em!important;font-weight:500!important;color:rgba(255,255,255,0.80)!important;margin:0!important;}
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > div[role="radiogroup"] > label > div:first-child{display:none!important;}
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){background:white!important;color:#9c5789!important;}
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > div[role="radiogroup"] > label:has(input:checked) > div:last-child p{color:#9c5789!important;}
                    div[data-testid="stRadio"][data-key="sankey_country_radio"] > label{display:none!important;}
                    </style>""", unsafe_allow_html=True)
                    sankey_country_filter = st.radio(label="sankey_filter",options=["Domestik","Internasional"],
                                                     index=0,horizontal=True,label_visibility="collapsed",key="sankey_country_radio")
                with row_ctrl[1]: top_n_company = st.selectbox("Perusahaan",[5,8,10,15,20],index=2,key="sankey_top_company")
                with row_ctrl[2]: top_n_city    = st.selectbox("Kota",[5,8,10,15,20,30],index=2,key="sankey_top_city")
                with row_ctrl[3]: top_n_hotel   = st.selectbox("Hotel",[10,15,20,30,50],index=2,key="sankey_top_hotel")
                with row_ctrl[4]: chart_height  = st.selectbox("Tinggi Chart",[600,750,900,1100],index=1,key="sankey_height")

                if "Country" in df_overview.columns:
                    _df_sk = df_overview.copy()
                    _df_sk["_cu"] = _df_sk["Country"].astype(str).str.strip().str.upper()
                    if sankey_country_filter == "Domestik":
                        df_sankey = _df_sk[_df_sk["_cu"]=="INDONESIA"].drop(columns=["_cu"])
                        sankey_label = "Domestik"
                    else:
                        df_sankey = _df_sk[_df_sk["_cu"]!="INDONESIA"].drop(columns=["_cu"])
                        sankey_label = "Internasional"
                else:
                    df_sankey = df_overview.copy()
                    sankey_label = "Semua"

                if df_sankey.empty:
                    st.warning(f"Tidak ada data untuk filter: {sankey_label}")
                else:
                    df_sk = df_sankey.dropna(subset=["Nama Perusahaan",city_sankey_col,"Hotel Name","Invoice Amount"]).copy()

                    top_companies = df_sk.groupby("Nama Perusahaan")["Invoice Amount"].sum().nlargest(top_n_company).index
                    top_cities    = df_sk.groupby(city_sankey_col)["Invoice Amount"].sum().nlargest(top_n_city).index
                    top_hotels    = df_sk.groupby("Hotel Name")["Invoice Amount"].sum().nlargest(top_n_hotel).index

                    df_sk = df_sk[df_sk["Nama Perusahaan"].isin(top_companies) &
                                  df_sk[city_sankey_col].isin(top_cities) &
                                  df_sk["Hotel Name"].isin(top_hotels)]

                    if df_sk.empty:
                        st.warning("Tidak ada data setelah filter.")
                    else:
                        co_ci = df_sk.groupby(["Nama Perusahaan",city_sankey_col])["Invoice Amount"].sum().reset_index()
                        ci_ho = df_sk.groupby([city_sankey_col,"Hotel Name"])["Invoice Amount"].sum().reset_index()

                        companies = list(co_ci["Nama Perusahaan"].unique())
                        cities    = list(co_ci[city_sankey_col].unique())
                        hotels    = list(ci_ho["Hotel Name"].unique())
                        node_labels = companies+cities+hotels

                        co_idx = {c:i                         for i,c in enumerate(companies)}
                        ci_idx = {c:len(companies)+i           for i,c in enumerate(cities)}
                        ho_idx = {h:len(companies)+len(cities)+i for i,h in enumerate(hotels)}

                        sources,targets,values = [],[],[]
                        for _,row in co_ci.iterrows():
                            if row["Nama Perusahaan"] in co_idx and row[city_sankey_col] in ci_idx:
                                sources.append(co_idx[row["Nama Perusahaan"]]); targets.append(ci_idx[row[city_sankey_col]]); values.append(row["Invoice Amount"])
                        for _,row in ci_ho.iterrows():
                            if row[city_sankey_col] in ci_idx and row["Hotel Name"] in ho_idx:
                                sources.append(ci_idx[row[city_sankey_col]]); targets.append(ho_idx[row["Hotel Name"]]); values.append(row["Invoice Amount"])

                        PASTEL_POOL=["#B39DDB","#9575CD","#CE93D8","#F48FB1","#F06292","#A5D6A7","#66BB6A","#80CBC4","#90CAF9","#64B5F6","#FFCC80","#FFA726","#FFD54F","#B2EBF2","#B2DFDB","#DCEDC8","#F0F4C3","#FFF9C4"]
                        def assign_pastel(n,offset=0): return [PASTEL_POOL[(offset+i)%len(PASTEL_POOL)] for i in range(n)]
                        co_colors = assign_pastel(len(companies),0)
                        ci_colors = assign_pastel(len(cities),6)
                        ho_colors = assign_pastel(len(hotels),12)
                        node_colors = co_colors+ci_colors+ho_colors
                        link_colors = ["rgba(180,180,180,0.40)"]*len(sources)

                        fig_sankey = go.Figure(go.Sankey(
                            arrangement="snap",
                            textfont=dict(family="'Segoe UI',Arial,sans-serif",size=12,color="#222222"),
                            node=dict(pad=28,thickness=20,line=dict(color="rgba(0,0,0,0.12)",width=0.8),
                                      label=node_labels,color=node_colors),
                            link=dict(source=sources,target=targets,value=values,color=link_colors)
                        ))
                        fig_sankey.update_layout(
                            paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",height=chart_height,
                            margin=dict(l=16,r=16,t=52,b=16),
                            title=dict(text=f"<b>Sankey Flow: Perusahaan → Kota → Hotel</b> · Top {top_n_company} Perusahaan · Top {top_n_city} Kota · Top {top_n_hotel} Hotel · {sankey_label}",
                                       x=0.01,xanchor="left",font=dict(size=13,color="#333333"))
                        )
                        st.plotly_chart(fig_sankey, use_container_width=True)

                        output_sankey = BytesIO()
                        export_df = (df_sk.groupby(["Nama Perusahaan",city_sankey_col,"Hotel Name"])["Invoice Amount"]
                                     .sum().reset_index().sort_values("Invoice Amount",ascending=False))
                        export_df.to_excel(output_sankey,index=False,sheet_name="Sankey Flow")
                        output_sankey.seek(0)

                        if st.session_state.get("role") == "Admin":
                            st.download_button(label="Download Data",data=output_sankey,
                                file_name=f"sankey_flow_{sankey_label}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        else:
                            st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #9c5789;border-radius:6px;
                            padding:10px 16px;font-size:0.82em;color:#9c5789;'>Download hanya tersedia untuk Admin</div>""", unsafe_allow_html=True)

        # ======================================
        # TAB 8: DATA HOTEL
        # ======================================
        with tab8:
            st.markdown("<div class='section-title'>Data Hotel</div>", unsafe_allow_html=True)

            # =========================
            # FILTER DOMESTIK / INTERNATIONAL
            # =========================
            if "Country" in df_overview.columns:

                st.markdown("""
                <style>
                div[data-testid="stRadio"][data-key="tab8_country_radio"] > div[role="radiogroup"]{
                    display:inline-flex!important;
                    background:#9c5789;
                    border-radius:50px;
                    padding:3px;
                }
                div[data-testid="stRadio"][data-key="tab8_country_radio"] > div[role="radiogroup"] > label{
                    cursor:pointer;
                    padding:4px 16px!important;
                    border-radius:50px!important;
                    font-size:0.78em!important;
                    font-weight:500!important;
                    color:rgba(255,255,255,0.80)!important;
                    margin:0!important;
                }
                div[data-testid="stRadio"][data-key="tab8_country_radio"] > div[role="radiogroup"] > label > div:first-child{
                    display:none!important;
                }
                div[data-testid="stRadio"][data-key="tab8_country_radio"] 
                > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
                    background:white!important;
                    color:#9c5789!important;
                }
                div[data-testid="stRadio"][data-key="tab8_country_radio"] 
                > div[role="radiogroup"] > label:has(input:checked) > div:last-child p{
                    color:#9c5789!important;
                }
                div[data-testid="stRadio"][data-key="tab8_country_radio"] > label{
                    display:none!important;
                }
                </style>
                """, unsafe_allow_html=True)

                tab8_country_filter = st.radio(
                    label="filter_tab8",
                    options=["Domestik", "Internasional"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="tab8_country_radio"
                )

                # Copy dataframe
                _df_tab8 = df_overview.copy()

                # Normalisasi country
                _df_tab8["_country_up"] = (
                    _df_tab8["Country"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                # Alias domestik (jika data tidak konsisten)
                domestic_alias = ["INDONESIA", "ID", "IDN"]

                if tab8_country_filter == "Domestik":
                    df_tab8 = _df_tab8[_df_tab8["_country_up"].isin(domestic_alias)].copy()
                    tab8_label = "🇮🇩 Domestik"
                else:
                    df_tab8 = _df_tab8[~_df_tab8["_country_up"].isin(domestic_alias)].copy()
                    tab8_label = "🌍 Internasional"

                df_tab8.drop(columns=["_country_up"], inplace=True)

                if df_tab8.empty:
                    st.warning(f"⚠️ Tidak ada data untuk filter: {tab8_label}")
                    st.stop()

            else:
                df_tab8 = df_overview.copy()
                tab8_label = "Semua Data"

            # =========================
            # VISUALISASI
            # =========================
            cols1, cols2 = st.columns(2)

            # ==================================================
            # COL 1 — TOP 100 HOTEL
            # ==================================================
            with cols1:
                if "Hotel Name" in df_tab8.columns and "Number of Rooms Night" in df_tab8.columns:

                    top_hotels_tab = (
                        df_tab8.groupby("Hotel Name")["Number of Rooms Night"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(100)
                        .reset_index()
                    )

                    top_hotels_tab["Rank"] = top_hotels_tab.index + 1
                    top_hotels_tab["Highlight"] = top_hotels_tab["Rank"].apply(
                        lambda x: "Top 20" if x <= 20 else "Others"
                    )

                    fig_hotels = px.bar(
                        top_hotels_tab,
                        x="Number of Rooms Night",
                        y="Hotel Name",
                        orientation="h",
                        color="Highlight",
                        color_discrete_map={
                            "Top 20": "#9c5789",
                            "Others": "#e0e0e0"
                        },
                        title=f"Top 100 Hotels by Total Room Nights · {tab8_label}"
                    )

                    fig_hotels.update_traces(
                        texttemplate="%{x:,.0f}",
                        textposition="outside",
                        textfont_size=10
                    )

                    fig_hotels.update_layout(
                        height=1700,
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(size=10)
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=80, t=50, b=10)
                    )

                    st.plotly_chart(fig_hotels, use_container_width=True)

                    # Download
                    output_hotels = BytesIO()
                    top_hotels_tab.drop(columns=["Rank", "Highlight"]).to_excel(
                        output_hotels,
                        index=False,
                        sheet_name="Top 100 Hotels"
                    )
                    output_hotels.seek(0)

                    if st.session_state.get("role") == "Admin":
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_hotels,
                            file_name="top_100_hotels_by_room_nights.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border-left:3px solid #9c5789;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;'>
                        🔒 Download hanya tersedia untuk <strong>Admin</strong>
                        </div>
                        """, unsafe_allow_html=True)

            # ==================================================
            # COL 2 — TOP 100 CITY
            # ==================================================
            with cols2:

                city_col = next(
                    (c for c in ["City", "City Destination"] if c in df_tab8.columns),
                    None
                )

                if city_col and "Number of Rooms Night" in df_tab8.columns:

                    top_cities_tab = (
                        df_tab8.groupby(city_col)["Number of Rooms Night"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(100)
                        .reset_index()
                    )

                    top_cities_tab["Rank"] = top_cities_tab.index + 1
                    top_cities_tab["Highlight"] = top_cities_tab["Rank"].apply(
                        lambda x: "Top 20" if x <= 20 else "Others"
                    )

                    fig_cities_tab = px.bar(
                        top_cities_tab,
                        x="Number of Rooms Night",
                        y=city_col,
                        orientation="h",
                        color="Highlight",
                        color_discrete_map={
                            "Top 20": "#9c5789",
                            "Others": "#e0e0e0"
                        },
                        title=f"Top 100 Cities by Total Room Nights · {tab8_label}"
                    )

                    fig_cities_tab.update_traces(
                        texttemplate="%{x:,.0f}",
                        textposition="outside",
                        textfont_size=10
                    )

                    fig_cities_tab.update_layout(
                        height=1700,
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(size=10)
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=80, t=50, b=10)
                    )

                    st.plotly_chart(fig_cities_tab, use_container_width=True)

                    # Download
                    output_cities = BytesIO()
                    top_cities_tab.drop(columns=["Rank", "Highlight"]).to_excel(
                        output_cities,
                        index=False,
                        sheet_name="Top 100 Cities"
                    )
                    output_cities.seek(0)

                    if st.session_state.get("role") == "Admin":
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_cities,
                            file_name="top_100_cities_by_room_nights.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border-left:3px solid #9c5789;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#9c5789;'>
                        🔒 Download hanya tersedia untuk <strong>Admin</strong>
                        </div>
                        """, unsafe_allow_html=True)

        # ======================================
        # TAB 9: DENDROGRAM CLUSTERING
        # ======================================
        with tab9:
            from plotly.subplots import make_subplots as _make_subplots
            from scipy.cluster.hierarchy import linkage as sk_linkage, dendrogram as scipy_dendrogram, fcluster
            from sklearn.preprocessing import normalize as sk_normalize

            st.markdown("""
            <div style="background:linear-gradient(135deg,#6a1a5a 0%,#9c5789 55%,#7a3568 100%);
            border-radius:14px;padding:28px 32px;margin-bottom:24px;box-shadow:0 8px 32px rgba(156,87,137,0.28);">
                <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
                    <div style="background:rgba(255,255,255,0.14);border-radius:10px;padding:10px 14px;font-size:1.6em;">🌿</div>
                    <div>
                        <div style="color:#fff;font-size:1.45em;font-weight:700;">Hierarchical Clustering</div>
                        <div style="color:rgba(255,255,255,0.60);font-size:0.82em;margin-top:4px;">Dendrogram · Segmentasi Pola Perjalanan · Interactive</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            dend_r1 = st.columns([2,2,2,2])
            with dend_r1[0]: dend_entity = st.selectbox("🎯 Entitas",["Hotel","Perusahaan","Kota"],key="dend_entity")
            with dend_r1[1]: dend_metric_col = st.selectbox("📐 Metric",["Invoice Amount","Number of Rooms Night","Travel Request Number"],key="dend_metric")
            with dend_r1[2]: dend_method = st.selectbox("🔗 Linkage",["ward","complete","average","single"],key="dend_method")
            with dend_r1[3]: dend_top_n = st.selectbox("🔢 Top N",[10,15,20,25,30,40,50],index=2,key="dend_top_n")

            dend_r2 = st.columns([3,2])
            with dend_r2[0]: n_clusters_dend = st.slider("🎨 Jumlah Klaster",2,8,4,key="dend_clusters")
            with dend_r2[1]: dend_show_bar = st.checkbox("Bar spend chart",value=True,key="dend_bar")

            _city_col = next((c for c in df_all.columns if c in ["City","City Destination"]),None)
            entity_map_dend = {"Hotel":"Hotel Name","Perusahaan":"Nama Perusahaan","Kota":_city_col}
            pivot_rows_dend = {"Hotel":"Nama Perusahaan","Perusahaan":_city_col if _city_col else "Hotel Name","Kota":"Nama Perusahaan"}

            entity_col_dend  = entity_map_dend.get(dend_entity)
            row_col_dend     = pivot_rows_dend.get(dend_entity)
            metric_col_dend  = next((c for c in df_all.columns if c==dend_metric_col),None)

            if entity_col_dend and entity_col_dend in df_all.columns and metric_col_dend and row_col_dend and row_col_dend in df_all.columns:
                df_dend = df_all[[entity_col_dend,row_col_dend,metric_col_dend]].dropna()
                top_ent = df_dend.groupby(entity_col_dend)[metric_col_dend].sum().nlargest(dend_top_n).index
                df_dend = df_dend[df_dend[entity_col_dend].isin(top_ent)]
                pivot_dend = df_dend.groupby([entity_col_dend,row_col_dend])[metric_col_dend].sum().unstack(fill_value=0)
                X_dend = sk_normalize(pivot_dend.values,norm="l2")

                if X_dend.shape[0] >= 2:
                    Z_dend = sk_linkage(X_dend,method=dend_method,metric="euclidean")
                    cluster_ids_dend = fcluster(Z_dend,t=n_clusters_dend,criterion="maxclust")

                    DEND_PALETTE=["#9c5789","#5B8DD9","#3dab7a","#E07A3A","#c47a3d","#7a6b8a","#4AABB8","#D45E8A"]
                    DEND_PALETTE_LIGHT=["#f5e8f2","#dde8f8","#d5f2e5","#fae3d5","#fdf0e0","#ece8f0","#d5f0f5","#fae0eb"]

                    labels_list = pivot_dend.index.tolist()
                    leaf_colors_map_dend = {lbl:DEND_PALETTE[(cid-1)%len(DEND_PALETTE)] for lbl,cid in zip(labels_list,cluster_ids_dend)}

                    def fmt_val(v,col): return f"Rp{v:,.0f}" if col=="Invoice Amount" else f"{v:,.0f}"

                    total_spend_dend = df_dend[metric_col_dend].sum()
                    avg_spend_dend   = total_spend_dend/len(pivot_dend) if len(pivot_dend) else 0

                    kc1,kc2,kc3,kc4 = st.columns(4)
                    for col_k,lbl_k,val_k,accent_k in [
                        (kc1,"Total Entitas",str(len(pivot_dend)),"#9c5789"),
                        (kc2,"Jumlah Klaster",str(n_clusters_dend),"#7a6b8a"),
                        (kc3,"Total "+dend_metric_col[:12],fmt_val(total_spend_dend,metric_col_dend),"#5B8DD9"),
                        (kc4,"Avg per Entitas",fmt_val(avg_spend_dend,metric_col_dend),"#3dab7a"),
                    ]:
                        with col_k:
                            st.markdown(f"""<div style="background:white;border:1px solid #e8d5e4;border-top:3px solid {accent_k};
                            border-radius:10px;padding:16px 18px;box-shadow:0 2px 12px rgba(156,87,137,0.07);">
                            <div style="color:#9c8fa0;font-size:0.70em;text-transform:uppercase;letter-spacing:0.07em;">{lbl_k}</div>
                            <div style="color:#2a1a2a;font-size:1.25em;font-weight:700;margin-top:8px;">{val_k}</div></div>""",
                            unsafe_allow_html=True)

                    color_thresh_dend = Z_dend[-(n_clusters_dend-1),2] if n_clusters_dend > 1 else 0

                    dend_no_plot = scipy_dendrogram(Z_dend,labels=labels_list,no_plot=True,color_threshold=color_thresh_dend)
                    leaves_order_ply  = dend_no_plot["leaves"]
                    labels_ordered_ply= [labels_list[i] for i in leaves_order_ply]
                    leaf_xs = {lbl:5+10*i for i,lbl in enumerate(labels_ordered_ply)}

                    spend_series_ply = (df_dend.groupby(entity_col_dend)[metric_col_dend]
                                        .sum().reindex(labels_ordered_ply).fillna(0))

                    if dend_show_bar:
                        fig_ply = _make_subplots(rows=1,cols=2,column_widths=[0.68,0.32],horizontal_spacing=0.04,shared_yaxes=True)
                        dr,dc,br,bc = 1,1,1,2
                    else:
                        fig_ply = go.Figure()
                        dr=dc=br=bc=None

                    def _add(trace):
                        if dend_show_bar: fig_ply.add_trace(trace,row=dr,col=dc)
                        else: fig_ply.add_trace(trace)

                    for xi,yi in zip(dend_no_plot["icoord"],dend_no_plot["dcoord"]):
                        _add(go.Scatter(x=yi,y=xi,mode="lines",
                                        line=dict(color="rgba(156,87,137,0.35)",width=1.8),hoverinfo="skip",showlegend=False))

                    for lbl in labels_ordered_ply:
                        xp = leaf_xs[lbl]; nc = leaf_colors_map_dend.get(lbl,"#9c5789")
                        cid = cluster_ids_dend[labels_list.index(lbl)]
                        sv  = spend_series_ply.get(lbl,0)
                        _add(go.Scatter(x=[0],y=[xp],mode="markers+text",
                                        marker=dict(size=10,color=nc,line=dict(color="white",width=1.8),symbol="circle"),
                                        text=[lbl],textposition="middle left",
                                        textfont=dict(size=9,color=nc,family="'Segoe UI',Arial"),
                                        hovertemplate=f"<b>{lbl}</b><br>Klaster: <b>{cid}</b><br>{dend_metric_col}: <b>{fmt_val(sv,metric_col_dend)}</b><extra></extra>",
                                        showlegend=False))

                    if n_clusters_dend > 1:
                        y_r=[min(leaf_xs.values())-5,max(leaf_xs.values())+5]
                        _add(go.Scatter(x=[color_thresh_dend,color_thresh_dend],y=y_r,mode="lines",
                                        line=dict(color="#E05A2B",width=1.4,dash="dash"),
                                        name=f"Cut @ {color_thresh_dend:.3f}",showlegend=True))

                    if dend_show_bar:
                        bar_colors_ply = [leaf_colors_map_dend.get(l,"#9c5789") for l in labels_ordered_ply]
                        fig_ply.add_trace(go.Bar(y=[leaf_xs[l] for l in labels_ordered_ply],
                                                  x=spend_series_ply.values,orientation="h",
                                                  marker=dict(color=bar_colors_ply,opacity=0.82,line=dict(color="white",width=0.5)),
                                                  showlegend=False,name="Spend"),row=br,col=bc)

                    for i in range(n_clusters_dend):
                        fig_ply.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
                                                      marker=dict(size=10,color=DEND_PALETTE[i%len(DEND_PALETTE)]),
                                                      name=f"Klaster {i+1}",showlegend=True))

                    chart_h_ply = max(520,len(pivot_dend)*26+80)
                    max_diss = max((max(d) for d in dend_no_plot["dcoord"]),default=1.0)
                    x_left   = -max(0.55*max_diss,0.35)

                    fig_ply.update_layout(
                        height=chart_h_ply,paper_bgcolor="#fdf7fc",plot_bgcolor="#fdf7fc",
                        font=dict(family="'Segoe UI',Arial",size=11,color="#2a1a2a"),
                        title=dict(text=f"<b>Dendrogram — {dend_entity}</b>  Linkage: {dend_method} · {n_clusters_dend} Klaster · {dend_metric_col}",
                                   x=0.01,xanchor="left",font=dict(size=13,color="#2a1a2a")),
                        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                                    bgcolor="rgba(253,247,252,0.95)",bordercolor="#e8d5e4",borderwidth=1),
                        margin=dict(l=10,r=20,t=60,b=40),hovermode="closest",bargap=0.10
                    )

                    _ax_style=dict(showgrid=True,gridcolor="rgba(156,87,137,0.08)",gridwidth=0.5,
                                   zeroline=False,tickfont=dict(size=8,color="#9c8fa0"),linecolor="#e8d5e4",linewidth=1,showline=True)
                    fig_ply.update_xaxes(**_ax_style)
                    fig_ply.update_yaxes(**_ax_style)
                    fig_ply.update_xaxes(title_text="Dissimilarity",range=[x_left,max_diss*1.08],row=dr,col=dc)
                    fig_ply.update_yaxes(showticklabels=False,showgrid=False,row=dr,col=dc)
                    if dend_show_bar:
                        fig_ply.update_xaxes(title_text=dend_metric_col[:18],row=br,col=bc)
                        fig_ply.update_yaxes(showticklabels=False,showgrid=False,row=br,col=bc)

                    st.plotly_chart(fig_ply, use_container_width=True)

                    cluster_df_dend = pd.DataFrame({
                        dend_entity:labels_list,"Klaster":cluster_ids_dend,
                        "Total":(df_dend.groupby(entity_col_dend)[metric_col_dend].sum().reindex(labels_list).values)
                    }).sort_values(["Klaster","Total"],ascending=[True,False])

                    unique_clusters = sorted(cluster_df_dend["Klaster"].unique())
                    cluster_cols    = st.columns(min(4,len(unique_clusters)))

                    for i,cid in enumerate(unique_clusters):
                        c_color = DEND_PALETTE[(cid-1)%len(DEND_PALETTE)]
                        c_light = DEND_PALETTE_LIGHT[(cid-1)%len(DEND_PALETTE_LIGHT)]
                        sub_dend = cluster_df_dend[cluster_df_dend["Klaster"]==cid]
                        total_c  = sub_dend["Total"].sum()
                        pct_c    = total_c/total_spend_dend*100 if total_spend_dend else 0
                        members  = ", ".join(sub_dend[dend_entity].tolist()[:5])
                        more_n   = max(0,len(sub_dend)-5)
                        more_html= f"<br><span style='color:#b8a0b8;font-size:0.85em;'>+{more_n} lainnya</span>" if more_n>0 else ""
                        with cluster_cols[i%len(cluster_cols)]:
                            st.markdown(f"""<div style="background:linear-gradient(145deg,{c_light} 0%,#fff 100%);
                            border:1px solid {c_color}30;border-top:4px solid {c_color};border-radius:10px;padding:16px 18px;margin-bottom:12px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                <span style="font-weight:700;color:{c_color};font-size:0.95em;">Klaster {cid}</span>
                                <span style="background:{c_color}18;color:{c_color};border-radius:20px;padding:2px 10px;font-size:0.72em;font-weight:600;">{len(sub_dend)} entitas</span>
                            </div>
                            <div style="color:#4a3a4a;font-size:0.75em;margin-bottom:10px;line-height:1.6;">{members}{"..." if more_n>0 else ""}{more_html}</div>
                            <div style="background:{c_color}0f;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:1.0em;font-weight:700;color:{c_color};">{fmt_val(total_c,metric_col_dend)}</div>
                                <div style="font-size:0.70em;color:{c_color};font-weight:600;">{pct_c:.1f}% of all</div>
                            </div></div>""", unsafe_allow_html=True)

        # ======================================
        # TAB 9: EXPORT
        # ======================================
        with tab10:
            st.markdown("<div class='section-title'>Export Data</div>", unsafe_allow_html=True)
            st.markdown("Export your data in various formats for further analysis.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### CSV Format")
                st.markdown("Compatible with Excel, Google Sheets")
                if st.button("Download CSV", use_container_width=True, type="primary"):
                    csv = df_all.to_csv(index=False).encode("utf-8")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=csv,
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",use_container_width=True)
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #9c5789;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#9c5789;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            with col2:
                st.markdown("#### Excel Format")
                st.markdown("Microsoft Excel workbook")
                if st.button("Download Excel", use_container_width=True, type="primary"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df_all.to_excel(writer, index=False, sheet_name="Travel Data")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=buffer.getvalue(),
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #9c5789;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#9c5789;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            with col3:
                st.markdown("#### JSON Format")
                st.markdown("For APIs and data exchange")
                if st.button("Download JSON", use_container_width=True, type="primary"):
                    json_data = df_all.to_json(orient="records", date_format="iso")
                    if st.session_state.get('role') == 'Admin':
                        st.download_button("⬇️ Download",data=json_data,
                            file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",use_container_width=True)
                    else:
                        st.markdown("""<div style='background:#f9f9f9;border-left:3px solid #9c5789;border-radius:6px;
                        padding:10px 16px;font-size:0.82em;color:#9c5789;'>🔒 Download hanya tersedia untuk <strong>Admin</strong></div>""",
                        unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### Export Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Records", f"{len(df_all):,}")
            with col2: st.metric("Total Columns", f"{len(df_all.columns)}")
            with col3:
                memory_usage = df_all.memory_usage(deep=True).sum()/1024/1024
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            with col4: st.metric("Est. File Size", f"{memory_usage*0.8:.2f} MB")

    # ======================================
    # DISCLAIMER + FOOTER
    # ======================================
    st.markdown("""
    <div style="background:white;padding:20px;border-radius:4px;border-left:4px solid #9c5789;font-size:0.9em;">
        <b>Disclaimer &amp; Compliance Notice</b><br><br>
        Aplikasi ini disediakan untuk tujuan analisis internal. Output yang dihasilkan tidak bersifat final,
        tidak mengikat, dan harus melalui proses validasi serta persetujuan sesuai kebijakan perusahaan yang berlaku.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='divider' style='margin-top:50px;'></div>
    <div style='text-align:center;padding:25px;color:#888888;font-size:0.85em;'>
        © 2025 Dikembangkan oleh 
        <a href="https://www.linkedin.com/in/rifyalt/" target="_blank" style='color:#9c5789;text-decoration:none;font-weight:500;'>
            Rifyal Tumber
        </a> · MTRAX Travel Analytics
    </div>
    """, unsafe_allow_html=True)


# ===============================
# ROUTING — WAJIB PALING BAWAH
# ── DIMODIFIKASI: tambah pengecekan pending_2fa ──
# ===============================
if not st.session_state.get("authenticated"):

    if st.session_state.get("pending_2fa"):
        # ── Password sudah benar, tunggu OTP ─────────────────────────
        username = st.session_state.pending_user
        totp_secrets = _load_totp_secrets()

        # Semua user berbagi secret admin — enrolled jika [totp][admin] ada di secrets.toml
        enrolled_via_toml = False
        try:
            enrolled_via_toml = bool(st.secrets["totp"]["admin"])
        except Exception:
            pass

        # ── Hanya percaya secrets.toml sebagai bukti enrollment permanen.
        # session_state.totp_enrolled bersifat volatile (hilang saat restart),
        # sehingga tidak boleh dijadikan satu-satunya penentu.
        # User dianggap enrolled HANYA jika secret sudah ada di secrets.toml.
        already_enrolled = enrolled_via_toml

        if already_enrolled:
            twofa_verify_page(username)   # user lama → langsung input OTP
        else:
            # Pastikan totp_enrolled di-reset agar tidak menyebabkan
            # inkonsistensi pada sesi berikutnya
            st.session_state.totp_enrolled[username] = False
            twofa_setup_page(username)    # user baru → setup QR code dulu

    else:
        login_page()   # belum login sama sekali

else:
    check_session_timeout()
    main_app()
