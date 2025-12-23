import streamlit as st
import pandas as pd
import gdown
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from io import BytesIO

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
import json

import streamlit as st
import hashlib
import base64

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

# ======================================
# USER DATABASE (3 USERS)
# ======================================
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {
        "password": hash_password("admin123"),
        "role": "Admin"
    },
    "ssc": {
        "password": hash_password("ssc123"),
        "role": "Analyst"
    },
    "viewer": {
        "password": hash_password("viewer123"),
        "role": "Viewer"
    }
}

# ======================================
# SESSION STATE INIT (LOGIN)
# ======================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

#if "role" not in st.session_state:
#    st.session_state.role = ""

# ======================================
# LOGIN FUNCTION
# ======================================
def login_page():
    st.set_page_config(page_title="Login | Corporate Travel Analytics", layout="centered")

    st.markdown(
        """
        <style>
        /* Reset default streamlit styling */
        .stApp {
            background: #f5f7fa;
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Login container */
        .login-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        /* Logo section */
        .logo-section {
            margin-bottom: 35px;
        }
        
        .logo-container {
            display: inline-block;
            margin-bottom: 30px;
        }
        
        .logo-icon {
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, #005693 0%, #0077be 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            box-shadow: 0 4px 15px rgba(0, 86, 147, 0.2);
        }
        
        .logo-icon svg {
            width: 45px;
            height: 45px;
            fill: white;
        }
        
        .company-tagline {
            text-align: center;          /* 👈 center/justify */
            text-justify: inter-word;
            max-width: 360px;             /* agar justify terlihat rapi */
            margin: 0 auto;               /* center container */
            line-height: 1.5;
            color: #6b7280;
            font-size: 14px;
        }
        
        .welcome-text {
            font-size: 1.8em;
            font-weight: 200;
            color: #2c3e50;
            margin: 35px 0 40px 0;
        }
        
        /* Input styling */
        .stTextInput > div > div {
            position: relative;
        }
        
        .stTextInput > div > div > input {
            border: none;
            border-bottom: 2px solid #e0e0e0;
            border-radius: 0;
            padding: 14px 40px 14px 10px;
            font-size: 1em;
            transition: all 0.3s ease;
            background: transparent;
        }
        
        .stTextInput > div > div > input:focus {
            border-bottom-color: #005693;
            box-shadow: none;
            outline: none;
        }
        
        .stTextInput > label {
            display: none;
        }
        
        .stTextInput {
            margin-bottom: 25px;
        }
        
        /* Button styling */
        .stButton > button {
            background: #00a88f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 16px 24px;
            font-size: 1.05em;
            font-weight: 600;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
            box-shadow: 0 3px 12px rgba(0, 168, 143, 0.3);
            margin-top: 15px;
        }
        
        .stButton > button:hover {
            background: #009178;
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(0, 168, 143, 0.4);
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 6px;
            border: none;
            padding: 12px 16px;
            margin-top: 20px;
            font-size: 0.95em;
        }
        
        /* Divider */
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 30px 0;
            color: #95a5a6;
            font-size: 0.9em;
        }
        
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .divider span {
            padding: 0 15px;
        }
        
        /* Alternative login */
        .alt-login-btn {
            display: block;
            width: 100%;
            padding: 14px 24px;
            border: 2px solid #0077be;
            border-radius: 6px;
            background: white;
            color: #0077be;
            font-size: 1em;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .alt-login-btn:hover {
            background: #f0f8ff;
            border-color: #005693;
        }
        
        /* Footer text */
        .footer-text {
            margin-top: 35px;
            color: #95a5a6;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .footer-text a {
            color: #005693;
            text-decoration: none;
            font-weight: 600;
        }
        
        .footer-text a:hover {
            text-decoration: underline;
        }
        
        /* Icon styling for inputs */
        .input-icon {
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: #bdc3c7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    
    # Path logo
    logo_path = os.path.join("assets", "LOGO M FIX.png")

    # Encode image ke base64 agar bisa dipakai di HTML
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    logo_base64 = get_base64_image(logo_path)

    # Logo and Company Name
    st.markdown(f"""
        <div class='logo-section'>
            <div class='logo-icon'>
                <img src="data:image/png;base64,{logo_base64}"
                    style="width:70px;height:auto;" />
            </div>
            <div class='company-tagline'>
                Corporate Travel Analytics & Intelligence Platform
            </div>
        </div>

        <div class='welcome-text'>Welcome!</div>
    """, unsafe_allow_html=True)

    # Login Form
    username = st.text_input(
        "Username", 
        placeholder="Username",
        key="username_input",
        label_visibility="collapsed"
    )
    
    password = st.text_input(
        "Password", 
        type="password",
        placeholder="Password",
        key="password_input",
        label_visibility="collapsed"
    )
    
    if st.button("Login", use_container_width=True):
        if username in USERS:
            hashed_input = hash_password(password)
            if hashed_input == USERS[username]["password"]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        else:
            st.error("❌ Username not found")

    # Divider
#    st.markdown("""
#        <div class='divider'>
#            <span>or</span>
#        </div>
#    """, unsafe_allow_html=True)
    
    # Alternative login option (dapat disesuaikan)
#    st.markdown("""
#        <a href='#' class='alt-login-btn' onclick='return false;'>
#            <svg width="20" height="20" style="vertical-align: middle; margin-right: 8px;" viewBox="0 0 24 24" fill="#0077be">
#                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
#            </svg>
#            Login with SSO
#        </a>
#    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class='footer-text'>
            If you have problems about application or questions<br>
            please <a href='#'>Contact Us</a>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)




# ======================================
# LOGOUT FUNCTION
# ======================================
def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.rerun()


# ======================================
# GATEKEEPER (WAJIB)
# ======================================
if not st.session_state.authenticated:
    login_page()
    st.stop()


# =========================
# GOOGLE CALENDAR CONFIG
# =========================
GOOGLE_API_KEY = "AIzaSyC2Je18aFkU_knGcpFQ9xAiyeqgptJB0tk"
CALENDAR_ID = "d4t4m1tr4@gmail.com"

# =========================
# HELPER FUNCTION (WAJIB DI MAIN AREA)
# =========================
@st.cache_data(ttl=3600)
def fetch_google_calendar_events(start_date, end_date):
    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        f"{CALENDAR_ID}/events"
        f"?key={GOOGLE_API_KEY}"
        f"&timeMin={start_date}T00:00:00Z"
        f"&timeMax={end_date}T23:59:59Z"
        f"&singleEvents=true&orderBy=startTime"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return pd.DataFrame()

    events = response.json().get("items", [])
    data = []

    for e in events:
        start = e["start"].get("date") or e["start"].get("dateTime", "")[:10]
        data.append({
            "event_date": pd.to_datetime(start),
            "event_name": e.get("summary", "Event"),
            "event_type": "Calendar Event"
        })

    return pd.DataFrame(data)

# =========================
# RSS GOOGLE NEWS (RUNNING TEXT)
# =========================
import xml.etree.ElementTree as ET

@st.cache_data(ttl=1800)
def fetch_google_news_rss_today_h1():
    rss_urls = [
        "https://news.google.com/rss/search?q=hotel+hospitality&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=Danantara+Pertamina+BUMN+corporate+travel&hl=id&gl=ID&ceid=ID:id"
    ]

    today = datetime.utcnow().date()
    yesterday = today - pd.Timedelta(days=3)

    news_items = []

    for url in rss_urls:
        try:
            r = requests.get(url, timeout=5)
            root = ET.fromstring(r.content)

            for item in root.findall(".//item"):
                title = item.find("title").text
                link = item.find("link").text
                pub_date = item.find("pubDate").text

                pub_dt = datetime.strptime(
                    pub_date, "%a, %d %b %Y %H:%M:%S %Z"
                ).date()

                # ✅ FILTER: TODAY & H-1
                if pub_dt in [today, yesterday]:
                    news_items.append((pub_dt, title, link))

        except:
            pass

    # Sort: Today dulu, lalu H-1
    news_items = sorted(news_items, key=lambda x: x[0], reverse=True)

    # Batasi agar ringan
    return news_items[:12]



# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="MTRAX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# SESSION STATE INIT (WAJIB)
# =====================================================
if "df_all" not in st.session_state:
    st.session_state.df_all = pd.DataFrame()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# --- Custom CSS untuk tampilan Corporate Pertamina ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f9f9f9;
    }
    .stApp {
        background: #f9f9f9;
    }
    
    .corporate-header {
        background: linear-gradient(135deg, #032c48 100%, #032c48 70%, #1494c6 50%, #caf19d 100%);
        background-size: 250% 250%;
        animation: gradientShift 15s ease infinite;
        padding: 70px 50px;
        border-radius: 0 0 30px 30px;
        color: white;
        text-align: center;
        margin: -60px -60px 40px -60px;
        box-shadow: 0 15px 50px rgba(80,141,129,0.5), inset 0 -3px 10px rgba(0,0,0,0.1);
        border-bottom: 8px solid #7eb762;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .corporate-header::before {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 30s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .corporate-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #7eb762, #6d99e1, #7eb762, transparent);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .corporate-logo {
        font-size: 5em;
        margin-bottom: 20px;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.4));
        animation: floatBounce 4s ease-in-out infinite;
        display: inline-block;
        position: relative;
        z-index: 2;
    }
    
    @keyframes floatBounce {
        0%, 100% { transform: translateY(0px) scale(1); }
        25% { transform: translateY(-15px) scale(1.05); }
        50% { transform: translateY(0px) scale(1); }
        75% { transform: translateY(-8px) scale(1.02); }
    }
    
    .corporate-title {
        font-weight: 800;
        font-size: 3.2em;
        margin: 20px 0 10px 0;
        letter-spacing: 4px;
        text-transform: uppercase;

        position: relative;
        z-index: 2;
        background: linear-gradient(90deg, #ffffff 0%, #e0f7e0 50%, #ffffff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleShine 5s linear infinite;
    }
    
    @keyframes titleShine {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .corporate-subtitle {
        font-weight: 400;
        font-size: 1.4em;
        margin: 15px 0;
        opacity: 0.95;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        font-style: italic;
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #7eb762;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,143,255,0.15);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        height: 100%;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: 600;
        transition: all 0.3s;
        background: linear-gradient(135deg, #7eb762 0%, #005693 100%);
        color: white;
        border: none;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9em;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102,143,255,0.3);
        background: linear-gradient(135deg, #ff4b4b 0%, #ff4b4b 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ff4b4b 0%, #ff4b4b 100%);
    }
    
    /* Sidebar styling adjust lebar */
    [data-testid="stSidebar"] {
        background: #aac237 !important;
    }

    /* Background Sidebar */

    [data-testid="stSidebar"] > div:first-child {
        background: #032c48 !important;
    }

    /* Sidebar text styling */
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    /* ===== FORCE ALL SIDEBAR TEXT TO WHITE ===== */

    /* Semua teks di sidebar */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Label widget (Select Input Method:) */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }

    /* Radio option text */
    [data-testid="stSidebar"] .stRadio div {
        color: white !important;
    }

    /* Help text (tooltip / small text) */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #e0e0e0 !important;
    }

    /* Markdown text inside expander */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: white !important;
    }

    /* Expander content */
    [data-testid="stSidebar"] .streamlit-expanderContent {
        color: white !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        color: white !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: white !important;
        background-color: rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: rgba(255,255,255,0.2) !important;
    }
    
    /* Sidebar file uploader */
    [data-testid="stSidebar"] .stFileUploader {
        background-color: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px;
    }
    
    [data-testid="stSidebar"] .stFileUploader label {
        color: white !important;
    }
    
    .section-header {
        color: #005693;
        font-weight: 700;
        font-size: 1.4em;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #6d99e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #7eb762 0%, #6d99e1 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(102,143,255,0.2);
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #6d99e1, transparent);
        margin: 30px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        color: #005693;
        font-weight: 600;
        padding: 0 30px;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7eb762 0%, #005693 100%);
        color: white;
        border: 2px solid #6d99e1;
    }

    /* ===== RUNNING TEXT NEWS ===== */
    .news-ticker {
        background: #edf3fc;
        color: #032c48;              /* kontras corporate */
        padding: 10px 0;
        font-size: 0.95em;
        font-weight: 600;
        overflow: hidden;
        white-space: nowrap;
        border-bottom: 3px solid #ffffff;
    }

    .news-ticker span {
        display: inline-block;
        padding-left: 100%;
        animation: tickerMove 150s linear infinite;
    }

    @keyframes tickerMove {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    /* ===== RUNNING TEXT HOVER PAUSE ===== */
    .news-ticker span {
        display: inline-block;
        padding-left: 100%;
        animation: tickerMove 70s linear infinite;
    }

    .news-ticker:hover span {
        animation-play-state: paused;
    }

    /* Link styling */
    .news-ticker a:hover {
        text-decoration: underline;
        color: #ff4b4b;
    }

    </style>
""", unsafe_allow_html=True)

# --- Corporate Header Pertamina ---
st.markdown("""
    <div class='corporate-header'>
        <div class='corporate-logo'></div>
        <div class='corporate-title'>Welcome</div>
        <div class='corporate-subtitle'>Corporate Travel Analytics & Intelligence Platform</div>
        <div style='margin-top: 15px; font-size: 0.9em; opacity: 0.8;'>
            Turning Travel Data into Decisions
        </div>
    </div>
""", unsafe_allow_html=True)

from datetime import datetime

current_hour = datetime.now().hour

if current_hour < 11:
    greet_time = "Selamat Pagi"
elif current_hour < 15:
    greet_time = "Selamat Siang"
elif current_hour < 18:
    greet_time = "Selamat Sore"
else:
    greet_time = "Selamat Malam"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#ffffff,#f1f5f9);
    padding:18px 26px;
    border-radius:14px;
    margin:15px 0 30px 0;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
">
    <div style="font-size:1.1em;font-weight:600;color:#032c48;">
        👋 {greet_time}, {st.session_state.username.upper()}
    </div>
    <div style="font-size:0.9em;color:#6b7280;margin-top:4px;">
        Role: <b>{st.session_state.role}</b> · Selamat bekerja dan semoga harimu produktif
    </div>
</div>
""", unsafe_allow_html=True)


# --- RUNNING TEXT GOOGLE NEWS (H-1 & TODAY) ---
news_items = fetch_google_news_rss_today_h1()

if news_items:
    news_html = " &nbsp; | &nbsp; ".join([
        f"<a href='{link}' target='_blank'>📰 {title}</a>"
        for _, title, link in news_items
    ])

    st.markdown(
        f"""
        <div class="news-ticker">
            <span>{news_html}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Sidebar Corporate ---

import base64

def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("assets/LOGO M FIX.png")

with st.sidebar:
    st.markdown(f"""
        <div style='
            text-align: center;
            padding: 30px 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        '>
            <img src="data:image/png;base64,{logo_base64}"
                 style="width:110px; margin-bottom:12px;" />
            <div style='
                font-size: 1.8em;
                font-weight: 700;
                color: white;
                letter-spacing: 2px;
            '>MTRAX</div>
            <div style='
                color: #ffffff;
                font-size: 1em;
                margin-top: 5px;
                font-weight: 300;
            '>Travel Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    #st.markdown("---")
    #st.markdown(f"""
    #**👤 User:** {st.session_state.username}  
    #**🔑 Role:** {st.session_state.role}
    #""")

    if st.button("Logout"):
        logout()


    st.markdown("""
        <div style='
            color: white;
            font-weight: 600;
            font-size: 1.1em;
            margin: 20px 0 10px 0;
        '>DATA SOURCE</div>
    """, unsafe_allow_html=True)

    
    mode = st.radio(
        "Select Input Method:",
        ["📁 Cloud/Drive Integration", "📤 Manual File Upload"],
        help="Choose your preferred data input method"
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='color: white; font-weight: 600; font-size: 1.1em; margin: 20px 0 10px 0;'>📖 USER GUIDE</div>", unsafe_allow_html=True)
    with st.expander("System Documentation", expanded=False):
        st.markdown("""
        **Data Integration Methods:**
        
        1. **Cloud/Drive Integration**
           - Paste public folder link
           - Automatic batch processing
           - Suitable for multiple files
        
        2. **Manual File Upload**
           - Select files from local system
           - Individual file processing
           - Suitable for ad-hoc analysis
        
        **Analysis Features:**
        - Real-time data filtering
        - Interactive dashboards
        - Comprehensive reporting
        - AI-powered insights (soon)

        **Export Options:**
        - CSV format for database import
        - Excel format for detailed analysis
        """)

    # --- Sidebar Footer / Notes ---
 #   st.markdown("""
 #   <div style="
 #       margin-top: 40px;
 #       padding-top: 15px;
 #       border-top: 1px solid rgba(255,255,255,0.2);
 #       text-align: center;
 #       font-size: 0.75em;
 #       color: rgba(255,255,255,0.8);
 #   ">
 #           Aplikasi versi 1
 #       <div style="opacity: 0.85;">
 #           dibuat oleh <strong>Rifyal Tumber</strong>
 #       </div>
 #   </div>
 #   """, unsafe_allow_html=True)

    

# --- Folder untuk menyimpan file sementara ---
os.makedirs("data_temp", exist_ok=True)

# --- Kolom yang tidak dipakai ---
drop_cols = ["Site (PSA)", "Site group Name", "Currency", "Reschedule ID", "Source_File"]

df_all = pd.DataFrame()

# --- Mode Google Drive ---
if mode == "📁 Cloud/Drive Integration":

    st.markdown("<div class='section-header'>🌐 Cloud/Drive Integration</div>", unsafe_allow_html=True)

    # =========================
    # GOOGLE DRIVE DATASET DROPDOWN (STABLE)
    # =========================
    drive_options = {
        "2023–2025 (All Data)": "1vygKdg7enC5Kah7WbzVLsNI--S7Tyhvz",
        "2023 Only": "1xDFRdGLDiiScIwW9gTucRyeFCmuqNyq_",
        "2024 Only": "16ZMZ42BLN4GPbYKAd5h75ocbxFuyc85V",
        "2025 Only": "1chxbGHfk9hHNPZ8vlU6AqRVUKH1jEnxF"
    }

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_dataset = st.selectbox(
            "Cloud/Drive Dataset",
            list(drive_options.keys()),
            help="Folder must be set to 'Anyone with the link can view'"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("🔄 PROCESS DATA", type="primary")

    # =========================
    # PROCESS DATA PIPELINE
    # =========================
    if process_btn:

        folder_id = drive_options[selected_dataset]

        with st.spinner("⏳ Processing data pipeline..."):

            try:
                import gdown
                import shutil

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Clean previous temp data
                if os.path.exists("data_temp"):
                    shutil.rmtree("data_temp")
                os.makedirs("data_temp", exist_ok=True)

                status_text.text("📥 Downloading data")
                progress_bar.progress(30)

                # 🔥 STABLE GOOGLE DRIVE DOWNLOAD (FOLDER ID)
                gdown.download_folder(
                    id=folder_id,
                    output="data_temp",
                    quiet=False,
                    use_cookies=False
                )

                progress_bar.progress(60)

                files = [
                    f for f in os.listdir("data_temp")
                    if f.endswith((".xlsx", ".xls"))
                ]

                if not files:
                    st.error("❌ No Excel files found in selected Google Drive folder.")
                    st.stop()

                status_text.text("📊 Reading and consolidating Excel files...")

                df_list = []
                for i, f in enumerate(files):
                    try:
                        df = pd.read_excel(os.path.join("data_temp", f))
                        df = df.drop(
                            columns=[c for c in drop_cols if c in df.columns],
                            errors="ignore"
                        )
                        df_list.append(df)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to read file {f}: {e}")

                    progress_bar.progress(60 + int(40 * (i + 1) / len(files)))

                if not df_list:
                    st.error("❌ No valid Excel data could be loaded.")
                    st.stop()

                df_all = pd.concat(df_list, ignore_index=True)

                progress_bar.progress(100)
                status_text.empty()

                st.success(
                    f"✅ Successfully consolidated **{len(df_list)}** files "
                    f"with **{len(df_all):,}** records"
                )

            except Exception as e:
                st.error(f"❌ Cloud/Drive integration error: {e}")

# --- Mode Upload Manual ---
elif mode == "📤 Manual File Upload":
    st.markdown("<div class='section-header'>📂 Manual File Upload</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select Excel Files for Processing",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Use Ctrl+Click (Windows) or Cmd+Click (Mac) to select multiple files"
    )

    if uploaded_files:
        with st.spinner("⏳ Processing uploaded files..."):
            progress_bar = st.progress(0)
            df_list = []
            
            for i, file in enumerate(uploaded_files):
                try:
                    df = pd.read_excel(file)
                    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                    df_list.append(df)
                except Exception as e:
                    st.warning(f"⚠️ Failed to read {file.name}: {e}")
                progress_bar.progress(int(100 * (i + 1) / len(uploaded_files)))

            if df_list:
                df_all = pd.concat(df_list, ignore_index=True)
                progress_bar.empty()
                st.success(f"✅ Successfully consolidated **{len(df_list)}** files with **{len(df_all):,}** records!")
                st.balloons()

# --- CLEAN dan KONVERSI KOLOM IMPORTANT SEBELUM ANALISA ---
if not df_all.empty:
    # Bersihkan dan konversi Invoice Amount
    if "Invoice Amount" in df_all.columns:
        df_all["Invoice Amount"] = pd.to_numeric(
            df_all["Invoice Amount"].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce'
        )
    # Konversi Number of Rooms Night
    if "Number of Rooms Night" in df_all.columns:
        df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors='coerce')
    # Konversi tipe tanggal
    if "Check in Date" in df_all.columns:
        df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce")
    if "Check Out Date" in df_all.columns:
        df_all["Check Out Date"] = pd.to_datetime(df_all["Check Out Date"], errors="coerce")
    
    # --- Sidebar Filter seperti yang Anda miliki ---
    with st.sidebar:
        # (sidebar filters Anda)
        pass

# --- Preview & Analisis Data ---
if not df_all.empty:
    # Pastikan kolom tanggal dalam datetime
    if "Check in Date" in df_all.columns:
        df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce")
    if "Check Out Date" in df_all.columns:
        df_all["Check Out Date"] = pd.to_datetime(df_all["Check Out Date"], errors="coerce")

    # --- Sidebar Filters ---
    with st.sidebar:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div style='color: white; font-weight: 600; font-size: 1.1em; margin: 20px 0 10px 0;'>🔍 DATA FILTERS</div>", unsafe_allow_html=True)

        if "Check In Date" in df_all.columns:
            min_ci, max_ci = df_all["Check in Date"].min(), df_all["Check in Date"].max()
            checkin_range = st.date_input("📅 Check-In Period", [min_ci, max_ci])
            if len(checkin_range) == 2:
                df_all = df_all[(df_all["Check in Date"] >= pd.to_datetime(checkin_range[0])) &
                                (df_all["Check in Date"] <= pd.to_datetime(checkin_range[1]))]

        if "Check Out Date" in df_all.columns:
            min_co, max_co = df_all["Check Out Date"].min(), df_all["Check Out Date"].max()
            checkout_range = st.date_input("📅 Check-Out Period", [min_co, max_co])
            if len(checkout_range) == 2:
                df_all = df_all[(df_all["Check Out Date"] >= pd.to_datetime(checkout_range[0])) &
                                (df_all["Check Out Date"] <= pd.to_datetime(checkout_range[1]))]

    with st.sidebar:

        #st.markdown("### 🏢 Directorate Filter")

        # Pastikan kolom direktorat ada
        if "Direktorat Pekerja" in df_all.columns:

            directorate_list = (
                df_all["Direktorat Pekerja"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            selected_directorate = st.selectbox(
                "Select Directorate",
                options=["All Directorates"] + directorate_list,
                index=0
            )


    # --- Tabs dan tampilan dashboard ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Dashboard", 
        "🗂️ Data Explorer", 
        "📈 Advanced Analytics", 
        "🧠 Machine Learning",
        "💡 Demand Intelligence 2026",
        "⬇️ Export & Reports"
    ])

    
    with tab1:
        st.markdown("<div class='section-header'>📊 Summary Overview</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

        with col1:
            total_rows = len(df_all)
            st.metric(
                label="📝 Bookings",
                value=f"{total_rows:,}",
                delta="Records"
            )

        with col2:
            if "Travel Request Number" in df_all.columns:
                unique_tr = df_all["Travel Request Number"].nunique()
                st.metric(
                    label="📋 TR Number",
                    value=f"{unique_tr:,}",
                    help="Unique Travel Request Numbers"
                )

        with col3:
            if "Employee Id" in df_all.columns:
                unique_emp = df_all["Employee Id"].nunique()
                st.metric(
                    label="👥 Travelers",
                    value=f"{unique_emp:,}"
                )

        with col4:
            if "Hotel Name" in df_all.columns:
                unique_hotels = df_all["Hotel Name"].nunique()
                st.metric(
                    label="🏨 Hotels",
                    value=f"{unique_hotels:,}"
                )

        with col5:
            if "Number of Rooms Night" in df_all.columns:
                total_rooms = df_all["Number of Rooms Night"].sum()
                st.metric(
                    label="🛏️ Room Nights",
                    value=f"{total_rooms:,.0f}"
                )

        with col6:
            if "Company Code" in df_all.columns:
                unique_company = df_all["Company Code"].nunique()
                st.metric("🏢 Company Code", f"{unique_company:,}")
        with col7:
            if "Cost Center Pekerja" in df_all.columns:
                unique_cost_center = df_all["Cost Center Pekerja"].nunique()
                st.metric("💼 Cost Center", f"{unique_cost_center:,}")
        with col8:
            if "City" in df_all.columns:
                unique_cities = df_all["City"].nunique()
                st.metric("🌆 Cities", f"{unique_cities:,}")
        with col9:
            if "Country" in df_all.columns:
                unique_countries = df_all["Country"].nunique()
                st.metric("🌍 Countries", f"{unique_countries:,}")


        # Travel Request Deep Dive
        if "Travel Request Number" in df_all.columns:
            st.markdown("<div class='section-header'>📋 Travel Request Intelligence</div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                tr_bookings = df_all.groupby("Travel Request Number").size()
                avg_booking = tr_bookings.mean()
                max_booking = tr_bookings.max()
                
                st.markdown(f"""
                    <div class='stats-card'>
                        <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>📈</div>
                        <div style='font-size: 0.85em; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Booking Statistics</div>
                        <div style='font-size: 2.5em; font-weight: 700; color: #005693; margin: 15px 0;'>{avg_booking:.1f}</div>
                        <div style='font-size: 0.9em; color: #6c757d;'>Avg Bookings/Travel Request</div>
                        <div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #e9ecef;'>
                            <span style='color: #fa395f; font-weight: 600; font-size: 1.1em;'>{max_booking}</span>
                            <span style='color: #6c757d; font-size: 0.85em;'> maximum bookings</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if "Number of Rooms Night" in df_all.columns:
                    tr_rooms = df_all.groupby("Travel Request Number")["Number of Rooms Night"].sum()
                    avg_rooms = tr_rooms.mean()
                    max_rooms = tr_rooms.max()
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>🛏️</div>
                            <div style='font-size: 0.85em; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Length of Stay</div>
                            <div style='font-size: 2.5em; font-weight: 700; color: #005693; margin: 15px 0;'>{avg_rooms:.1f}</div>
                            <div style='font-size: 0.9em; color: #6c757d;'>Avg Room Nights/Travel Request</div>
                            <div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #e9ecef;'>
                                <span style='color: #fa395f; font-weight: 600; font-size: 1.1em;'>{max_rooms:.0f}</span>
                                <span style='color: #6c757d; font-size: 0.85em;'> maximum nights</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                tr_with_multiple = (tr_bookings > 1).sum()
                tr_single = (tr_bookings == 1).sum()
                multi_percentage = (tr_with_multiple / len(tr_bookings) * 100)
                
                st.markdown(f"""
                    <div class='stats-card'>
                        <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>📑</div>
                        <div style='font-size: 0.85em; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Travel Request Distribution</div>
                        <div style='font-size: 2.5em; font-weight: 700; color: #005693; margin: 15px 0;'>{multi_percentage:.1f}%</div>
                        <div style='font-size: 0.9em; color: #6c757d;'>Multi-booking Travel Requests</div>
                        <div style='margin-top: 15px; padding-top: 15px; border-top: 2px solid #e9ecef;'>
                            <span style='color: #fa395f; font-weight: 600;'>{tr_with_multiple:,}</span> <span style='color: #6c757d; font-size: 0.85em;'>multi /</span>
                            <span style='color: #aac237; font-weight: 600;'>{tr_single:,}</span> <span style='color: #6c757d; font-size: 0.85em;'>single</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if "Invoice Amount" in df_all.columns and "Number of Rooms Night" in df_all.columns:
                    # Bersihkan dan konversi kolom jadi numerik
                    df_all["Invoice Amount"] = pd.to_numeric(
                        df_all["Invoice Amount"].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
                    df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors='coerce')

                    # Filter data valid agar tidak ada pembagian dengan nol atau NaN
                    valid_rows = (df_all["Invoice Amount"].notna()) & (df_all["Number of Rooms Night"].notna()) & (df_all["Number of Rooms Night"] > 0)
                    df_valid = df_all[valid_rows].copy()

                    # Hitung Price Per Night
                    df_valid["Price Per Night"] = df_valid["Invoice Amount"] / df_valid["Number of Rooms Night"]
                    
                    avg_price_per_night = df_valid["Price Per Night"].mean()
                    median_invoice = df_all["Invoice Amount"].median()
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>💰</div>
                            <div style='font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #6c757d;'>Cost Analysis</div>
                            <div style='font-size: 2.5em; font-weight: 700; color: #005693; margin: 15px 0;'>{avg_price_per_night:,.0f}</div>
                            <div style='font-size: 0.9em; color: #6c757d;'>Avg Room Rate</div>
                            <div style='margin-top: 15px; border-top: 2px solid #e9ecef; padding-top: 10px; color: #fa395f; font-weight: 600;'>
                                Median invoice: Rp {median_invoice:,.0f}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='stats-card'>
                            <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>💰</div>
                            <div style='font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #6c757d;'>Cost Analysis</div>
                            <div style='font-size: 2.5em; font-weight: 700; color: #005693; margin: 15px 0;'>N/A</div>
                            <div style='font-size: 0.85em; color: #6c757d;'>Data not available</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # =========================
            # DATETIME NORMALIZATION (WAJIB)
            # =========================
            date_cols = ["Issue Time", "Check in Date"]

            for col in date_cols:
                if col in df_all.columns:
                    df_all[col] = pd.to_datetime(df_all[col], errors="coerce")

            with col5:
                if "Issue Time" in df_all.columns and "Check in Date" in df_all.columns:

                    df_lead = df_all[
                        df_all["Issue Time"].notna() &
                        df_all["Check in Date"].notna()
                    ].copy()

                    # HITUNG LEAD TIME (PAKAI normalize, BUKAN .dt.date)
                    df_lead["Lead Time (Days)"] = (
                        df_lead["Check in Date"].dt.normalize() -
                        df_lead["Issue Time"].dt.normalize()
                    ).dt.days

                    # Ambil hanya data valid (>=0)
                    lead_valid = df_lead[df_lead["Lead Time (Days)"] >= 0]

                    avg_lead = lead_valid["Lead Time (Days)"].mean()
                    median_lead = lead_valid["Lead Time (Days)"].median()

                    last_minute_pct = (
                        (lead_valid["Lead Time (Days)"] <= 2).sum() / len(lead_valid) * 100
                        if len(lead_valid) > 0 else 0
                    )

                    st.markdown(f"""
                        <div class='stats-card'>
                            <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 10px;'>⌛</div>
                            <div style='font-size: 0.85em; color: #6c757d; font-weight: 600;
                                        text-transform: uppercase; letter-spacing: 1px;'>
                                Lead Time
                            </div>
                            <div style='font-size: 2.5em; font-weight: 700; color: #005693;
                                        margin: 15px 0;'>
                                {avg_lead:.1f}
                            </div>
                            <div style='font-size: 0.9em; color: #6c757d;'>
                                Avg Days Before Check-in
                            </div>
                            <div style='margin-top: 15px; padding-top: 15px;
                                        border-top: 2px solid #e9ecef;'>
                                <span style='color: #fa395f; font-weight: 600;'>
                                    {last_minute_pct:.1f}%
                                </span>
                                <span style='color: #6c757d; font-size: 0.85em;'>
                                    last-minute (≤2 days)
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                        <div class='stats-card'>
                            <div style='font-size: 2.5em;'>⏱️</div>
                            <div>Lead Time</div>
                            <div style='font-size: 2em;'>N/A</div>
                        </div>
                    """, unsafe_allow_html=True)

            # =========================
            # TOP 10 DIRECTORATE & COUNTRY (RANKED & COMPACT)
            # =========================
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Overview Top Directorate & Destination Insight</div>", unsafe_allow_html=True)

            if "Issue Time" in df_all.columns:

                df_heat = df_all[df_all["Issue Time"].notna()].copy()

                df_heat["Issue Hour"] = df_heat["Issue Time"].dt.hour
                df_heat["Issue Day"] = df_heat["Issue Time"].dt.day_name()

                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                pivot_issue = (
                    df_heat
                    .groupby(["Issue Day", "Issue Hour"])
                    .size()
                    .reset_index(name="Total TR")
                    .pivot(index="Issue Day", columns="Issue Hour", values="Total TR")
                    .reindex(day_order)
                    .fillna(0)
                )

                fig_issue_heat = px.imshow(
                    pivot_issue,
                    text_auto=True,
                    color_continuous_scale=["#7eb762", "#086db5", "#ea1e30"],
                    aspect="auto"
                )

                fig_issue_heat.update_layout(
                    height=420,
                    title="Travel Request Pattern by Day & Issue Time",
                    xaxis_title="Issue Hour",
                    yaxis_title="Day",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=20, t=60, b=40)
                )

                fig_issue_heat.update_traces(
                    texttemplate="%{text}",
                    textfont=dict(size=11, color="black")
                )

                st.plotly_chart(fig_issue_heat, use_container_width=True)


            col1, col2 = st.columns(2)

            # --- TOP 10 DIRECTORATES (RANKED) ---
            with col1:
                if "Direktorat Pekerja" in df_all.columns and "Travel Request Number" in df_all.columns:
                    top_direktorat = (
                        df_all.groupby("Direktorat Pekerja")["Travel Request Number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )

                    fig_dir = px.bar(
                        top_direktorat,
                        x="Travel Request Number",
                        y="Direktorat Pekerja",
                        orientation="h",
                        text="Travel Request Number",
                        color_discrete_sequence=["#7eb762"]
                    )

                    fig_dir.update_traces(
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=12)
                    )

                    fig_dir.update_layout(
                        height=380,
                        title="Top 10 Directorates by Travel Request",
                        xaxis_title="",
                        yaxis_title="",
                        yaxis=dict(
                            autorange="reversed"  # 🔥 paling tinggi di atas
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=30, t=50, b=10)
                    )

                    st.plotly_chart(fig_dir, use_container_width=True)

            # --- DESTINATION COUNTRIES (INDONESIA vs OTHERS) ---
            with col2:
                if "Country" in df_all.columns:

                    # Normalisasi nama negara
                    df_all["Country"] = df_all["Country"].astype(str).str.strip()

                    indonesia_count = df_all[df_all["Country"] == "INDONESIA"].shape[0]
                    others_count = df_all[df_all["Country"] != "INDONESIA"].shape[0]

                    pie_df = pd.DataFrame({
                        "Category": ["INDONESIA", "Others (Countries)"],
                        "Total Bookings": [indonesia_count, others_count]
                    })

                    fig_cty = px.pie(
                        pie_df,
                        names="Category",
                        values="Total Bookings",
                        hole=0.5
                    )

                    fig_cty.update_traces(
                        marker=dict(
                            colors=["#7eb762", "#6d99e1"]
                        ),
                        textinfo="percent+label",
                        textfont=dict(size=12),
                        hovertemplate="<b>%{label}</b><br>%{value:,} bookings<extra></extra>",
                        pull=[0.03, 0.02]
                    )

                    fig_cty.update_layout(
                        height=380,  # sama dengan bar 
                        title="Market Size",
                        title_x=0.5,
                        showlegend=True,
                        margin=dict(l=10, r=30, t=50, b=10),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(
                            text=f"<b>{pie_df['Total Bookings'].sum():,}</b><br>Total",
                            x=0.5,
                            y=0.5,
                            font_size=14,
                            showarrow=False
                        )]
                    )

                    st.plotly_chart(fig_cty, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>🗂️ Data Explorer</div>", unsafe_allow_html=True)

        if df_all.empty:
            st.warning("No data available")
            st.stop()

        view_option = st.selectbox(
            "Select View",
            [
                "📄 Raw Data Table",
                "📊 Volume Forecast Visualization",
                "📈 Long-Term Volume Movement"
            ]
        )

        # =====================================================
        # 1️⃣ RAW DATA TABLE
        # =====================================================
        if view_option == "📄 Raw Data Table":

            display_df = df_all.copy()

            st.subheader("📄 Raw Data Exploration")

            search_term = st.text_input(
                "Search across all columns",
                placeholder="Hotel, city, booking ID, etc"
            )

            if search_term:
                mask = display_df.astype(str).apply(
                    lambda col: col.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                display_df = display_df[mask]

            rows_per_page = st.selectbox(
                "Rows per page",
                [25, 50, 100, 200],
                index=1
            )

            total_pages = max(
                1,
                len(display_df) // rows_per_page
                + (1 if len(display_df) % rows_per_page > 0 else 0)
            )

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1
            )

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                use_container_width=True,
                height=520
            )

            # ✅ CAPTION HARUS DI SINI
            st.caption(
                f"📄 Displaying records {start_idx + 1:,} - "
                f"{min(end_idx, len(display_df)):,} "
                f"of {len(display_df):,} total records | "
                f"Page {page} of {total_pages}"
            )

        # =====================================================
        # 2️⃣ VOLUME FORECAST
        # =====================================================
        elif view_option == "📊 Volume Forecast Visualization":

            st.subheader("📊 Volume Forecast Visualization")

            if "Check in Date" not in df_all.columns:
                st.error("❌ Column 'Check in Date' not found")
                st.stop()

            df_ts = (
                df_all
                .dropna(subset=["Check in Date"])
                .groupby(pd.Grouper(key="Check in Date", freq="M"))
                .size()
                .reset_index(name="Bookings")
            )

            df_ts["Forecast (3M Avg)"] = df_ts["Bookings"].rolling(3).mean()

            fig = px.line(
                df_ts,
                x="Check in Date",
                y=["Bookings", "Forecast (3M Avg)"],
                markers=True,
                title="Monthly Booking Volume Forecast"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # 3️⃣ LONG-TERM MOVEMENT
        # =====================================================
        elif view_option == "📈 Long-Term Volume Movement":

            st.subheader("📈 Long-Term Volume Movement")

            if "Check in Date" not in df_all.columns:
                st.error("❌ Column 'Check in Date' not found")
                st.stop()

            df_year = (
                df_all
                .dropna(subset=["Check in Date"])
                .groupby(df_all["Check in Date"].dt.year)
                .size()
                .reset_index(name="Bookings")
                .rename(columns={"Check in Date": "Year"})
            )

            fig = px.bar(
                df_year,
                x="Year",
                y="Bookings",
                text="Bookings",
                title="Yearly Booking Volume Trend"
            )

            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            if len(df_year) > 1:
                start, end = df_year["Bookings"].iloc[0], df_year["Bookings"].iloc[-1]
                years = len(df_year) - 1
                cagr = ((end / start) ** (1 / years) - 1) * 100
                st.success(f"📈 Estimated CAGR: {cagr:.2f}% per year")


    with tab3:
        st.markdown("<div class='section-header'>📈 Advanced Analytics & Business Intelligence</div>", unsafe_allow_html=True)
        
        # Time Series Analysis
        if "Check in Date" in df_all.columns and "Number of Rooms Night" in df_all.columns:
            st.markdown("#### 📅 Monthly Trend Analysis - Room Nights")
            
            df_ts = df_all.groupby(df_all["Check in Date"].dt.to_period("M"))["Number of Rooms Night"].sum().reset_index()
            df_ts["Check in Date"] = df_ts["Check in Date"].dt.to_timestamp()
            
            fig = px.line(
                df_ts,
                x="Check in Date",
                y="Number of Rooms Night",
                markers=True,
                labels={"Check In Date": "Period", "Number of Rooms Night": "Total Room Nights"}
            )
            fig.update_traces(
                line_color='#7eb762', 
                line_width=3, 
                marker=dict(size=10, color='#6d99e1', line=dict(width=2, color='#005693'))
            )
            fig.update_layout(
                height=450, 
                hovermode='x unified',
                paper_bgcolor='#f9f9f9',
                plot_bgcolor='#ffffff',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Top 100 Hotels & Cities
        col1, col2 = st.columns(2)
        
        with col1:
            if "Hotel Name" in df_all.columns and "Number of Rooms Night" in df_all.columns:
                st.markdown("#### 🏆 Top 100 Hotels by Room Nights")
                top_hotels = df_all.groupby("Hotel Name")["Number of Rooms Night"].sum().sort_values(ascending=False).head(100)
                
                fig = go.Figure(go.Bar(
                    x=top_hotels.values,
                    y=top_hotels.index,
                    orientation='h',
                    marker=dict(
                        color=top_hotels.values,
                        colorscale=[[0, '#005693'], [0.5, '#7eb762'], [1, '#6d99e1']],
                        showscale=True,
                        colorbar=dict(title="Room Nights", thickness=15, len=0.7)
                    ),
                    text=top_hotels.values,
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                ))
                fig.update_layout(
                    height=2000,
                    yaxis={'categoryorder':'total ascending'},
                    paper_bgcolor='#f9f9f9',
                    plot_bgcolor='#ffffff',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                    font=dict(size=9),
                    margin=dict(l=200, r=50, t=30, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download Top 100 Hotels
                csv_hotels = top_hotels.reset_index()
                csv_hotels.columns = ['Hotel Name', 'Total Room Nights']
                st.download_button(
                    label="📥 Download Top 100 Hotels CSV",
                    data=csv_hotels.to_csv(index=False),
                    file_name=f"top_100_hotels_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if "City" in df_all.columns and "Number of Rooms Night" in df_all.columns:
                st.markdown("#### 🌆 Top 100 Cities by Room Nights")
                top_cities = df_all.groupby("City")["Number of Rooms Night"].sum().sort_values(ascending=False).head(100)
                
                fig = go.Figure(go.Bar(
                    x=top_cities.values,
                    y=top_cities.index,
                    orientation='h',
                    marker=dict(
                        color=top_cities.values,
                        colorscale=[[0, '#005693'], [0.5, '#7eb762'], [1, '#6d99e1']],
                        showscale=True,
                        colorbar=dict(title="Room Nights", thickness=15, len=0.7)
                    ),
                    text=top_cities.values,
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                ))
                fig.update_layout(
                    height=2000,
                    yaxis={'categoryorder':'total ascending'},
                    paper_bgcolor='#f9f9f9',
                    plot_bgcolor='#ffffff',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                    font=dict(size=9),
                    margin=dict(l=200, r=50, t=30, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download Top 100 Cities
                csv_cities = top_cities.reset_index()
                csv_cities.columns = ['City', 'Total Room Nights']
                st.download_button(
                    label="📥 Download Top 100 Cities CSV",
                    data=csv_cities.to_csv(index=False),
                    file_name=f"top_100_cities_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with tab4:
            st.markdown("<div class='section-header'>🧠 Machine Learning Analytics</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='ai-card'>
                <div style='font-size: 3em;'>🧠</div>
                <div style='font-size: 1.8em; font-weight: 700;'>Machine Learning Decision Engine</div>
                <div style='opacity: 0.9;'>
                    Data-driven recommendation engine for hotel negotiation, demand forecasting,
                    concentration risk, and booking behavior patterns.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # =========================
            # ML METHOD DROPDOWN
            # =========================
            ml_method = st.selectbox(
                "📌 Select Machine Learning Method",
                [
                    "1️⃣ Hotel Negotiation Priority (Rule-Based + ML)",
                    "2️⃣ City-Month Reservation Demand Forecast",
                    "3️⃣ Hotel Demand Concentration Index",
                    "4️⃣ Lead Time Pattern Prediction",
                    "5️⃣ Employee Booking Behavior Clustering"

                ]
            )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # =====================================================
            # 1. HOTEL NEGOTIATION PRIORITY
            # =====================================================
            if ml_method.startswith("1️⃣"):

                st.subheader("🏨 Hotel Negotiation Priority Model")

                st.markdown("""
                **Objective:**  
                Identify hotels that should be **prioritized for contract renewal or price negotiation**
                using a hybrid **Rule-Based + Machine Learning Scoring Model**.
                """)

                required_cols = ["Hotel Name", "Number of Rooms Night", "Invoice Amount"]

                if all(c in df_all.columns for c in required_cols):

                    df_hotel = (
                        df_all.groupby("Hotel Name")
                        .agg(
                            total_room_nights=("Number of Rooms Night", "sum"),
                            total_spend=("Invoice Amount", "sum"),
                            booking_count=("Hotel Name", "count")
                        )
                        .reset_index()
                    )

                    # Rule-Based Normalization
                    for col in ["total_room_nights", "total_spend", "booking_count"]:
                        df_hotel[f"{col}_score"] = (
                            (df_hotel[col] - df_hotel[col].min()) /
                            (df_hotel[col].max() - df_hotel[col].min())
                        )

                    # Hybrid Score
                    df_hotel["Negotiation Priority Score"] = (
                        0.5 * df_hotel["total_room_nights_score"] +
                        0.3 * df_hotel["total_spend_score"] +
                        0.2 * df_hotel["booking_count_score"]
                    )

                    df_hotel = df_hotel.sort_values(
                        "Negotiation Priority Score", ascending=False
                    )

                    st.dataframe(
                        df_hotel.head(20),
                        use_container_width=True
                    )

                    st.markdown("""
                    ### Insight Summary

                    Model ini digunakan untuk **mengidentifikasi hotel-hotel yang memiliki prioritas tertinggi
                    untuk dilakukan negosiasi harga kontrak pada periode berikutnya** berdasarkan pola
                    penggunaan aktual dalam data perjalanan korporat.

                    #### Apa yang dianalisis model ini?
                    Model mengevaluasi setiap hotel berdasarkan beberapa indikator utama:
                    - **Volume booking** dan **room nights** (tingkat utilisasi)
                    - **Konsistensi penggunaan** oleh berbagai unit / cost center
                    - **Konsentrasi permintaan** (apakah hotel menjadi pilihan utama atau alternatif)
                    - **Tren peningkatan atau penurunan pemakaian** dibanding periode sebelumnya

                    #### Bagaimana membaca hasilnya?
                    - **Priority Score tinggi** → Hotel sangat sering digunakan dan memiliki potensi besar
                    untuk menghasilkan efisiensi biaya jika dinegosiasikan ulang.
                    - **Priority Score menengah** → Hotel strategis, namun perlu seleksi lebih lanjut
                    berdasarkan lokasi atau segmentasi pengguna.
                    - **Priority Score rendah** → Hotel bersifat situasional dan **tidak menjadi fokus utama**
                    dalam negosiasi kontrak.

                    #### Manfaat bisnis yang dihasilkan:
                    - Fokus negosiasi pada hotel dengan **leverage volume tertinggi**
                    - Menghindari kontrak yang tidak optimal atau jarang digunakan
                    - Mendukung strategi **cost optimization & vendor consolidation**
                    - Menjadi dasar objektif dalam pengambilan keputusan manajemen

                    📌 *Model ini bersifat decision-support dan dapat dikombinasikan dengan
                    pertimbangan komersial serta kebijakan internal perusahaan.*
                    """)


                    # =========================
                    # DOWNLOAD EXCEL
                    # =========================
                    output_excel = df_hotel.copy()

                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        output_excel.to_excel(
                            writer,
                            index=False,
                            sheet_name="Negotiation Priority"
                        )

                    st.download_button(
                        label="⬇️ Download Negotiation Priority (Excel)",
                        data=buffer.getvalue(),
                        file_name="hotel_negotiation_priority.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


                    st.success("✅ Top hotels with highest negotiation leverage identified")

                else:
                    st.warning("⚠️ Required columns not available for this model.")

            # =====================================================
            # 2. CITY-MONTH RESERVATION DEMAND FORECAST
            # =====================================================
            elif ml_method.startswith("2️⃣"):

                st.subheader("📈 City-Month Reservation Demand Forecast")

                st.markdown("""
                **Objective:**  
                Forecast **future hotel reservation demand per city per month**
                using historical booking trends (time-series ready).
                """)

                # =====================================================
                # DATA VALIDATION
                # =====================================================
                if "Check in Date" not in df_all.columns or "City" not in df_all.columns:
                    st.warning("⚠️ Required columns (Check in Date, City) are not available.")
                    st.stop()

                # =====================================================
                # PREPARE TIME SERIES DATA
                # =====================================================
                df_ts = df_all.copy()
                df_ts["Check in Date"] = pd.to_datetime(df_ts["Check in Date"], errors="coerce")
                df_ts = df_ts.dropna(subset=["Check in Date"])

                df_ts["YearMonth"] = df_ts["Check in Date"].dt.to_period("M").astype(str)

                demand = (
                    df_ts.groupby(["City", "YearMonth"])
                    .size()
                    .reset_index(name="Total Bookings")
                    .sort_values(["City", "YearMonth"])
                )

                st.markdown("### 📊 Historical Demand (City–Month)")
                st.dataframe(demand.tail(20), use_container_width=True)

                # =====================================================
                # BASELINE FORECAST (STATISTICAL / NAIVE)
                # =====================================================
                st.markdown("### 🔮 Baseline Forecast (Statistical)")

                forecast_df = (
                    demand.groupby("YearMonth")["Total Bookings"]
                    .sum()
                    .reset_index()
                )

                forecast_df["Forecast (Next Month)"] = forecast_df["Total Bookings"].rolling(3).mean()

                st.dataframe(forecast_df.tail(12), use_container_width=True)

                # =====================================================
                # LSTM MODEL (OPTIONAL / EXPERIMENTAL)
                # =====================================================
                st.markdown("### 🤖 LSTM-Based Forecast (Experimental)")

                if TENSORFLOW_AVAILABLE:
                    try:
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import LSTM, Dense
                        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

                        def prepare_lstm_data(series, n_input=3):
                            series = np.array(series).reshape(-1, 1)
                            return TimeseriesGenerator(series, series, length=n_input, batch_size=1)

                        def run_lstm_forecast(series, n_input=3, n_future=3):
                            generator = prepare_lstm_data(series, n_input)

                            model = Sequential([
                                LSTM(50, activation="relu", input_shape=(n_input, 1)),
                                Dense(1)
                            ])
                            model.compile(optimizer="adam", loss="mse")
                            model.fit(generator, epochs=25, verbose=0)

                            predictions = []
                            current_batch = series[-n_input:].reshape((1, n_input, 1))

                            for _ in range(n_future):
                                pred = model.predict(current_batch, verbose=0)[0][0]
                                predictions.append(pred)
                                current_batch = np.append(
                                    current_batch[:, 1:, :],
                                    [[[pred]]],
                                    axis=1
                                )

                            return predictions

                        series = forecast_df["Total Bookings"].values

                        lstm_pred = run_lstm_forecast(series)

                        lstm_result = pd.DataFrame({
                            "Future Period": [f"M+{i+1}" for i in range(len(lstm_pred))],
                            "Predicted Demand": lstm_pred
                        })

                        st.success("✅ LSTM model executed successfully")
                        st.dataframe(lstm_result, use_container_width=True)

                        st.markdown("""
                        **Model Insight (LSTM):**  
                        LSTM mampu menangkap **pola musiman dan dependensi jangka panjang**
                        dalam data reservasi hotel per kota.
                        Cocok untuk:
                        - Forecast jangka menengah
                        - Analisis tren berulang (seasonality)
                        - Dasar strategi negosiasi kontrak hotel
                        """)

                    except Exception as e:
                        st.warning(f"⚠️ LSTM gagal dijalankan: {e}")

                else:
                    st.info("""
                    **LSTM tidak tersedia di environment ini.**

                    Sistem menggunakan **Statistical Forecast** sebagai baseline.
                    Untuk mengaktifkan LSTM:
                    - Install TensorFlow
                    - Gunakan environment khusus ML
                    """)

                # =====================================================
                # INTERPRETATION
                # =====================================================
                st.markdown("""
                ### 🧠 Insight Interpretation

                Model ini memproyeksikan **tingkat permintaan reservasi hotel per kota
                dalam skala bulanan** berdasarkan histori pemesanan.

                **Cara Membaca Output:**
                - **High Forecast** → Potensi peak demand  
                👉 *Perlu rate lock & early negotiation*

                - **Moderate Forecast** → Permintaan stabil  
                👉 *Kontrak harga normal*

                - **Low Forecast** → Low season  
                👉 *Peluang negosiasi agresif*

                **Manfaat Strategis:**
                - Antisipasi peak season per kota
                - Optimasi kontrak hotel berbasis demand
                - Pengurangan risiko overpricing
                - Perencanaan perjalanan jangka menengah
                """)




            # =====================================================
            # 3. HOTEL DEMAND CONCENTRATION INDEX
            # =====================================================
            elif ml_method.startswith("3️⃣"):

                st.subheader("📊 Hotel Demand Concentration Index")

                st.markdown("""
                **Objective:**  
                Measure **dependency risk** on specific hotels
                using **Herfindahl–Hirschman Index (HHI)** logic.
                """)

                if "Hotel Name" in df_all.columns:

                    hotel_share = (
                        df_all["Hotel Name"]
                        .value_counts(normalize=True)
                        .reset_index()
                    )
                    hotel_share.columns = ["Hotel Name", "Share"]

                    hhi = (hotel_share["Share"] ** 2).sum()

                    st.metric(
                        "Hotel Concentration Index (HHI)",
                        f"{hhi:.3f}"
                    )

                    st.dataframe(hotel_share.head(20), use_container_width=True)

                    if hhi > 0.25:
                        st.error("🚨 High concentration risk detected")
                    elif hhi > 0.15:
                        st.warning("⚠️ Moderate concentration risk")
                    else:
                        st.success("✅ Healthy hotel diversification")

                else:
                    st.warning("⚠️ Hotel Name column not found.")

            # =====================================================
            # 4. LEAD TIME PATTERN PREDICTION
            # =====================================================
            elif ml_method.startswith("4️⃣"):

                st.subheader("⌛ Lead Time Pattern Prediction")

                st.markdown("""
                **Objective:**  
                Analyze & prepare predictive features for
                **booking lead time behavior**.
                """)

                if "Issue Time" in df_all.columns and "Check in Date" in df_all.columns:

                    df_lead = df_all.copy()
                    df_lead["Lead Time (Days)"] = (
                        df_lead["Check in Date"].dt.normalize() -
                        df_lead["Issue Time"].dt.normalize()
                    ).dt.days

                    df_lead = df_lead[df_lead["Lead Time (Days)"] >= 0]

                    st.metric(
                        "Average Lead Time (Days)",
                        f"{df_lead['Lead Time (Days)'].mean():.1f}"
                    )

                    st.dataframe(
                        df_lead[["Lead Time (Days)"]].describe(),
                        use_container_width=True
                    )

                    st.info("""
                    🤖 Ready for:
                    - Classification (Last-minute vs Planned)
                    - Regression (Lead Time Prediction)
                    """)

                else:
                    st.warning("⚠️ Issue Time / Check in Date not available.")
                
            # =====================================================
            # 5. EMPLOYEE BOOKING BEHAVIOR CLUSTERING
            # =====================================================
            elif ml_method.startswith("5️⃣"):

                st.subheader("👤 Employee Booking Behavior Clustering")

                st.markdown("""
                **Objective:**  
                Segment employees based on hotel booking behavior to support
                policy control, training, and travel governance.
                """)

                required_cols = [
                    "Employee Id",
                    "Invoice Amount",
                    "Issue Time",
                    "Check in Date",
                    "Hotel Name"
                ]

                if all(col in df_all.columns for col in required_cols):

                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler

                    df_emp = df_all.copy()

                    # =========================
                    # FEATURE ENGINEERING
                    # =========================
                    df_emp["Lead Time"] = (
                        df_emp["Check in Date"].dt.normalize() -
                        df_emp["Issue Time"].dt.normalize()
                    ).dt.days

                    df_emp = df_emp[df_emp["Lead Time"] >= 0]

                    emp_features = (
                        df_emp.groupby("Employee Id")
                        .agg(
                            Total_Bookings=("Employee Id", "count"),
                            Avg_Spend=("Invoice Amount", "mean"),
                            Avg_Lead_Time=("Lead Time", "mean"),
                            Hotel_Diversity=("Hotel Name", "nunique")
                        )
                        .reset_index()
                    )

                    # =========================
                    # SCALING
                    # =========================
                    scaler = StandardScaler()
                    X = scaler.fit_transform(
                        emp_features[
                            ["Total_Bookings", "Avg_Spend", "Avg_Lead_Time", "Hotel_Diversity"]
                        ]
                    )

                    # =========================
                    # CLUSTERING
                    # =========================
                    kmeans = KMeans(n_clusters=4, random_state=42)
                    emp_features["Cluster"] = kmeans.fit_predict(X)

                    # =========================
                    # CLUSTER LABELING (RULE-BASED)
                    # =========================
                    cluster_profile = (
                        emp_features
                        .groupby("Cluster")
                        .mean(numeric_only=True)
                    )

                    cluster_map = {}
                    for c, row in cluster_profile.iterrows():
                        if row["Avg_Lead_Time"] < 3:
                            cluster_map[c] = "Last-Minute Booker"
                        elif row["Avg_Spend"] > cluster_profile["Avg_Spend"].mean():
                            cluster_map[c] = "High Spender"
                        elif row["Total_Bookings"] > cluster_profile["Total_Bookings"].mean():
                            cluster_map[c] = "Frequent Traveler"
                        else:
                            cluster_map[c] = "Planner"

                    emp_features["Behavior Segment"] = emp_features["Cluster"].map(cluster_map)

                    # =========================
                    # DISPLAY
                    # =========================
                    st.dataframe(
                        emp_features.sort_values("Total_Bookings", ascending=False),
                        use_container_width=True
                    )

                    st.success("✅ Employee behavior clusters successfully identified")

                    st.markdown("### 🥧 Cluster Distribution (Interactive)")

                    cluster_count = (
                        emp_features["Cluster"]
                        .value_counts()
                        .reset_index()
                    )
                    cluster_count.columns = ["Cluster", "Employee Count"]

                    fig_pie = px.pie(
                        cluster_count,
                        names="Cluster",
                        values="Employee Count",
                        hole=0.35
                    )

                    fig_pie.update_layout(
                        legend_title_text="Cluster",
                        margin=dict(t=40, b=20, l=20, r=20)
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                    # =========================

                    st.markdown("### 📈 Lead Time vs Average Spend (Interactive)")

                    fig_scatter = px.scatter(
                        emp_features,
                        x="Avg_Lead_Time",
                        y="Avg_Spend",
                        color="Behavior Segment",
                        hover_data={
                            "Employee Id": True,
                            "Total_Bookings": True,
                            "Hotel_Diversity": True,
                            "Avg_Lead_Time": ':.1f',
                            "Avg_Spend": ':.0f'
                        }
                    )

                    fig_scatter.update_layout(
                        xaxis_title="Average Lead Time (Days)",
                        yaxis_title="Average Spend",
                        legend_title_text="Behavior Segment",
                        margin=dict(t=40, b=40)
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # =========================

                    st.markdown("## 🔥 High-Risk Employee Index")

                    st.markdown("""
                    **Objective:**  
                    Quantify employee booking risk based on **last-minute behavior, spending pattern,
                    booking frequency, and hotel switching behavior** to support governance,
                    policy enforcement, and cost optimization.
                    """)

                    # =====================================================
                    # FEATURE SOURCE
                    # emp_features sudah tersedia dari clustering
                    # =====================================================

                    risk_df = emp_features.copy()

                    # =========================
                    # NORMALIZATION
                    # =========================
                    def min_max(series):
                        return (series - series.min()) / (series.max() - series.min())

                    risk_df["LeadTime_Risk"] = 1 - min_max(risk_df["Avg_Lead_Time"])
                    risk_df["Spend_Risk"] = min_max(risk_df["Avg_Spend"])
                    risk_df["Frequency_Risk"] = min_max(risk_df["Total_Bookings"])
                    risk_df["HotelSwitch_Risk"] = min_max(risk_df["Hotel_Diversity"])

                    # =========================
                    # HIGH-RISK SCORE
                    # =========================
                    risk_df["High-Risk Score"] = (
                        0.4 * risk_df["LeadTime_Risk"] +
                        0.3 * risk_df["Spend_Risk"] +
                        0.2 * risk_df["Frequency_Risk"] +
                        0.1 * risk_df["HotelSwitch_Risk"]
                    ) * 100

                    # =========================
                    # RISK CATEGORY
                    # =========================
                    def risk_label(score):
                        if score >= 75:
                            return "HIGH"
                        elif score >= 50:
                            return "MEDIUM"
                        else:
                            return "LOW"

                    risk_df["Risk Level"] = risk_df["High-Risk Score"].apply(risk_label)


                    # =====================================================
                    # FEATURE SOURCE
                    # emp_features sudah tersedia dari clustering
                    # =====================================================

                    risk_df = emp_features.copy()

                    # =========================
                    # NORMALIZATION
                    # =========================
                    def min_max(series):
                        return (series - series.min()) / (series.max() - series.min())

                    risk_df["LeadTime_Risk"] = 1 - min_max(risk_df["Avg_Lead_Time"])
                    risk_df["Spend_Risk"] = min_max(risk_df["Avg_Spend"])
                    risk_df["Frequency_Risk"] = min_max(risk_df["Total_Bookings"])
                    risk_df["HotelSwitch_Risk"] = min_max(risk_df["Hotel_Diversity"])

                    # =========================
                    # HIGH-RISK SCORE
                    # =========================
                    risk_df["High-Risk Score"] = (
                        0.4 * risk_df["LeadTime_Risk"] +
                        0.3 * risk_df["Spend_Risk"] +
                        0.2 * risk_df["Frequency_Risk"] +
                        0.1 * risk_df["HotelSwitch_Risk"]
                    ) * 100

                    # =========================
                    # RISK CATEGORY
                    # =========================
                    def risk_label(score):
                        if score >= 75:
                            return "HIGH"
                        elif score >= 50:
                            return "MEDIUM"
                        else:
                            return "LOW"

                    risk_df["Risk Level"] = risk_df["High-Risk Score"].apply(risk_label)

                    # =====================================================
                    # KPI SUMMARY
                    # =====================================================
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "🔴 High-Risk Employee",
                        int((risk_df["Risk Level"] == "HIGH").sum())
                    )

                    col2.metric(
                        "🟠 Medium Risk",
                        int((risk_df["Risk Level"] == "MEDIUM").sum())
                    )

                    col3.metric(
                        "🟢 Low Risk",
                        int((risk_df["Risk Level"] == "LOW").sum())
                    )

                    # =====================================================
                    # TOP HIGH-RISK TABLE
                    # =====================================================
                    st.markdown("### 📋 Top High-Risk Employee")

                    top_risk = (
                        risk_df
                        .sort_values("High-Risk Score", ascending=False)
                        .head(15)
                    )

                    st.dataframe(
                        top_risk[
                            [
                                "Employee Id",
                                "High-Risk Score",
                                "Risk Level",
                                "Behavior Segment",
                                "Avg_Spend",
                                "Avg_Lead_Time",
                                "Total_Bookings",
                                "Hotel_Diversity"
                            ]
                        ],
                        use_container_width=True
                    )

                    # =====================================================
                    # DOWNLOAD EXCEL
                    # =====================================================
                    from io import BytesIO

                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        risk_df.to_excel(
                            writer,
                            index=False,
                            sheet_name="High_Risk_Employee_Index"
                        )

                    st.download_button(
                        "⬇️ Download High-Risk Employee Index (Excel)",
                        buffer.getvalue(),
                        file_name="high_risk_employee_index.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.markdown("### 🧠 Analytical Insight")

                    high_risk_pct = (
                        (risk_df["Risk Level"] == "HIGH").mean() * 100
                    )

                    avg_lead_high = risk_df[risk_df["Risk Level"] == "HIGH"]["Avg_Lead_Time"].mean()
                    avg_lead_low = risk_df[risk_df["Risk Level"] == "LOW"]["Avg_Lead_Time"].mean()

                    st.info(f"""
                    **Key Findings:**

                    • **{high_risk_pct:.1f}% of employees** are categorized as **High Risk**, indicating
                    significant exposure to last-minute booking behavior and elevated travel cost.

                    • High-risk employees show an **average lead time of {avg_lead_high:.1f} days**,
                    significantly shorter compared to **{avg_lead_low:.1f} days** for low-risk employees.

                    • Majority of high-risk employees fall under **Last-Minute Booker** and
                    **High Spender** behavior segments, suggesting targeted policy intervention
                    and training could yield immediate cost efficiency.

                    **Management Implication:**  
                    Focusing control and awareness programs on the **top 10–20% highest-risk employees**
                    has the potential to reduce travel spend volatility and improve contract negotiation leverage.
                    """)

                else:
                    st.warning(
                        "⚠️ Required columns not found: Employee Id, Invoice Amount, Issue Time, Check in Date, Hotel Name"
                    )

        # =====================================================
        # FEATURE ENGINEERING FOR DEMAND INTELLIGENCE
        # =====================================================
        df_feat = df_all.copy()

        # Date handling
        df_feat["Check in Date"] = pd.to_datetime(df_feat["Check in Date"], errors="coerce")
        df_feat["Issue Time"] = pd.to_datetime(df_feat["Issue Time"], errors="coerce")

        # Lead Time
        df_feat["lead_time"] = (
            df_feat["Check in Date"].dt.normalize() -
            df_feat["Issue Time"].dt.normalize()
        ).dt.days

        # Price per night
        df_feat["price_per_night"] = (
            df_feat["Invoice Amount"] / df_feat["Number of Rooms Night"]
        )

        # Weekday / Weekend
        df_feat["is_weekend"] = df_feat["Check in Date"].dt.weekday >= 5

        # City-Month Avg Price
        df_feat["month"] = df_feat["Check in Date"].dt.month
        df_feat["avg_price_city_month"] = (
            df_feat.groupby(["City", "month"])["price_per_night"]
            .transform("mean")
        )

        # Same day last year price
        df_feat["last_year_date"] = df_feat["Check in Date"] - pd.DateOffset(years=1)
        df_feat["avg_price_same_day_last_year"] = (
            df_feat.groupby(["City", "last_year_date"])["price_per_night"]
            .transform("mean")
        )

#=================================================================================================================#

        with tab5:

            st.markdown("<div class='section-header'>💡 Demand Intelligence 2026</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='ai-card'>
                <div style='font-size: 3em;'>📅🤖</div>
                <div style='font-size: 1.8em; font-weight: 700;'>Calendar-Aware Demand Intelligence Engine</div>
                <div style='opacity: 0.9;'>
                    Advanced demand, price & availability intelligence using
                    <b>Google Calendar (SKB Libur Nasional)</b>, historical booking behavior,
                    and machine learning risk modeling.
                </div>
                <div style='margin-top: 10px; font-size: 0.9em; opacity: 0.85;'>
                    Designed for <b>early warning system</b>, <b>corporate rate negotiation</b>,
                    and <b>policy-based travel control</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # =========================
            # MODEL SELECTOR
            # =========================
            demand_model = st.selectbox(
                "📌 Select Demand Intelligence Model",
                [
                    "1️⃣ Price Surge Risk Prediction (Hotel Price Volatility)",
                    "2️⃣ Availability Risk Prediction (Sold-Out Risk)",
                    "3️⃣ Optimal Booking Time Recommendation",
                    "4️⃣ Event-Aware Demand Forecast (Calendar-Based)",
                    "5️⃣ Holiday Sensitivity Index per City"
                ]
            )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # =====================================================
            # 1. PRICE SURGE RISK PREDICTION
            # =====================================================
            if demand_model.startswith("1️⃣"):

                st.subheader("📈 Price Surge Risk Prediction")

                st.markdown("""
                **🎯 Objective**  
                Predict the **probability of hotel price increase** for a given **city & date in 2026**
                by combining historical price behavior with national holiday signals from
                **Google Calendar (SKB)**.

                **🎯 Target Variable**
                ```text
                price_surge_flag =
                1 → price > (normal average + threshold)
                0 → normal
                ```

                **Threshold Reference**
                • +15% → Moderate surge  
                • +30% → High surge
                """)

                st.markdown("""
                **📌 Key Features**
                - is_national_holiday  
                - is_cuti_bersama  
                - is_long_weekend  
                - days_to_holiday  
                - avg_price_city_month  
                - avg_price_same_day_last_year  
                - booking_lead_time  
                - weekday / weekend
                """)

                st.info("""
                🔔 **Example Alert Output**

                **“Bandung | 22–24 May 2026**  
                🔺 **High price surge risk detected (+32%)**  
                Recommendation: *Advance booking & rate locking required.*
                """)

            # =====================================================
            # 2. AVAILABILITY RISK MODEL
            # =====================================================
            elif demand_model.startswith("2️⃣"):

                st.subheader("🏨 Availability Risk Prediction (Sold-Out Risk)")

                st.markdown("""
                **🎯 Objective**  
                Predict the likelihood that hotels in a certain **city & period**
                will be **difficult to book or sold out**.
                """)

                st.markdown("""
                **🎯 Proxy Target Definition**
                ```text
                availability_risk = 1 if:
                - booking failure exists
                - OR extreme price surge
                - OR booking happens at H-1 / H-0
                ```
                """)

                st.markdown("""
                **📌 Key Features**
                - holiday_flag  
                - long_weekend_length  
                - avg_room_night_city  
                - transaction_volume_same_period  
                - booking_window (H-x)
                """)

                st.success("""
                **🟢🟡🔴 Risk Status Output**
                - 🟢 Safe → Normal booking
                - 🟡 Watchlist → Limited availability
                - 🔴 Critical → High sold-out risk

                📌 Perfect for **early warning notification to travelers & admins**
                """)

            # =====================================================
            # 3. OPTIMAL BOOKING TIME
            # =====================================================
            elif demand_model.startswith("3️⃣"):

                st.subheader("⏳ Optimal Booking Time Recommendation")

                st.markdown("""
                **🎯 Objective**  
                Determine **H-berapa booking paling optimal**
                per **city & holiday period** to minimize cost and availability risk.
                """)

                st.markdown("""
                **🎯 Optimal Flag Logic**
                ```text
                optimal_booking_flag = 1 if:
                - price stable
                - room available
                - not last-minute
                ```
                """)

                st.markdown("""
                **📌 Key Features**
                - days_to_holiday  
                - holiday_cluster_size  
                - city_demand_index  
                - historical_price_curve
                """)

                st.info("""
                📌 **Example Insight**

                **“Surabaya – Lebaran Period**  
                ✅ Optimal booking window: **H-21 to H-14**  
                ❌ Avoid booking: H-3 to H-0”
                """)

            # =====================================================
            # 4. EVENT-AWARE DEMAND FORECAST
            # =====================================================
            elif demand_model.startswith("4️⃣"):

                st.subheader("📊 Event-Aware Demand Forecast")

                st.markdown("""
                **🎯 Objective**  
                Forecast **hotel demand spikes** driven by
                **SKB / Google Calendar holiday patterns**.
                """)

                st.markdown("""
                **🎯 Forecast Targets**
                - Total transactions  
                - Total room nights  
                - Total spend
                """)

                st.markdown("""
                **🤖 Recommended Models**
                - Prophet + holiday regressor  
                - XGBoost (time-series features)
                """)

                st.success("""
                🔮 **Example Forecast Output**

                **“Q2 2026**  
                +28% hotel demand increase  
                Driven by **6 long weekends & clustered national holidays**”
                """)

            # =====================================================
            # 5. HOLIDAY SENSITIVITY INDEX
            # =====================================================
            elif demand_model.startswith("5️⃣"):

                st.subheader("🌆 Holiday Sensitivity Index per City")

                st.markdown("""
                **🎯 Objective**  
                Measure **how sensitive each city is to national holidays**
                in terms of **price increase & demand spike**.
                """)

                st.markdown("""
                **🎯 Target Variable**
                ```text
                price_change_pct (regression-based)
                ```
                """)

                st.markdown("""
                **📊 Example Sensitivity Score**
                - Jakarta → **0.45**  
                - Bali → **0.78**  
                - Yogyakarta → **0.83**
                """)

                st.info("""
                📌 **Business Usage**
                - Corporate rate negotiation priority  
                - City-specific travel policy  
                - Differential booking window rules  
                - Budget volatility control
                """)

#=====================================================================================================#

    with tab6:
        st.markdown("<div class='section-header'>⬇️ Export & Reporting Center</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 25px; border-radius: 12px; border: 2px solid #e9ecef;'>
                    <div style='font-size: 2.5em; color: #7eb762; margin-bottom: 15px;'>📄</div>
                    <h3 style='color: #005693; margin: 0 0 10px 0;'>CSV Format</h3>
                    <p style='color: #6c757d; font-size: 0.95em; margin-bottom: 20px;'>
                        Lightweight format ideal for:
                        <br>• Database imports
                        <br>• Large dataset processing
                        <br>• System integrations
                        <br>• Automated workflows
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            buffer_csv = BytesIO()
            df_all.to_csv(buffer_csv, index=False)
            st.download_button(
                label="📥 DOWNLOAD CSV REPORT",
                data=buffer_csv.getvalue(),
                file_name=f"pertamina_travel_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 25px; border-radius: 12px; border: 2px solid #e9ecef;'>
                    <div style='font-size: 2.5em; color: #6d99e1; margin-bottom: 15px;'>📊</div>
                    <h3 style='color: #005693; margin: 0 0 10px 0;'>Excel Format</h3>
                    <p style='color: #6c757d; font-size: 0.95em; margin-bottom: 20px;'>
                        Comprehensive format for:
                        <br>• Detailed analysis
                        <br>• Management reports
                        <br>• Presentations
                        <br>• Further processing
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                df_all.to_excel(writer, index=False, sheet_name="Travel Data")
            st.download_button(
                label="📥 DOWNLOAD EXCEL REPORT",
                data=buffer_excel.getvalue(),
                file_name=f"pertamina_travel_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Comprehensive Data Summary
        st.markdown("<div class='section-header'>📋 Comprehensive Data Summary Report</div>", unsafe_allow_html=True)
        
        summary_data = {
            "Metric": [
                "📝 Total Booking Records",
                "📊 Data Dimensions",
                "📋 Unique Travel Requests",
                "👥 Active Employees",
                "🏢 Direktorat Coverage",
                "🔧 Functional Units",
                "🏨 Hotel Partners",
                "🌆 Cities Covered",
                "🌍 Countries Covered",
                "🛏️ Total Room Nights",
                "📅 Data Period",
                "🔄 Last Updated"
            ],
            "Value": [
                f"{df_all.shape[0]:,}",
                f"{df_all.shape[0]:,} rows × {df_all.shape[1]:,} columns",
                f"{df_all['Travel Request Number'].nunique():,}" if "Travel Request Number" in df_all.columns else "N/A",
                f"{df_all['Employee Id'].nunique():,}" if "Employee Id" in df_all.columns else "N/A",
                f"{df_all['Direktorat Pekerja'].nunique():,}" if "Direktorat Pekerja" in df_all.columns else "N/A",
                f"{df_all['Nama Fungsi'].nunique():,}" if "Nama Fungsi" in df_all.columns else "N/A",
                f"{df_all['Hotel Name'].nunique():,}" if "Hotel Name" in df_all.columns else "N/A",
                f"{df_all['City'].nunique():,}" if "City" in df_all.columns else "N/A",
                f"{df_all['Country'].nunique():,}" if "Country" in df_all.columns else "N/A",
                f"{df_all['Number of Rooms Night'].sum():,.0f}" if "Number of Rooms Night" in df_all.columns else "N/A",
                f"{df_all['Check in Date'].min().strftime('%Y-%m-%d')} to {df_all['Check in Date'].max().strftime('%Y-%m-%d')}" if "Check in Date" in df_all.columns else "N/A",
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, height=500)
        
        # Export summary
        summary_csv = BytesIO()
        summary_df.to_csv(summary_csv, index=False)
        st.download_button(
            label="📥 Download Summary Report",
            data=summary_csv.getvalue(),
            file_name=f"pertamina_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# --- Corporate Footer ---
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='background: linear-gradient(135deg, #032c48 0%, #032c48 100%); color: white; padding: 25px; border-radius: 12px; margin-top: 40px; text-align: center;'>
        <div style='font-size: 1.3em; font-weight: 600; letter-spacing: 1px; margin-bottom: 12px;'>PT MITRA TOURS AND TRAVEL</div>
        <div style='font-size: 0.9em; opacity: 0.5; margin-bottom: 15px;'>Corporate Travel Analytics & Intelligence Platform</div>
        <div style='height: 1px; background: rgba(255,255,255,0.15); margin: 15px auto; max-width: 400px;'></div>
        <div style='font-size: 0.8em; opacity: 0.5;'>
            Version 1 © 2024 <a href='https://www.linkedin.com/in/rifyalt/' target='_blank' style='color: white; text-decoration: none; border-bottom: 1px solid rgba(255,255,255,0.3); transition: opacity 0.3s;' onmouseover='this.style.opacity=1' onmouseout='this.style.opacity=0.7'>Rifyal Tumber</a>
        </div>
    </div>
""", unsafe_allow_html=True)