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

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

    gdown.download_folder(id=folder_id, output="data_temp", quiet=True, use_cookies=False)

    files = [f for f in os.listdir("data_temp") if f.endswith((".xlsx", ".xls"))]
    dfs = []

    for f in files:
        df = pd.read_excel(os.path.join("data_temp", f))
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

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
    "admin": {"password": hash_password("admin123"), "role": "Admin"},
    "ssc": {"password": hash_password("ssc123"), "role": "Analyst"},
    "viewer": {"password": hash_password("viewer123"), "role": "Viewer"},
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
# LOGIN FUNCTION
# ======================================
def login_page():
    """Halaman login dengan desain minimalis"""
    st.set_page_config(page_title="Login | MTRAX", layout="centered")

    st.markdown(
        """
        <style>
        * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif; }
        
        .stApp { 
            background: #fafafa;
        }
        
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
        
        .login-logo {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .login-logo img {
            width: 70px;
            opacity: 0.9;
        }
        
        .app-name {
            font-size: 2em;
            font-weight: 600;
            color: #9c5789;
            margin: 20px 0 8px 0;
        }
        
        .app-subtitle {
            color: #888888;
            font-size: 0.9em;
        }
        
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
        
        .stButton > button:hover {
            background: #8a4d78;
        }
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
        if username in USERS:
            if hash_password(password) == USERS[username]["password"]:
                st.session_state.authenticated = True
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.success("Welcome back")
                st.rerun()
            else:
                st.error("Incorrect password")
        else:
            st.error("Username not found")

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
        
 #       st.markdown("### 📊 DATA MANAGEMENT")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Drive options
        drive_options = {
            "2023–2025 (All Data)": "1vygKdg7enC5Kah7WbzVLsNI--S7Tyhvz",
            "2023": "1xDFRdGLDiiScIwW9gTucRyeFCmuqNyq_",
            "2024": "16ZMZ42BLN4GPbYKAd5h75ocbxFuyc85V",
            "2025": "1chxbGHfk9hHNPZ8vlU6AqRVUKH1jEnxF"
        }
        
        selected_period = st.selectbox(
            "Select Data Period",
            options=list(drive_options.keys()),
            help="Choose which year's data to load"
        )
        
        if st.button("Drive Data", use_container_width=True, type="primary"):
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

#        st.markdown("**OR**")

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

#        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

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

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Dashboard", "Explorer", "Analytics", "ML Models", "Forecast", "Export"
        ])

        # ======================================
        # TAB 1: DASHBOARD
        # ======================================
        with tab1:
            st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
            
            # Primary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                total_rows = len(df_all)
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>Bookings</div>
                        <div class='metric-value'>{total_rows:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                if "Travel Request Number" in df_all.columns:
                    unique_tr = df_all["Travel Request Number"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Requests</div>
                            <div class='metric-value'>{unique_tr:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col3:
                if "Employee Id" in df_all.columns:
                    unique_travelers = df_all["Employee Id"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Travelers</div>
                            <div class='metric-value'>{unique_travelers:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col4:
                if "Hotel Name" in df_all.columns:
                    unique_hotels = df_all["Hotel Name"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Hotels</div>
                            <div class='metric-value'>{unique_hotels:,}</div>
                        </div>
                    """, unsafe_allow_html=True)

            with col5:
                if "Number of Rooms Night" in df_all.columns:
                    total_nights = df_all["Number of Rooms Night"].sum()
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
                if "Company Code" in df_all.columns:
                    unique_company = df_all["Company Code"].nunique()
                    st.metric("Companies", f"{unique_company:,}")

            with col2:
                if "Cost Center Pekerja" in df_all.columns:
                    unique_cc = df_all["Cost Center Pekerja"].nunique()
                    st.metric("Cost Centers", f"{unique_cc:,}")

            with col3:
                if "City" in df_all.columns:
                    unique_cities = df_all["City"].nunique()
                    st.metric("Cities", f"{unique_cities:,}")

            with col4:
                if "Country" in df_all.columns:
                    unique_countries = df_all["Country"].nunique()
                    st.metric("Countries", f"{unique_countries:,}")

            # Travel Request Analysis
            if "Travel Request Number" in df_all.columns:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Travel Request Analysis</div>", unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    tr_bookings = df_all.groupby("Travel Request Number").size()
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
                    if "Number of Rooms Night" in df_all.columns:
                        tr_nights = df_all.groupby("Travel Request Number")["Number of Rooms Night"].sum()
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
                    if "Invoice Amount" in df_all.columns and "Number of Rooms Night" in df_all.columns:
                        valid_rows = (df_all["Invoice Amount"].notna()) & \
                                    (df_all["Number of Rooms Night"].notna()) & \
                                    (df_all["Number of Rooms Night"] > 0)
                        df_valid = df_all[valid_rows].copy()
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
                        if col in df_all.columns:
                            df_all[col] = pd.to_datetime(df_all[col], errors="coerce")

                    if "Issue Time" in df_all.columns and "Check in Date" in df_all.columns:
                        df_lead = df_all[
                            df_all["Issue Time"].notna() &
                            df_all["Check in Date"].notna()
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

            # Visualizations
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Insights</div>", unsafe_allow_html=True)

            if "Issue Time" in df_all.columns:
                df_heat = df_all[df_all["Issue Time"].notna()].copy()
                df_heat["Issue Hour"] = df_heat["Issue Time"].dt.hour
                df_heat["Issue Day"] = df_heat["Issue Time"].dt.day_name()

                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                pivot_issue = (
                    df_heat
                    .groupby(["Issue Day", "Issue Hour"])
                    .size()
                    .reset_index(name="Total")
                    .pivot(index="Issue Day", columns="Issue Hour", values="Total")
                    .reindex(day_order)
                    .fillna(0)
                )

                fig_heat = px.imshow(
                    pivot_issue,
                    text_auto=True,
                    color_continuous_scale=["#ffffff", "#ddd", "#9c5789"],
                    aspect="auto"
                )

                fig_heat.update_layout(
                    height=380,
                    title="Booking Heatmap",
                    xaxis_title="Hour",
                    yaxis_title="Day",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=40, r=20, t=50, b=40),
                    font=dict(size=11)
                )

                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if "City Destination" in df_all.columns:
                    top_cities = df_all["City Destination"].value_counts().head(10)

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
                if "Country Destination" in df_all.columns:
                    top_countries = df_all["Country Destination"].value_counts().head(10)

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

            # ======================================
            # TWO COLUMNS: CHARTS & TRENDS
            # ======================================
            col1, col2 = st.columns(2)

            # ======================================
            # COLUMN 1: TOP DIRECTORATES & TR TREND
            # ======================================
            with col1:
                # Top 10 Directorates Chart
                if "Direktorat Pekerja" in df_all.columns and "Travel Request Number" in df_all.columns:
                    top_dir = (
                        df_all.groupby("Direktorat Pekerja")["Travel Request Number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index(name="Travel Requests")
                    )

                    fig_dir = px.bar(
                        top_dir,
                        x="Travel Requests",
                        y="Direktorat Pekerja",
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

                # ================================
                # MONTHLY TREND — TRAVEL REQUEST
                # ================================
                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Monthly Trend — Travel Requests</div>", unsafe_allow_html=True)

                if "Issue Time" in df_all.columns and "Travel Request Number" in df_all.columns:

                    trend_df = prepare_monthly_trend(
                        df_all,
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

            # ======================================
            # COLUMN 2: MARKET DISTRIBUTION & ROOM NIGHTS TREND
            # ======================================
            with col2:
                # Market Distribution (Domestic vs International)
                if "Country" in df_all.columns:
                    df_all["Country"] = df_all["Country"].astype(str).str.strip().str.upper()

                    indo = df_all[df_all["Country"] == "INDONESIA"].shape[0]
                    intl = df_all[df_all["Country"] != "INDONESIA"].shape[0]

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
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=-0.15, 
                            xanchor="center", 
                            x=0.5
                        ),
                        margin=dict(l=10, r=30, t=50, b=10),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        annotations=[
                            dict(
                                text=f"<b>{total:,}</b><br>Total", 
                                x=0.5, 
                                y=0.5, 
                                font=dict(size=16), 
                                showarrow=False
                            )
                        ]
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

                # ================================
                # MONTHLY TREND — ROOM NIGHTS
                # ================================
                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Monthly Trend — Room Nights</div>", unsafe_allow_html=True)

                if "Issue Time" in df_all.columns and "Number of Rooms Night" in df_all.columns:

                    trend_df = prepare_monthly_trend(
                        df_all,
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

                else:
                    st.info("Column 'Issue Time' or 'Number of Rooms Night' not available.")

        # ======================================
        # TAB 2: EXPLORER
        # ======================================
        with tab2:
            st.markdown("<div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)

            # Search and filter
            col1, col2 = st.columns([2, 1])

            with col1:
                search_term = st.text_input("🔍 Search", placeholder="Search in any column...")

            with col2:
                if "City Destination" in df_all.columns:
                    cities = ["All"] + sorted(df_all["City Destination"].dropna().unique().tolist())
                    selected_city = st.selectbox("Filter by City", cities)
                else:
                    selected_city = "All"

            # Apply filters
            df_filtered = df_all.copy()

            if search_term:
                mask = df_filtered.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                df_filtered = df_filtered[mask]

            if selected_city != "All" and "City Destination" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["City Destination"] == selected_city]

            # Display data
            st.markdown(f"**Showing {len(df_filtered):,} of {len(df_all):,} records**")
            st.dataframe(df_filtered, use_container_width=True, height=500)

            # Dataset info
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

        # ======================================
        # TAB 3: ANALYTICS
        # ======================================
        with tab3:
            st.markdown("<div class='section-title'>Advanced Analytics</div>", unsafe_allow_html=True)

            if "Issue Time" in df_all.columns:
                df_trend = df_all.copy()
                df_trend["Issue Time"] = pd.to_datetime(df_trend["Issue Time"], errors="coerce")
                df_trend = df_trend.dropna(subset=["Issue Time"])
                df_trend["YearMonth"] = df_trend["Issue Time"].dt.to_period("M").astype(str)

                st.markdown("### Monthly Booking Trends")
                
                monthly_bookings = df_trend.groupby("YearMonth").size().reset_index(name="Bookings")

                fig_trend = px.line(
                    monthly_bookings,
                    x="YearMonth",
                    y="Bookings",
                    markers=True
                )

                fig_trend.update_traces(
                    line=dict(color="#9c5789", width=3),
                    marker=dict(size=8)
                )

                fig_trend.update_layout(
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hovermode="x unified"
                )

                st.plotly_chart(fig_trend, use_container_width=True)

                # Seasonal patterns
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("### Seasonal Patterns")

                df_trend["Month"] = df_trend["Issue Time"].dt.month_name()
                monthly_avg = df_trend.groupby("Month").size().reindex([
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ], fill_value=0)

                fig_seasonal = px.bar(
                    x=monthly_avg.index,
                    y=monthly_avg.values
                )

                fig_seasonal.update_traces(marker_color="#9c5789")

                fig_seasonal.update_layout(
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis_title="",
                    yaxis_title="Average Bookings"
                )

                st.plotly_chart(fig_seasonal, use_container_width=True)

        # ======================================
        # TAB 4: ML MODELS
        # ======================================
        with tab4:
            st.markdown("<div class='section-title'>Machine Learning Insights</div>", unsafe_allow_html=True)

            st.markdown("### Hotel Clustering Analysis")

            if "Hotel Name" in df_all.columns:
                hotel_data = df_all.groupby("Hotel Name").agg({
                    "Travel Request Number": "count",
                    "Number of Rooms Night": "sum"
                }).reset_index()

                hotel_data.columns = ["Hotel Name", "Bookings", "Total_Nights"]

                if "Hotel Price" in df_all.columns:
                    avg_price = df_all.groupby("Hotel Name")["Hotel Price"].mean()
                    hotel_data = hotel_data.merge(
                        avg_price.rename("Avg_Price"),
                        left_on="Hotel Name",
                        right_index=True,
                        how="left"
                    )
                    hotel_data["Avg_Price"].fillna(0, inplace=True)
                else:
                    hotel_data["Avg_Price"] = 0

                hotel_features = hotel_data[["Hotel Name", "Bookings", "Total_Nights", "Avg_Price"]].copy()

                if len(hotel_features) >= 5:
                    scaler = StandardScaler()
                    features_for_clustering = hotel_features[["Bookings", "Total_Nights", "Avg_Price"]]
                    features_scaled = scaler.fit_transform(features_for_clustering)

                    n_clusters = min(5, len(hotel_features))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    hotel_features["Cluster"] = kmeans.fit_predict(features_scaled)

                    fig_cluster = px.scatter(
                        hotel_features,
                        x="Bookings",
                        y="Avg_Price",
                        color="Cluster",
                        size="Total_Nights",
                        hover_data=["Hotel Name"],
                        title="Hotel Clusters",
                        color_continuous_scale=["#9c5789", "#b36d9c", "#c983af", "#df99c2", "#f5afd5"]
                    )

                    fig_cluster.update_layout(
                        height=500,
                        plot_bgcolor="white",
                        paper_bgcolor="white"
                    )

                    st.plotly_chart(fig_cluster, use_container_width=True)

                    st.markdown("### Cluster Summary")
                    cluster_summary = hotel_features.groupby("Cluster").agg({
                        "Bookings": ["mean", "count"],
                        "Total_Nights": "mean",
                        "Avg_Price": "mean"
                    }).round(2)

                    st.dataframe(cluster_summary, use_container_width=True)
                else:
                    st.warning("Not enough data for clustering analysis")
            else:
                st.error("Hotel Name column not found")

        # ======================================
        # TAB 5: FORECAST
        # ======================================
        with tab5:
            st.markdown("<div class='section-title'>Demand Forecast</div>", unsafe_allow_html=True)

            if "Issue Time" in df_all.columns:
                forecast_data = prepare_monthly_trend(
                    df_all,
                    date_col="Issue Time",
                    value_col="Travel Request Number",
                    agg="nunique"
                )

                st.markdown("### Historical Trend")

                fig_forecast = px.line(
                    forecast_data,
                    x="YearMonth",
                    y="Value",
                    markers=True
                )

                fig_forecast.update_traces(
                    line=dict(color="#9c5789", width=3),
                    marker=dict(size=8)
                )

                fig_forecast.update_layout(
                    height=400,
                    title="Monthly Travel Request Trend",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hovermode="x unified"
                )

                st.plotly_chart(fig_forecast, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_monthly = forecast_data["Value"].mean()
                    st.metric("Avg Monthly", f"{avg_monthly:.0f}")

                with col2:
                    max_monthly = forecast_data["Value"].max()
                    st.metric("Peak Month", f"{max_monthly:.0f}")

                with col3:
                    min_monthly = forecast_data["Value"].min()
                    st.metric("Lowest Month", f"{min_monthly:.0f}")

                with col4:
                    std_monthly = forecast_data["Value"].std()
                    st.metric("Std Deviation", f"{std_monthly:.0f}")

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                if TENSORFLOW_AVAILABLE:
                    st.markdown("### LSTM Forecast")
                    st.info("Advanced forecasting - Feature in development")
                else:
                    st.warning("TensorFlow not available")
            else:
                st.error("Issue Time column not found")

        # ======================================
        # TAB 6: EXPORT
        # ======================================
        with tab6:
            st.markdown("<div class='section-title'>Export Data</div>", unsafe_allow_html=True)

            st.markdown("Export your data in various formats for further analysis.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### CSV Format")
                st.markdown("Compatible with Excel, Google Sheets")

                if st.button("Download CSV", use_container_width=True, type="primary"):
                    csv = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download",
                        data=csv,
                        file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col2:
                st.markdown("#### Excel Format")
                st.markdown("Microsoft Excel workbook")

                if st.button("Download Excel", use_container_width=True, type="primary"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df_all.to_excel(writer, index=False, sheet_name="Travel Data")

                    st.download_button(
                        "⬇️ Download",
                        data=buffer.getvalue(),
                        file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            with col3:
                st.markdown("#### JSON Format")
                st.markdown("For APIs and data exchange")

                if st.button("Download JSON", use_container_width=True, type="primary"):
                    json_data = df_all.to_json(orient="records", date_format="iso")
                    st.download_button(
                        "⬇️ Download",
                        data=json_data,
                        file_name=f"mtrax_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### Export Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", f"{len(df_all):,}")

            with col2:
                st.metric("Total Columns", f"{len(df_all.columns)}")

            with col3:
                memory_usage = df_all.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")

            with col4:
                st.metric("Est. File Size", f"{memory_usage * 0.8:.2f} MB")

    else:
        st.info("👆 Please load data from Cloud/Drive or upload files to begin")

    # ======================================
    # FOOTER
    # ======================================
    st.markdown("""
    <div class='divider' style='margin-top:50px;'></div>

    <div style='text-align:center;padding:25px;color:#888888;font-size:0.85em;'>
        © 2025 Dikembangkan oleh 
        <a href="https://www.linkedin.com/in/rifyalt/" target="_blank" style='color:#9c5789;text-decoration:none;font-weight:500;'>
            Rifyal Tumber
        </a> · MTRAX Travel Analytics
    </div>
    """, unsafe_allow_html=True)


# ======================================
# RUN APP
# ======================================
if __name__ == "__main__":
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()
