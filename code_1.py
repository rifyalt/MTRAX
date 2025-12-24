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

from datetime import datetime
from io import BytesIO

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


# ======================================
# USER DATABASE
# ======================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {"password": hash_password("admin123"), "role": "Admin"},
    "ssc": {"password": hash_password("ssc123"), "role": "Analyst"},
    "viewer": {"password": hash_password("viewer123"), "role": "Viewer"},
}

# ======================================
# SESSION STATE INIT
# ======================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""


# ======================================
# LOGIN FUNCTION
# ======================================
def login_page():
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
            padding: 60px 50px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            max-width: 400px;
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
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.success("Welcome back")
                st.rerun()
            else:
                st.error("Incorrect password")
        else:
            st.error("Username not found")


# ======================================
# LOGOUT FUNCTION
# ======================================
def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.rerun()


# ======================================
# GATEKEEPER
# ======================================
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="MTRAX",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# MINIMALIST CSS
# ======================================
st.markdown("""
<style>
* { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif;
}

/* NEWS TICKER */
.news-ticker {
    background: #9c5789;
    color: white;
    padding: 8px 0;
    font-size: 0.85em;
    overflow: hidden;
    white-space: nowrap;
    position: relative;
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
    transition: color 0.2s ease;
}

.news-item:hover {
    text-decoration: underline;
}

@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}

.stApp { 
    background: #fafafa;
}

/* HEADER */
.header {
    background: white;
    padding: 30px 40px;
    border-bottom: 1px solid #e0e0e0;
    margin: -60px -60px 30px -60px;
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
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 25px 0;
}

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

.metric-change {
    font-size: 0.85em;
    color: #888888;
    margin-top: 6px;
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

.stButton > button[kind="secondary"] {
    background: white;
    color: #9c5789;
    border: 1px solid #9c5789;
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

/* FILE UPLOADER */
.stFileUploader {
    background: white;
    padding: 25px;
    border-radius: 4px;
    border: 2px dashed #e0e0e0;
}

/* SELECTBOX */
.stSelectbox > div > div {
    border-radius: 4px;
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

/* CHARTS */
.js-plotly-plot {
    background: white;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# RENDER NEWS TICKER
# ======================================

def fetch_rss_news(limit=10):
    urls = [
        "https://news.google.com/rss/search?q=danantara&hl=id&gl=ID&ceid=ID:id",
        "https://news.google.com/rss/search?q=pertamina&hl=id&gl=ID&ceid=ID:id"
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
# HEADER
# ======================================
current_hour = datetime.now().hour
if current_hour < 12:
    greeting = "Good Morning"
elif current_hour < 18:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

st.markdown(f"""
<div class='header'>
    <div class='header-content'>
        <div>
            <div class='header-title'>MTRAX</div>
            <div class='header-subtitle'>Corporate Travel Analytics</div>
        </div>
        <div class='header-user'>
            <div class='user-name'>{greeting}, {st.session_state.username.title()}</div>
            <div class='user-role'>{st.session_state.role} · {datetime.now().strftime('%d %b %Y')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

news_data = fetch_rss_news()
render_news_ticker(news_data)

# ======================================
# SIDEBAR
# ======================================
def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("assets/LOGO M FIX.png")

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:30px 15px;'>
        <img src="data:image/png;base64,{logo_base64}" style="width:80px;opacity:0.9;" />
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        logout()

    st.markdown("### DATA SOURCE")

    mode = st.radio(
        "",
        ["Cloud/Drive", "Upload Files"],
        label_visibility="collapsed"
    )

# ======================================
# DATA PREP
# ======================================
os.makedirs("data_temp", exist_ok=True)
drop_cols = ["Site (PSA)", "Site group Name", "Currency", "Reschedule ID", "Source_File"]
df_all = pd.DataFrame()

# ======================================
# CLOUD MODE
# ======================================
if mode == "Cloud/Drive":

    st.markdown("<div class='section-title'>Cloud Integration</div>", unsafe_allow_html=True)

    drive_options = {
        "2023–2025 (All Data)": "1vygKdg7enC5Kah7WbzVLsNI--S7Tyhvz",
        "2023 Only": "1xDFRdGLDiiScIwW9gTucRyeFCmuqNyq_",
        "2024 Only": "16ZMZ42BLN4GPbYKAd5h75ocbxFuyc85V",
        "2025 Only": "1chxbGHfk9hHNPZ8vlU6AqRVUKH1jEnxF"
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_dataset = st.selectbox("Dataset", list(drive_options.keys()))
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_btn = st.button("Load Data", type="primary", use_container_width=True)

    if process_btn:
        folder_id = drive_options[selected_dataset]

        with st.spinner("Loading..."):
            progress_bar = st.progress(0)
            try:
                import shutil
                progress_bar.progress(20)
                shutil.rmtree("data_temp", ignore_errors=True)
                os.makedirs("data_temp", exist_ok=True)

                progress_bar.progress(40)
                gdown.download_folder(id=folder_id, output="data_temp", quiet=False, use_cookies=False)

                progress_bar.progress(60)
                files = [f for f in os.listdir("data_temp") if f.endswith((".xlsx", ".xls"))]
                if not files:
                    st.error("No files found")
                    st.stop()

                df_list = []
                for i, f in enumerate(files):
                    try:
                        df = pd.read_excel(os.path.join("data_temp", f))
                        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                        df_list.append(df)
                        progress_bar.progress(60 + int(30 * (i + 1) / len(files)))
                    except Exception:
                        pass

                if not df_list:
                    st.error("No valid data")
                    st.stop()

                df_all = pd.concat(df_list, ignore_index=True)
                progress_bar.progress(100)
                st.success(f"Loaded {len(df_all):,} records")

            except Exception as e:
                st.error(f"Error: {e}")

# ======================================
# UPLOAD MODE
# ======================================
elif mode == "Upload Files":

    st.markdown("<div class='section-title'>Upload Files</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select Excel files",
        type=["xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing..."):
            df_list = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                try:
                    df = pd.read_excel(file)
                    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                    df_list.append(df)
                    progress_bar.progress(int((i + 1) / len(uploaded_files) * 100))
                except Exception:
                    pass

            if df_list:
                df_all = pd.concat(df_list, ignore_index=True)
                st.success(f"Loaded {len(df_all):,} records")

# ======================================
# DATA CLEANING
# ======================================
if not df_all.empty:
    if "Invoice Amount" in df_all.columns:
        df_all["Invoice Amount"] = pd.to_numeric(
            df_all["Invoice Amount"].astype(str).str.replace("$", "").str.replace(",", ""),
            errors="coerce"
        )
    if "Number of Rooms Night" in df_all.columns:
        df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors="coerce")
    if "Check in Date" in df_all.columns:
        df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce")
    if "Check Out Date" in df_all.columns:
        df_all["Check Out Date"] = pd.to_datetime(df_all["Check Out Date"], errors="coerce")

# ======================================
# DASHBOARD
# ======================================
if not df_all.empty:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard", 
        "Explorer", 
        "Analytics", 
        "ML Models",
        "Forecast",
        "Export"
    ])

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
                unique_emp = df_all["Employee Id"].nunique()
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>Travelers</div>
                        <div class='metric-value'>{unique_emp:,}</div>
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
                total_rooms = df_all["Number of Rooms Night"].sum()
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>Room Nights</div>
                        <div class='metric-value'>{total_rooms:,.0f}</div>
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
                unique_cost_center = df_all["Cost Center Pekerja"].nunique()
                st.metric("Cost Centers", f"{unique_cost_center:,}")

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
                    tr_rooms = df_all.groupby("Travel Request Number")["Number of Rooms Night"].sum()
                    avg_rooms = tr_rooms.mean()
                    max_rooms = tr_rooms.max()
                    
                    st.markdown(f"""
                        <div class='stats-card'>
                            <div class='stats-label'>Avg Nights</div>
                            <div class='stats-number'>{avg_rooms:.1f}</div>
                            <div class='stats-detail'>Max: {max_rooms:.0f}</div>
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
                if "Direktorat Pekerja" in df_all.columns and "Travel Request Number" in df_all.columns:
                    top_dir = (
                        df_all.groupby("Direktorat Pekerja")["Travel Request Number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )

                    fig_dir = px.bar(
                        top_dir,
                        x="Travel Request Number",
                        y="Direktorat Pekerja",
                        orientation="h",
                        text="Travel Request Number"
                    )

                    fig_dir.update_traces(
                        marker_color='#9c5789',
                        texttemplate="%{text:,}",
                        textposition="outside",
                        textfont=dict(size=11)
                    )

                    fig_dir.update_layout(
                        height=380,
                        title="Top 10 Directorates",
                        xaxis_title="",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=40, t=50, b=10),
                        showlegend=False
                    )

                    st.plotly_chart(fig_dir, use_container_width=True)

            with col2:
                if "Country" in df_all.columns:
                    df_all["Country"] = df_all["Country"].astype(str).str.strip()

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
                        color_discrete_sequence=["#9c5789", "#888888"]
                    )

                    fig_pie.update_traces(
                        textinfo="percent+label",
                        textfont=dict(size=12)
                    )

                    total = pie_df['Bookings'].sum()
                    
                    fig_pie.update_layout(
                        height=380,
                        title="Market Distribution",
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
                        annotations=[dict(
                            text=f"{total:,}",
                            x=0.5,
                            y=0.5,
                            font=dict(size=16),
                            showarrow=False
                        )]
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)
        st.info("Coming soon")
        
    with tab3:
        st.markdown("<div class='section-title'>Advanced Analytics</div>", unsafe_allow_html=True)
        st.info("Coming soon")
        
    with tab4:
        st.markdown("<div class='section-title'>Machine Learning</div>", unsafe_allow_html=True)
        st.info("Coming soon")
        
    with tab5:
        st.markdown("<div class='section-title'>Demand Forecast</div>", unsafe_allow_html=True)
        st.info("Coming soon")
        
    with tab6:
        st.markdown("<div class='section-title'>Export Data</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Download CSV", use_container_width=True, type="primary"):
                csv = df_all.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Click to Download",
                    data=csv,
                    file_name=f"data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if st.button("Download Excel", use_container_width=True, type="primary"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_all.to_excel(writer, index=False, sheet_name="Data")
                st.download_button(
                    "Click to Download",
                    data=buffer.getvalue(),
                    file_name=f"data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# ======================================
# FOOTER
# ======================================
st.markdown("""
<div class='divider' style='margin-top:50px;'></div>
<div style='text-align:center;padding:25px;color:#888888;font-size:0.85em;'>
    © 2025 Pertamina · MTRAX Travel Analytics
</div>
""", unsafe_allow_html=True)
