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

#==========================#
# FUNGSI AUTO-CANONICAL MAPPING
#==========================#
def auto_canonical_hotel_mapping(
    df,
    hotel_col="Hotel Name",
    threshold=0.88
):
    """
    Membuat canonical mapping hotel otomatis berbasis text similarity
    Output:
    - df dengan kolom tambahan: Canonical Hotel Name
    - mapping dataframe (raw → canonical)
    """

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
    "admin": {"password": hash_password("admin123"), "role": "Admin"},
    "ssc": {"password": hash_password("ssc123"), "role": "Analyst"},
    "dtm": {"password": hash_password("dtm123"), "role": "Viewer"},
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
        # TAB SEMUA
        # ======================================

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Dashboard", "Explorer", "Analytics", "ML Models", "Forecast", "Export"
        ])

        # ======================================
        # TAB 1: DASHBOARD
        # ======================================
        with tab1:

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

                selected_company = st.selectbox(
                    "Filter Overview berdasarkan Nama Perusahaan",
                    options=["All"] + company_list,
                    index=0
                )

                if selected_company != "All":
                    df_overview = df_all[df_all[company_col] == selected_company]
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
                    unique_tr = df_all["Travel Request Number"].nunique()
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Requests</div>
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
                    unique_cities = df_all["City"].nunique()
                    st.metric("Cities", f"{unique_cities:,}")

            with col4:
                if "Country" in df_overview.columns:
                    unique_countries = df_all["Country"].nunique()
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

                # =========================
                # CORE HEATMAP (COLOR)
                # =========================
                pivot_core = (
                    df_heat
                    .groupby(["Issue Day", "Issue Hour"])["Travel Request Number"]
                    .nunique()
                    .reset_index(name="TR_Count")
                    .pivot(index="Issue Day", columns="Issue Hour", values="TR_Count")
                    .reindex(day_order)
                    .fillna(0)
                )

                # TOTAL
                total_day = pivot_core.sum(axis=1)
                total_hour = pivot_core.sum(axis=0)
                grand_total = total_day.sum()

                fig = px.imshow(
                    pivot_core,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#ddd", "#9c5789"]
                )

                # =========================
                # ANNOTATION TOTAL (NO COLOR)
                # =========================
                annotations = []

                # Total per Hari (kanan)
                for i, day in enumerate(pivot_core.index):
                    annotations.append(dict(
                        x=len(pivot_core.columns),
                        y=i,
                        text=f"<b>{int(total_day.loc[day])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                # Total per Jam (bawah)
                for j, hour in enumerate(pivot_core.columns):
                    annotations.append(dict(
                        x=j,
                        y=len(pivot_core.index),
                        text=f"<b>{int(total_hour.loc[hour])}</b>",
                        showarrow=False,
                        font=dict(color="black", size=12)
                    ))

                # Grand Total (pojok kanan bawah)
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

                # Tambah space axis untuk total
                fig.update_xaxes(range=[-0.5, len(pivot_core.columns) + 0.5])
                fig.update_yaxes(range=[len(pivot_core.index) + 0.5, -0.5])

                st.plotly_chart(fig, use_container_width=True)


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

            # ======================================
            # TWO COLUMNS: CHARTS & TRENDS
            # ======================================
            col1, col2 = st.columns(2)

            # ======================================
            # COLUMN 1: TOP COMPANY & TR TREND
            # ======================================
            with col1:
                # Top 10 Directorates Chart
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

                # ================================
                # MONTHLY TREND — TRAVEL REQUEST
                # ================================
                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Monthly Trend — Travel Requests</div>", unsafe_allow_html=True)

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
                    
                    # === DOWNLOAD MONTHLY TRAVEL REQUEST TREND ===
                    output_tr_trend = BytesIO()
                    trend_df.to_excel(
                        output_tr_trend,
                        index=False,
                        sheet_name="Monthly Travel Request Trend"
                    )
                    output_tr_trend.seek(0)

                    st.download_button(
                        label="⬇️ Download Data (Excel)",
                        data=output_tr_trend,
                        file_name="monthly_travel_request_trend.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            # ======================================
            # COLUMN 2: MARKET DISTRIBUTION & ROOM NIGHTS TREND
            # ======================================
            with col2:
                # Market Distribution (Domestic vs International)
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

                    # === DOWNLOAD MONTHLY ROOM NIGHTS TREND ===
                    output_rn_trend = BytesIO()
                    trend_df.to_excel(
                        output_rn_trend,
                        index=False,
                        sheet_name="Monthly Room Nights Trend"
                    )
                    output_rn_trend.seek(0)

                    st.download_button(
                        label="⬇️ Download Data (Excel)",
                        data=output_rn_trend,
                        file_name="monthly_room_nights_trend.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                else:
                    st.info("Column 'Issue Time' or 'Number of Rooms Night' not available.")

            # ======================================
            # CITY NAME NORMALIZATION (PROPER CASE)
            # ======================================
            for city_col in ["City", "City Destination"]:
                if city_col in df_overview.columns:
                    df_overview[city_col] = (
                        df_overview[city_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .str.title()
                    )

            # ======================================
            # TOP 100 ANALYSIS BY ROOM NIGHTS
            # ======================================
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Top 100 Analysis by Room Nights</div>", unsafe_allow_html=True)
            
            cols1, cols2 = st.columns(2)
            
            # -------------------------------
            # COL 1 — Top 100 Hotel Name
            # -------------------------------
            with cols1:
                if "Hotel Name" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:
                    top_hotels = (
                        df_overview.groupby("Hotel Name")["Number of Rooms Night"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(100)
                        .reset_index()
                    )

                    # Ranking & Highlight
                    top_hotels["Rank"] = top_hotels.index + 1
                    top_hotels["Highlight"] = top_hotels["Rank"].apply(
                        lambda x: "Top 20" if x <= 20 else "Others"
                    )

                    fig_hotels = px.bar(
                        top_hotels,
                        x="Number of Rooms Night",
                        y="Hotel Name",
                        orientation="h",
                        color="Highlight",
                        color_discrete_map={
                            "Top 20": "#9c5789",
                            "Others": "#e0e0e0"
                        },
                        title="Top 100 Hotels by Total Room Nights"
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
                        xaxis=dict(
                            tickfont=dict(size=10)
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=80, t=50, b=10),
                        legend_title_text="",
                        showlegend=True
                    )

                    st.plotly_chart(fig_hotels, use_container_width=True)

                    # === DOWNLOAD TOP 100 HOTELS ===
                    output_hotels = BytesIO()
                    top_hotels.drop(columns=["Rank", "Highlight"]).to_excel(
                        output_hotels, index=False, sheet_name="Top 100 Hotels"
                    )
                    output_hotels.seek(0)

                    st.download_button(
                        label="⬇️ Download Data (Excel)",
                        data=output_hotels,
                        file_name="top_100_hotels_by_room_nights.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("Kolom 'Hotel Name' atau 'Number of Rooms Night' tidak ditemukan.")


            # -------------------------------
            # COL 2 — Top 100 City
            # -------------------------------
            with cols2:
                city_col = None
                for c in ["City", "City Destination"]:
                    if c in df_overview.columns:
                        city_col = c
                        break

                if city_col and "Number of Rooms Night" in df_overview.columns:
                    top_cities = (
                        df_overview.groupby(city_col)["Number of Rooms Night"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(100)
                        .reset_index()
                    )

                    # Ranking & Highlight
                    top_cities["Rank"] = top_cities.index + 1
                    top_cities["Highlight"] = top_cities["Rank"].apply(
                        lambda x: "Top 20" if x <= 20 else "Others"
                    )

                    fig_cities = px.bar(
                        top_cities,
                        x="Number of Rooms Night",
                        y=city_col,
                        orientation="h",
                        color="Highlight",
                        color_discrete_map={
                            "Top 20": "#9c5789",
                            "Others": "#e0e0e0"
                        },
                        title="Top 100 Cities by Total Room Nights"
                    )

                    fig_cities.update_traces(
                        texttemplate="%{x:,.0f}",
                        textposition="outside",
                        textfont_size=10
                    )

                    fig_cities.update_layout(
                        height=1700,
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(size=10)
                        ),
                        xaxis=dict(
                            tickfont=dict(size=10)
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=80, t=50, b=10),
                        legend_title_text="",
                        showlegend=True
                    )

                    st.plotly_chart(fig_cities, use_container_width=True)

                    # === DOWNLOAD TOP 100 CITIES ===
                    output_cities = BytesIO()
                    top_cities.drop(columns=["Rank", "Highlight"]).to_excel(
                        output_cities, index=False, sheet_name="Top 100 Cities"
                    )
                    output_cities.seek(0)

                    st.download_button(
                        label="⬇️ Download Data (Excel)",
                        data=output_cities,
                        file_name="top_100_cities_by_room_nights.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("Kolom City / City Destination atau Number of Rooms Night tidak ditemukan.")
        
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
                    st.dataframe(
                        sim_df,
                        use_container_width=True,
                        height=450
                    )

                    st.caption(
                        "🔎 Digunakan untuk mendeteksi potensi duplikasi nama hotel akibat perbedaan penulisan."
                    )

                    # Download
                    output = BytesIO()
                    sim_df.to_excel(output, index=False, sheet_name="Hotel Name Similarity")
                    output.seek(0)

                    st.download_button(
                        label="⬇️ Download Similarity Result (Excel)",
                        data=output,
                        file_name="hotel_name_similarity.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
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
                hotel_mapping.to_excel(
                    output,
                    index=False,
                    sheet_name="Hotel Canonical Mapping"
                )
                output.seek(0)

                st.download_button(
                    "⬇️ Download Canonical Mapping (Excel)",
                    data=output,
                    file_name="hotel_canonical_mapping.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Tidak ada mapping canonical yang terbentuk.")


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
