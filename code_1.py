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
import secrets

from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import xml.etree.ElementTree as ET

# Penyeragaman nama Hotel Name / Supplier Name (Meliá, Mövenpick, éL Hotel,
# Courtyard by Marriott, + kamus data/unique_hotels.xlsx)
from hotel_name_standardizer import standardize_hotel_columns, standardize_name

# Anonimisasi: whitelist 13 kolom + kode acak TR/Booking ID (kunci baru tiap penarikan data)
from mask_data import mask_dataframe, select_columns, KEEP_COLUMNS

# Tab Cost Avoidance
from cost_avoidance import render_cost_avoidance

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
        img = qr.make_image(fill_color="#1BA0E2", back_color="white")
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
                dark="#1BA0E2", light="white")
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
    Secret TOTP PER PENGGUNA. Prioritas: secrets.toml [totp][<username>].
    Bila belum ada, dibuat sementara di sesi pengguna itu sendiri (untuk halaman setup)
    dan harus disimpan Admin ke secrets.toml. Tidak ada lagi secret bersama.
    """
    result = {}
    for uname in USERS.keys():
        own = _own_totp_secret(uname)
        if own:
            result[uname] = own
        else:
            key = f"_totp_secret_{uname}"
            if key not in st.session_state:
                st.session_state[key] = _generate_totp_secret()
            result[uname] = st.session_state[key]
    return result


def _own_totp_secret(username: str):
    """Secret TOTP milik user dari [totp][<username>], atau None."""
    try:
        val = st.secrets["totp"].get(username)
        return str(val).strip() if val else None
    except Exception:
        return None


# ======================================
# ADVANCED LOGIN SECURITY CONFIG
# ======================================

# Jika pakai secrets.toml (recommended)
MAX_LOGIN_ATTEMPTS = st.secrets.get("security", {}).get("max_login_attempts", 5)
LOCKOUT_SECONDS = st.secrets.get("security", {}).get("lockout_seconds", 300)  # 5 menit
SESSION_TIMEOUT = st.secrets.get("security", {}).get("session_timeout_minutes", 30) * 60

def secure_compare(plain_password, stored_hash):
    """
    Verifikasi password. Mendukung hash Argon2id (disarankan, diawali "$argon2")
    dan hash SHA-256 lama (64 karakter hex) agar migrasi bisa bertahap.
    Buat hash Argon2: python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash('PASSWORD'))"
    """
    stored_hash = str(stored_hash or "")
    if stored_hash.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            from argon2.exceptions import VerifyMismatchError, InvalidHashError
            try:
                return PasswordHasher().verify(stored_hash, plain_password)
            except (VerifyMismatchError, InvalidHashError):
                return False
        except ImportError:
            return False
    hashed_input = hashlib.sha256(plain_password.encode()).hexdigest()
    return hmac.compare_digest(hashed_input, stored_hash)


@st.cache_resource
def _login_lock_store():
    """Penguncian login di sisi server per username (tidak reset walau halaman di-refresh)."""
    return {}


def _user_locked(username: str) -> float:
    e = _login_lock_store().get(username)
    return max(0.0, e["until"] - time.time()) if e else 0.0


def _user_fail(username: str):
    store = _login_lock_store()
    e = store.setdefault(username, {"failed": 0, "until": 0.0})
    e["failed"] += 1
    if e["failed"] >= MAX_LOGIN_ATTEMPTS:
        e["until"] = time.time() + LOCKOUT_SECONDS
        e["failed"] = 0


def _user_ok(username: str):
    _login_lock_store().pop(username, None)

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
            _revoke_session_token(st.session_state.get("_session_token"))
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _clear_session_cookie()
            st.warning("Session expired. Please login again.")
            st.stop()


# ======================================
# PERSISTENT LOGIN ACROSS PAGE RELOAD
# ── session_state Streamlit terikat ke koneksi WebSocket browser, jadi akan
#    kosong lagi setiap kali halaman di-reload (F5). Token sesi acak disimpan
#    di COOKIE browser (bukan URL query param — st.query_params terbukti tidak
#    reliable bertahan setelah reload, ini bug yang sudah dikonfirmasi tim
#    Streamlit sendiri: github.com/streamlit/streamlit/issues/10406).
#
#    Streamlit hanya bisa MEMBACA cookie secara native (st.context.cookies,
#    read-only, sejak v1.38+). Untuk MENULIS cookie, tetap perlu sedikit
#    JavaScript lewat components.html — Streamlit belum punya API native
#    untuk itu (github.com/streamlit/streamlit/issues/7892).
#
#    CATATAN KEAMANAN: cookie diset tanpa flag HttpOnly (karena ditulis lewat
#    JS di sisi klien, bukan header HTTP dari server), jadi secara teknis
#    bisa dibaca script lain di origin yang sama. Untuk aplikasi internal ini
#    risikonya wajar selama diakses di jaringan/perangkat tepercaya, dan
#    session_timeout_minutes di secrets.toml (default 30 menit) membatasi
#    umur token-nya.
# ======================================
_SESSION_COOKIE_NAME = "mtrax_session"

@st.cache_resource
def _get_session_store():
    """Dict bersama di semua sesi selama proses server ini hidup: token -> {username, role, expires_at}."""
    return {}

def _create_session_token(username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    store = _get_session_store()
    store[token] = {
        "username": username,
        "role": role,
        "expires_at": time.time() + SESSION_TIMEOUT,
    }
    return token

def _validate_session_token(token: str):
    """Return (username, role) kalau token valid & belum kedaluwarsa, else None. Otomatis bersihkan token basi."""
    if not token:
        return None
    store = _get_session_store()
    entry = store.get(token)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        store.pop(token, None)
        return None
    # Sliding expiry: perpanjang masa berlaku selama masih aktif dipakai
    entry["expires_at"] = time.time() + SESSION_TIMEOUT
    return entry["username"], entry["role"]

def _revoke_session_token(token):
    if token:
        _get_session_store().pop(token, None)

def _set_session_cookie(token: str):
    """Tulis cookie sesi di browser lewat JS (Streamlit tidak punya API native untuk set cookie).

    ── CATATAN PERBAIKAN ───────────────────────────────────────────────────
    Sempat dicoba memicu `window.parent.location.reload()` dari DALAM script
    ini (persis setelah `document.cookie = ...`) supaya urutan cookie->reload
    terjamin tanpa bergantung ke `st.rerun()`. Ternyata pendekatan itu TIDAK
    reliable: iframe yang dipakai `components.html` tidak selalu diberi izin
    untuk menavigasi/reload frame induknya oleh browser, sehingga reload bisa
    gagal diam-diam atau menyebabkan reload terjadi SEBELUM cookie sempat
    tersimpan — hasilnya user malah tidak pernah berhasil masuk ke dashboard.

    Jadi pendekatan dikembalikan ke cara yang terbukti reliable: cookie
    ditulis di sini, lalu pemanggil (mis. _finish_login) memakai `st.rerun()`
    untuk lanjut ke dashboard DALAM SESI YANG SAMA — ini tidak bergantung
    sama sekali pada cookie/JS karena `st.session_state.authenticated` sudah
    di-set duluan sebelum rerun. Cookie di sini murni untuk keperluan lain:
    memulihkan sesi kalau nanti browser BENAR-BENAR di-refresh (F5). Supaya
    itu tetap reliable, pemanggil perlu memberi jeda (time.sleep) yang cukup
    sebelum memanggil st.rerun(), agar browser sempat memuat iframe & benar-
    benar mengeksekusi baris document.cookie di dalamnya sebelum DOM diganti
    oleh render hasil rerun.
    """
    import streamlit.components.v1 as _components
    max_age = int(SESSION_TIMEOUT)
    _components.html(
        f"""<script>
        document.cookie = "{_SESSION_COOKIE_NAME}={token}; max-age={max_age}; path=/; SameSite=Lax";
        </script>""",
        height=0
    )

def _clear_session_cookie():
    """Hapus cookie sesi di browser (dipanggil saat logout / timeout)."""
    import streamlit.components.v1 as _components
    _components.html(
        f"""<script>
        document.cookie = "{_SESSION_COOKIE_NAME}=; max-age=0; path=/; SameSite=Lax";
        </script>""",
        height=0
    )

def _try_restore_session_from_cookie():
    """Dipanggil sebelum pengecekan authenticated — pulihkan sesi dari cookie kalau ada & valid."""
    if st.session_state.get("authenticated"):
        return
    try:
        token = st.context.cookies.get(_SESSION_COOKIE_NAME)
    except Exception:
        token = None
    if not token:
        return
    result = _validate_session_token(token)
    if result is None:
        return
    username, role = result
    # Tolak token milik akun yang sudah dinonaktifkan/kedaluwarsa atau rolenya berubah
    if username not in USERS or USERS[username]["role"] != role:
        _revoke_session_token(token)
        return
    st.session_state.login_attempts = 0
    st.session_state.lockout_until  = 0
    st.session_state.authenticated  = True
    st.session_state.logged_in      = True
    st.session_state.username       = username
    st.session_state.role           = role
    st.session_state.login_time     = time.time()
    st.session_state.pending_2fa    = False
    st.session_state.pending_user   = ""
    st.session_state._session_token = token

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


# =====================================================================
# GOOGLE DRIVE LOADER VIA SERVICE ACCOUNT (menggantikan gdown)
# Butuh: google-api-python-client, google-auth di requirements.txt
# Secrets: [gcp_service_account] dan [drive_folders]
# =====================================================================
import io
import re

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]  # hanya baca

_MIME_FOLDER = "application/vnd.google-apps.folder"
_MIME_GSHEET = "application/vnd.google-apps.spreadsheet"
_MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_MIME_XLS = "application/vnd.ms-excel"

_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]{10,100}")


@st.cache_resource
def _drive_service():
    """Klien Drive API yang diautentikasi dengan service account dari st.secrets."""
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_children(service, folder_id):
    """Daftar semua item (bukan sampah) di dalam satu folder, dengan paginasi."""
    items, page_token = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,          # supaya Shared Drive juga terbaca
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            return items


def _collect_excel_files(service, folder_id, depth=0, max_depth=5):
    """Cari file Excel / Google Sheets secara rekursif (meniru gdown.download_folder)."""
    found = []
    for item in _list_children(service, folder_id):
        mime, name = item["mimeType"], item["name"].lower()
        if mime == _MIME_FOLDER:
            if depth < max_depth:
                found.extend(_collect_excel_files(service, item["id"], depth + 1, max_depth))
        elif mime in (_MIME_XLSX, _MIME_XLS, _MIME_GSHEET) or name.endswith((".xlsx", ".xls")):
            found.append(item)
    return found


def _download_to_memory(service, item):
    """Unduh file ke RAM (tidak ada file sementara di disk server)."""
    if item["mimeType"] == _MIME_GSHEET:
        request = service.files().export_media(fileId=item["id"], mimeType=_MIME_XLSX)
    else:
        request = service.files().get_media(fileId=item["id"], supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------
# PERCEPATAN PENARIKAN DATA
#   1. Cache Parquet di disk VPS: setiap file Excel yang sudah dibaca disimpan
#      sebagai .parquet, diberi penanda "modifiedTime" dari Drive. File yang
#      tidak berubah tidak diunduh/dibaca ulang, dan cache tetap ada walau
#      service di-restart. Folder cache (di luar folder Git), urutan prioritas:
#        secrets [cache] dir  ->  env MTRAX_CACHE_DIR  ->  ~/.cache/mtrax/drive
#   2. File yang perlu diunduh diproses paralel (beberapa sekaligus).
#   3. Excel dibaca dengan engine "calamine" (jauh lebih cepat) bila
#      python-calamine terpasang; jika tidak, otomatis kembali ke openpyxl.
# requirements.txt: pyarrow, python-calamine  (keduanya opsional, ada fallback)
# ---------------------------------------------------------------------
import glob as _glob
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPool

DRIVE_CACHE_TTL = 30 * 60      # cek ulang daftar file Drive tiap 30 menit (murah: file lama dari Parquet)
DRIVE_DOWNLOAD_WORKERS = 4     # jumlah file yang diunduh bersamaan
_drive_thread_local = _threading.local()


def _drive_cache_base():
    base = ""
    try:
        base = str(dict(st.secrets.get("cache", {})).get("dir", "") or "")
    except Exception:
        base = ""
    base = base or os.environ.get("MTRAX_CACHE_DIR", "") or os.path.expanduser("~/.cache/mtrax/drive")
    return base


def _drive_cache_dir(folder_id):
    """Folder cache per folder Drive (izin 700). None jika tidak bisa dibuat -> tanpa cache disk."""
    try:
        path = os.path.join(_drive_cache_base(), str(folder_id))
        os.makedirs(path, mode=0o700, exist_ok=True)
        return path if os.access(path, os.W_OK) else None
    except Exception:
        return None


def _drive_creds():
    return service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_DRIVE_SCOPES
    )


def _thread_drive_service(creds):
    """Klien Drive per thread (klien googleapiclient tidak thread-safe)."""
    svc = getattr(_drive_thread_local, "service", None)
    if svc is None:
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        _drive_thread_local.service = svc
    return svc


def _read_excel_fast(buf):
    try:
        return pd.read_excel(buf, engine="calamine")
    except Exception:
        buf.seek(0)
        return pd.read_excel(buf)


def _cache_stem(item):
    stamp = re.sub(r"[^0-9A-Za-z]", "", str(item.get("modifiedTime", "")))
    return f"{item['id']}_{stamp or 'nostamp'}"


def _read_cached(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        # Parquet mengembalikan None untuk sel kosong di kolom teks; samakan
        # dengan hasil read_excel (np.nan) supaya logika lama (mis. "nan") tetap sama.
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].where(df[col].notna(), np.nan)
        return df
    return pd.read_pickle(path)


def _write_cached(df, cache_dir, stem, file_id):
    """Simpan atomik (tulis ke .tmp lalu rename). Parquet, atau pickle jika tipe data tidak didukung."""
    for old in _glob.glob(os.path.join(cache_dir, f"{file_id}_*")):
        try:
            os.remove(old)                      # versi lama file yang sama
        except OSError:
            pass
    for ext in (".parquet", ".pkl"):
        final = os.path.join(cache_dir, stem + ext)
        tmp = final + f".{os.getpid()}.{_threading.get_ident()}.tmp"
        try:
            if ext == ".parquet":
                df.to_parquet(tmp, index=False)
            else:
                df.to_pickle(tmp)
            os.chmod(tmp, 0o600)
            os.replace(tmp, final)
            return
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_drive_item(item, creds, cache_dir):
    stem = _cache_stem(item)
    if cache_dir:
        for ext in (".parquet", ".pkl"):
            path = os.path.join(cache_dir, stem + ext)
            if os.path.exists(path):
                try:
                    return _read_cached(path)
                except Exception:
                    try:
                        os.remove(path)          # cache rusak -> unduh ulang
                    except OSError:
                        pass
    service = _thread_drive_service(creds)
    df = _read_excel_fast(_download_to_memory(service, item))
    # Whitelist 13 kolom SEBELUM disimpan ke cache disk: kolom lain tidak pernah tersimpan di VPS
    df = select_columns(df)[0]
    df = trim_string_columns(df)  # fungsi yang sudah ada di file utama
    if cache_dir:
        _write_cached(df, cache_dir, stem, item["id"])
    return df


def _prune_cache(cache_dir, items):
    """Hapus cache untuk file yang sudah tidak ada di folder Drive."""
    keep = {it["id"] for it in items}
    for path in _glob.glob(os.path.join(cache_dir, "*")):
        name = os.path.basename(path)
        if name.endswith(".tmp"):
            continue
        if name.rsplit("_", 1)[0] not in keep:
            try:
                os.remove(path)
            except OSError:
                pass


@st.cache_data(show_spinner=False, ttl=DRIVE_CACHE_TTL, max_entries=6)
def load_drive_data(folder_id, drop_cols):
    if not _DRIVE_ID_RE.fullmatch(str(folder_id)):
        raise ValueError("Folder ID tidak valid.")

    items = _collect_excel_files(_drive_service(), folder_id)
    if not items:
        return pd.DataFrame()

    creds = _drive_creds()
    cache_dir = _drive_cache_dir(folder_id)
    workers = max(1, min(DRIVE_DOWNLOAD_WORKERS, len(items)))
    if workers == 1:
        dfs = [_load_drive_item(it, creds, cache_dir) for it in items]
    else:
        with _ThreadPool(max_workers=workers) as pool:
            dfs = list(pool.map(lambda it: _load_drive_item(it, creds, cache_dir), items))
    if cache_dir:
        _prune_cache(cache_dir, items)

    dfs = [df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore") for df in dfs]
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return pd.DataFrame()
    # Masking setelah SEMUA file digabung -> kode konsisten di seluruh data.
    # Kunci acak baru setiap kali data ditarik ulang (cache habis / tombol 🔄).
    return mask_dataframe(pd.concat(dfs, ignore_index=True))

#==========================#
# FUNGSI AUTO-CANONICAL MAPPING
#==========================#
@st.cache_data(show_spinner=False)
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
@st.cache_data(show_spinner=False)
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


#==========================#
# PROPERTY MERGE (Price & Room Intelligence)
#==========================#
# Kata generik yang dibuang sebelum membandingkan nama hotel, supaya
# "Aston Balikpapan Hotel & Residence" == "Aston Balikpapan".
_PROPERTY_GENERIC_WORDS = {
    "hotel", "hotels", "the", "and", "by", "at", "of", "residence", "residences",
    "resort", "resorts", "inn", "suite", "suites", "convention", "center", "centre",
    "pt", "tbk", "spa", "villa", "villas",
}

# Alias manual untuk kasus yang tidak tertangkap otomatis.
# Format: "Nama asli persis di data": "Nama properti yang diinginkan"
# Contoh: "Swissbel Hotel Balikpapan": "Swiss-Belhotel Balikpapan",
PROPERTY_MANUAL_ALIAS = {
}


def _fold_text(value) -> str:
    """Huruf kecil + buang aksen (é→e, ü→u) + rapikan spasi non-standar."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    return s.lower()


# Awalan/akhiran kota yang diabaikan saat membandingkan kota
_CITY_NOISE_WORDS = {"kota", "kab", "kabupaten", "city", "regency", "adm", "administrasi",
                     "daerah", "istimewa", "provinsi", "prov", "dki", "di"}

# Variasi penulisan kota / wilayah metro -> satu kunci kota.
# Tambahkan sendiri bila menemukan variasi lain di data.
CITY_ALIAS = {
    "JOGJA": "YOGYAKARTA", "JOGJAKARTA": "YOGYAKARTA", "YOGYA": "YOGYAKARTA",
    "YOGJAKARTA": "YOGYAKARTA", "DIY": "YOGYAKARTA", "SLEMAN": "YOGYAKARTA", "BANTUL": "YOGYAKARTA",
    "JAKARTA PUSAT": "JAKARTA", "JAKARTA SELATAN": "JAKARTA", "JAKARTA BARAT": "JAKARTA",
    "JAKARTA TIMUR": "JAKARTA", "JAKARTA UTARA": "JAKARTA", "JKT": "JAKARTA",
    "BADUNG": "BALI", "DENPASAR": "BALI", "KUTA": "BALI", "NUSA DUA": "BALI",
    "SEMINYAK": "BALI", "UBUD": "BALI", "GIANYAR": "BALI", "JIMBARAN": "BALI", "SANUR": "BALI",
    "BANDUNG BARAT": "BANDUNG", "CIMAHI": "BANDUNG",
}


def _normalize_city_key(city) -> str:
    if city is None or (isinstance(city, float) and np.isnan(city)):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", _fold_text(city))
    tokens = [t for t in s.split() if t not in _CITY_NOISE_WORDS]
    key = " ".join(tokens).upper()
    if key in ("NAN", "NONE", "NULL", "-"):
        return ""
    return CITY_ALIAS.get(key, key)


def _city_words(city_key: str) -> set:
    """Semua kata yang merujuk ke kota ini (nama kota + seluruh aliasnya)."""
    if not city_key:
        return set()
    words = set(city_key.lower().split())
    for alias, canon in CITY_ALIAS.items():
        if canon == city_key:
            words |= set(alias.lower().split())
    return words


def _singular(token: str) -> str:
    """Bentuk tunggal sederhana: 'points' -> 'point' (kata >3 huruf, bukan akhiran 'ss')."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalize_property_name(name, city_key: str = "") -> str:
    """
    Normalisasi nama hotel untuk dibandingkan: tanpa aksen, tanpa kata generik,
    tanpa nama kotanya sendiri (perbandingan sudah dibatasi per kota), kata diurutkan.
    'Hotel Meliá Purosani Yogyakarta' dan 'Melia Purosani Hotel' -> 'melia purosani'
    """
    s = re.sub(r"[^a-z0-9 ]", " ", _fold_text(name))
    tokens = [_singular(t) for t in s.split() if t not in _PROPERTY_GENERIC_WORDS]
    city_ws = _city_words(city_key)
    no_city = [t for t in tokens if t not in city_ws]
    if no_city:                 # jangan sampai nama kosong (mis. "Hotel Bandung")
        tokens = no_city
    if not tokens:              # nama hanya berisi kata generik -> pakai apa adanya
        tokens = s.split()
    return " ".join(sorted(tokens))


@st.cache_data(show_spinner=False)
def build_property_merge_map(pairs_df: pd.DataFrame, threshold: float = 0.88) -> pd.DataFrame:
    """
    pairs_df: kolom [_hotel_raw, _city_key, _freq] (unik per hotel+kota).
    Return: kolom [_hotel_raw, _city_key, Hotel / Properti].
    Hotel hanya digabung bila kotanya sama; nama tampilan properti =
    nama asli yang paling sering muncul di data.
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=["_hotel_raw", "_city_key", "Hotel / Properti"])

    df = pairs_df.copy()
    df["_alias"] = df["_hotel_raw"].map(PROPERTY_MANUAL_ALIAS).fillna(df["_hotel_raw"])
    df["_norm"] = [_normalize_property_name(n, c) for n, c in zip(df["_alias"], df["_city_key"])]
    df["_cluster"] = ""

    for city, grp in df.groupby("_city_key", sort=False):
        norms = grp["_norm"].unique().tolist()
        parent = list(range(len(norms)))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(norms) > 1:
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(norms)
            sim = cosine_similarity(tfidf)
            ii, jj = np.where(np.triu(sim, k=1) >= threshold)
            for i, j in zip(ii, jj):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[rj] = ri

            # Aturan subset: nama pendek (>=2 kata) yang seluruh katanya terkandung
            # di nama lain, mis. "four point" ⊂ "four point sheraton"
            # (Four Points by Sheraton). Hanya digabung bila kandidatnya menunjuk
            # ke SATU properti saja, agar nama umum tidak menarik banyak properti.
            token_sets = [set(n.split()) for n in norms]
            for i, ti in enumerate(token_sets):
                if len(ti) < 2:
                    continue
                cand_roots = {_find(j) for j, tj in enumerate(token_sets) if j != i and ti < tj}
                if len(cand_roots) == 1:
                    ri, rj = _find(i), cand_roots.pop()
                    if ri != rj:
                        parent[ri] = rj

        root_of = {n: f"{city}||{_find(k)}" for k, n in enumerate(norms)}
        df.loc[grp.index, "_cluster"] = grp["_norm"].map(root_of)

    # Baris tanpa kota: ikut cluster dengan nama ternormalisasi yang sama
    # bila nama tersebut hanya ada di satu cluster berkota.
    blank = df["_city_key"].eq("")
    if blank.any() and (~blank).any():
        known = df[~blank]
        lookup = {}  # (kota, nama ternormalisasi) -> set cluster
        for cl, ck, nm in zip(known["_cluster"], known["_city_key"], known["_norm"]):
            lookup.setdefault((ck, nm), set()).add(cl)
        cities = known["_city_key"].unique().tolist()
        for idx in df.index[blank]:
            raw_name = df.at[idx, "_alias"]
            cands = set()
            for ck in cities:  # normalisasi ulang memakai tiap kota kandidat
                cands |= lookup.get((ck, _normalize_property_name(raw_name, ck)), set())
                if len(cands) > 1:
                    break
            if len(cands) == 1:
                df.at[idx, "_cluster"] = cands.pop()

    # Nama tampilan = alias/nama asli dengan frekuensi terbesar di cluster
    # (jika frekuensi sama, pilih nama terpanjang — biasanya nama resmi lengkap)
    df["_len"] = df["_alias"].str.len()
    display = (
        df.sort_values(["_freq", "_len"], ascending=[False, False])
          .groupby("_cluster")["_alias"].first()
    )
    df["Hotel / Properti"] = df["_cluster"].map(display)
    return df[["_hotel_raw", "_city_key", "Hotel / Properti"]]


def get_property_merge_map(df: pd.DataFrame, hotel_col: str = "Hotel Name",
                           city_col: str = "City") -> pd.DataFrame:
    """Bangun peta penggabungan properti dari df (hasil di-cache)."""
    if hotel_col not in df.columns:
        return pd.DataFrame(columns=["_hotel_raw", "_city_key", "Hotel / Properti"])
    tmp = pd.DataFrame({
        "_hotel_raw": df[hotel_col].fillna("").astype(str).str.strip(),
        "_city_key": (df[city_col].map(_normalize_city_key)
                      if city_col in df.columns else "ALL"),
    })
    tmp = tmp[tmp["_hotel_raw"].ne("") & tmp["_hotel_raw"].str.lower().ne("nan")]
    pairs = tmp.value_counts().reset_index(name="_freq")
    return build_property_merge_map(pairs)


def apply_property_merge(df: pd.DataFrame, prop_map: pd.DataFrame,
                         hotel_col: str = "Hotel Name", city_col: str = "City") -> pd.DataFrame:
    """Tambahkan kolom 'Hotel / Properti' ke df berdasarkan prop_map."""
    out = df.copy()
    out["_hotel_raw"] = out[hotel_col].fillna("").astype(str).str.strip()
    out["_city_key"] = (out[city_col].map(_normalize_city_key)
                        if city_col in out.columns else "ALL")
    out = out.merge(prop_map, on=["_hotel_raw", "_city_key"], how="left")
    out["Hotel / Properti"] = out["Hotel / Properti"].fillna(out["_hotel_raw"])
    return out.drop(columns=["_hotel_raw", "_city_key"])


def build_member_names(df: pd.DataFrame, key_col: str, hotel_col: str = "Hotel Name",
                       weight_col: str = None) -> pd.Series:
    """Series: key -> 'Nama A; Nama B' (diurutkan dari kontribusi terbesar)."""
    tmp = df[[key_col, hotel_col]].copy()
    tmp[hotel_col] = tmp[hotel_col].astype(str).str.strip()
    if weight_col and weight_col in df.columns:
        tmp["_w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    else:
        tmp["_w"] = 1
    agg = (tmp.groupby([key_col, hotel_col])["_w"].sum().reset_index()
              .sort_values([key_col, "_w"], ascending=[True, False]))
    return agg.groupby(key_col)[hotel_col].apply(lambda s: "; ".join(s.tolist()))


def excel_bytes_with_formats(df: pd.DataFrame, sheet_name: str,
                             pct_cols=(), money_cols=(), int_cols=(), dec_cols=()) -> BytesIO:
    """Tulis df ke Excel; kolom persen (skala 0-100) disimpan sebagai persen asli Excel."""
    out = df.copy()
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c] / 100.0
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            out.to_excel(writer, index=False, sheet_name=sheet_name)
            wb, ws = writer.book, writer.sheets[sheet_name]
            fmt = {
                "pct": wb.add_format({"num_format": "0.00%"}),
                "money": wb.add_format({"num_format": '"Rp"#,##0'}),
                "int": wb.add_format({"num_format": "#,##0"}),
                "dec": wb.add_format({"num_format": "#,##0.00"}),
            }
            for i, c in enumerate(out.columns):
                kind = ("pct" if c in pct_cols else "money" if c in money_cols
                        else "int" if c in int_cols else "dec" if c in dec_cols else None)
                width = min(60, max(12, int(out[c].astype(str).str.len().max() or 0) + 2, len(c) + 2))
                ws.set_column(i, i, width, fmt[kind] if kind else None)
    except ImportError:
        buf = BytesIO()
        out.to_excel(buf, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf


def tab_date_range_filter(df: pd.DataFrame, date_col, key_prefix: str, col_from, col_to):
    """
    Filter tanggal From/To khusus satu tab (di dalam rentang Global Filter).
    Return: (df_terfilter, label_periode). Jika kolom tanggal tidak ada, df dikembalikan utuh.
    """
    if not date_col or date_col not in df.columns or df.empty:
        return df, "Semua periode"
    _dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    _valid = _dates.dropna()
    if _valid.empty:
        return df, "Semua periode"
    mn, mx = _valid.min().date(), _valid.max().date()

    k_from, k_to = f"{key_prefix}_date_from", f"{key_prefix}_date_to"
    # Jaga nilai tersimpan tetap di dalam rentang data (mis. setelah Global Filter berubah)
    if k_from not in st.session_state or not (mn <= st.session_state[k_from] <= mx):
        st.session_state[k_from] = mn
    if k_to not in st.session_state or not (mn <= st.session_state[k_to] <= mx):
        st.session_state[k_to] = mx

    _lbl = ("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;"
            "letter-spacing:0.08em;margin-bottom:6px;'>{}</div>")
    with col_from:
        st.markdown(_lbl.format(f"📅 From · {date_col}"), unsafe_allow_html=True)
        d_from = st.date_input("from", min_value=mn, max_value=mx, key=k_from,
                               format="YYYY/MM/DD", label_visibility="collapsed")
    with col_to:
        st.markdown(_lbl.format("📅 To"), unsafe_allow_html=True)
        d_to = st.date_input("to", min_value=mn, max_value=mx, key=k_to,
                             format="YYYY/MM/DD", label_visibility="collapsed")

    if d_from > d_to:
        st.warning("⚠️ Tanggal 'From' lebih besar dari 'To'. Filter tanggal tab ini diabaikan.")
        return df, f"{mn:%d %b %Y} – {mx:%d %b %Y}"

    mask = (_dates.dt.date >= d_from) & (_dates.dt.date <= d_to)
    return df[mask.fillna(False)], f"{d_from:%d %b %Y} – {d_to:%d %b %Y}"


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
# AKSES, AUDIT LOG, AKUN DEMO & KONVERSI MATA UANG
#
# Sumber data hanya berisi 13 kolom transaksi hotel (lihat mask_data.KEEP_COLUMNS),
# tanpa nama / Employee Id / perusahaan / cost center. No. TR & Booking ID sudah
# berupa kode acak saat data dimuat (mask_data.mask_dataframe).
#
# Akun & role (lihat USER_ROLES):
#   superadmin (Super Admin) : semua akses + audit log
#   admin      (Admin)       : semua tab, unduh, upload Excel, kelola corp rate & target manual
#   ssc        (Analyst)     : semua tab, unduh laporan agregat
#   dtm        (Viewer)      : semua tab, unduh laporan agregat
#   demo       (Demo)        : semua tab dengan nominal disamarkan, TANPA unduhan
# Setiap unduhan dan login dicatat di audit log.
#
# Contoh secrets:
#   [audit]
#   path = "/var/lib/mtrax/audit.log"      # opsional
#   [fx.rates]                              # kurs cadangan ke Rupiah
#   USD = 16300
#   SGD = 12150
#   [fx]
#   file = "/var/lib/mtrax/fx_rates.xlsx"   # opsional: kolom Bulan (YYYY-MM), Currency, Rate
#   [access]
#   login_required = false                  # SEMENTARA tanpa login (default: true)
#   guest_role = "Viewer"                   # hak akses pengunjung: Viewer / Analyst / Demo / Admin / Super Admin
#   [demo]
#   enabled = true                          # default: true (akun demo tetap butuh [users.demo] password_hash)
#   periods = ["2026"]
#   expires = "2026-12-31"
# ===============================

def _audit_path() -> str:
    p = ""
    try:
        p = str(dict(st.secrets.get("audit", {})).get("path", "") or "")
    except Exception:
        p = ""
    p = p or os.environ.get("MTRAX_AUDIT_LOG", "") or os.path.expanduser("~/.local/share/mtrax/audit.log")
    try:
        os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
    except Exception:
        pass
    return p


def audit(event: str, **detail):
    """Catat aktivitas (login, unduhan, upload) ke file audit & log server. Tanpa data transaksi."""
    ip = ""
    try:
        ip = str(st.context.ip_address or "")
    except Exception:
        pass
    rec = {
        "waktu": datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S"),
        "akun": str(detail.pop("user", None) or st.session_state.get("username", "")),
        "role": str(detail.pop("role", None) or st.session_state.get("role", "")),
        "event": event, "ip": ip, **{k: str(v) for k, v in detail.items()},
    }
    try:
        line = json.dumps(rec, ensure_ascii=False, default=str)
    except Exception:
        return  # audit log tidak boleh menghentikan aplikasi
    print(f"[AUDIT] {line}", flush=True)
    try:
        path = _audit_path()
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if new:
            os.chmod(path, 0o600)
    except Exception:
        pass


def read_audit_tail(n: int = 200) -> pd.DataFrame:
    try:
        with open(_audit_path(), encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return pd.DataFrame([json.loads(x) for x in lines if x.strip()]).iloc[::-1]
    except Exception:
        return pd.DataFrame()


def _demo_config():
    """Konfigurasi demo, atau None bila dimatikan ([demo] enabled = false) / kedaluwarsa.
    Password demo diisi di [users.demo] password_hash (tidak ada password bawaan)."""
    try:
        cfg = dict(st.secrets.get("demo", {}))
    except Exception:
        cfg = {}
    if not bool(cfg.get("enabled", True)):
        return None
    expires = str(cfg.get("expires", "")).strip()
    if expires:
        try:
            today = datetime.now(ZoneInfo("Asia/Jakarta")).date()
            if today > datetime.strptime(expires, "%Y-%m-%d").date():
                return None
        except ValueError:
            return None
    periods = cfg.get("periods") or []
    return {"periods": [str(p) for p in periods]}


def is_demo_user():
    return st.session_state.get("role") == "Demo"


def is_admin() -> bool:
    """Super Admin & Admin: upload Excel, kelola corp rate & target manual renegosiasi."""
    return st.session_state.get("role") in ("Super Admin", "Admin")


def is_super_admin() -> bool:
    return st.session_state.get("role") == "Super Admin"


def can_download() -> bool:
    """Semua akun kecuali Demo boleh mengunduh LAPORAN AGREGAT; setiap unduhan dicatat.
    Demo tidak bisa mengunduh apa pun (tombol tidak ditampilkan & file tidak pernah dikirim)."""
    return st.session_state.get("role") in ("Super Admin", "Admin", "Analyst", "Viewer")


# ── Satu titik kontrol untuk SEMUA tombol download ──
# Demo: tombol tidak dirender. Akun lain: unduhan dicatat di audit log.
def _install_download_guard():
    from streamlit.delta_generator import DeltaGenerator
    if getattr(DeltaGenerator.download_button, "_mtrax_guard", False):
        return
    _orig_dg = DeltaGenerator.download_button
    _orig_st = st.download_button

    def _log(res, args, kwargs):
        if res:
            name = kwargs.get("file_name") or (args[2] if len(args) > 2 else "")
            audit("unduh", file=name)
        return res

    def _dg_guard(self, *args, **kwargs):
        if not can_download():
            return False
        return _log(_orig_dg(self, *args, **kwargs), args, kwargs)

    def _st_guard(*args, **kwargs):
        if not can_download():
            return False
        return _log(_orig_st(*args, **kwargs), args, kwargs)

    _dg_guard._mtrax_guard = True
    _st_guard._mtrax_guard = True
    DeltaGenerator.download_button = _dg_guard
    st.download_button = _st_guard


_install_download_guard()


def _demo_audit(event: str):
    audit(f"demo_{event}", user="demo", role="Demo")


DEMO_MAX_FAILED = 5
DEMO_LOCK_SECONDS = 15 * 60


@st.cache_resource
def _demo_lock_store():
    return {"failed": 0, "locked_until": 0.0}


def demo_is_locked():
    return time.time() < _demo_lock_store()["locked_until"]


def demo_register_failure():
    store = _demo_lock_store()
    store["failed"] += 1
    if store["failed"] >= DEMO_MAX_FAILED:
        store["locked_until"] = time.time() + DEMO_LOCK_SECONDS
        store["failed"] = 0
        _demo_audit("locked")


def demo_register_success():
    _demo_lock_store()["failed"] = 0


def prepare_demo_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Data demo: 13 kolom, kode acak, nominal dikali faktor acak (tidak bisa dihitung balik)."""
    if _demo_config() is None or df is None or df.empty:
        return pd.DataFrame()
    out = mask_dataframe(df)
    if "Invoice Amount" in out.columns:
        rng = np.random.default_rng(secrets.randbits(64))
        factor = (0.45 + rng.random() * 0.4) * (1 + rng.uniform(-0.12, 0.12, len(out)))
        amount = pd.to_numeric(out["Invoice Amount"].astype(str).str.replace("$", "", regex=False)
                               .str.replace(",", "", regex=False).str.strip(), errors="coerce")
        out["Invoice Amount"] = np.round(amount.to_numpy(dtype=float) * factor, 0)
    return out


def _fx_monthly_table() -> pd.DataFrame:
    path = ""
    try:
        path = str(dict(st.secrets.get("fx", {})).get("file", "") or "")
    except Exception:
        path = ""
    path = path or os.environ.get("MTRAX_FX_FILE", "")
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        t = pd.read_excel(path) if path.endswith(("xlsx", "xls")) else pd.read_csv(path)
        t = t.rename(columns={c: str(c).strip().title() for c in t.columns})
        t["Bulan"] = t["Bulan"].astype(str).str[:7]
        t["Currency"] = t["Currency"].astype(str).str.upper().str.strip()
        t["Rate"] = pd.to_numeric(t["Rate"], errors="coerce")
        return t.dropna(subset=["Rate"])
    except Exception:
        return pd.DataFrame()


def convert_currency_to_idr(df: pd.DataFrame):
    """Invoice Amount -> Rupiah memakai kolom Currency.
    Prioritas kurs: tabel bulanan [fx] file (per bulan Issue Time) -> [fx.rates] tetap.
    Nilai asli disimpan di kolom 'Invoice Amount (Asli)'. Return (df, info)."""
    info = {"converted": 0, "missing": 0, "missing_currencies": []}
    if "Currency" not in df.columns or "Invoice Amount" not in df.columns:
        return df, info
    out = df.copy()
    cur = out["Currency"].astype(str).str.upper().str.strip().replace({"NAN": "IDR", "": "IDR", "RP": "IDR"})
    amount = pd.to_numeric(out["Invoice Amount"], errors="coerce")
    rate = pd.Series(np.where(cur == "IDR", 1.0, np.nan), index=out.index)

    monthly = _fx_monthly_table()
    if not monthly.empty and "Issue Time" in out.columns:
        month = pd.to_datetime(out["Issue Time"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m")
        key = pd.MultiIndex.from_arrays([month, cur])
        lut = monthly.set_index(["Bulan", "Currency"])["Rate"]
        lut = lut[~lut.index.duplicated()]
        rate = rate.fillna(pd.Series(lut.reindex(key).to_numpy(), index=out.index))
    try:
        fixed = {str(k).upper(): float(v) for k, v in dict(dict(st.secrets.get("fx", {})).get("rates", {})).items()}
    except Exception:
        fixed = {}
    if fixed:
        rate = rate.fillna(cur.map(fixed))

    out["Invoice Amount (Asli)"] = amount
    out["Invoice Amount"] = amount * rate
    non_idr = cur != "IDR"
    info["converted"] = int((non_idr & rate.notna()).sum())
    miss = non_idr & rate.isna() & amount.notna()
    info["missing"] = int(miss.sum())
    info["missing_currencies"] = sorted(cur[miss].unique().tolist())
    return out, info


# ── MODE TANPA LOGIN (sementara) ──
# Aktif bila [access] login_required = false. Semua pengunjung masuk sebagai akun "tamu"
# dengan role [access] guest_role (default Viewer). Semua unduhan tetap tercatat di audit log
# beserta alamat IP. Kembalikan ke true untuk mengaktifkan login + 2FA lagi.
GUEST_USERNAME = "tamu"
_VALID_ROLES = ("Super Admin", "Admin", "Analyst", "Viewer", "Demo")


def login_required() -> bool:
    try:
        return bool(dict(st.secrets.get("access", {})).get("login_required", True))
    except Exception:
        return True


def guest_role() -> str:
    try:
        r = str(dict(st.secrets.get("access", {})).get("guest_role", "Viewer"))
    except Exception:
        r = "Viewer"
    return r if r in _VALID_ROLES else "Viewer"


def guest_login():
    """Masuk otomatis tanpa password (hanya bila login_required = false)."""
    st.session_state.authenticated = True
    st.session_state.logged_in = True
    st.session_state.username = GUEST_USERNAME
    st.session_state.role = guest_role()
    st.session_state.login_time = time.time()
    st.session_state.pending_2fa = False
    if not st.session_state.get("_guest_logged"):
        st.session_state._guest_logged = True
        audit("masuk_tanpa_login")

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

def hash_password(password: str) -> str:
    """Hash password (Argon2id bila tersedia, fallback SHA-256)."""
    try:
        from argon2 import PasswordHasher
        return PasswordHasher().hash(password)
    except ImportError:
        return hashlib.sha256(password.encode()).hexdigest()


# Lima akun tetap. Role ditentukan oleh nama akun, bukan oleh isi secrets.
USER_ROLES = {
    "superadmin": "Super Admin",   # semua akses + audit log
    "admin": "Admin",              # semua akses kecuali audit log
    "ssc": "Analyst",              # lihat & unduh laporan
    "dtm": "Viewer",               # lihat & unduh laporan
    "demo": "Demo",                # lihat saja (nominal disamarkan), tanpa unduhan
}


def _load_users() -> dict:
    """
    Password tiap akun diambil dari secrets.toml:
        [users.superadmin]
        password_hash = "$argon2id$..."
    Untuk sementara, akun lama juga dibaca dari [auth] (admin_password / ssc_password / dtm_password).
    Akun demo hanya aktif bila password_hash-nya diisi dan [demo] enabled tidak false / belum kedaluwarsa.
    """
    try:
        cfg = {str(k): dict(v) for k, v in dict(st.secrets.get("users", {})).items()}
    except Exception:
        cfg = {}
    try:
        legacy = dict(st.secrets.get("auth", {}))
    except Exception:
        legacy = {}
    users = {}
    for name, role in USER_ROLES.items():
        pw = cfg.get(name, {}).get("password_hash") or legacy.get(f"{name}_password")
        if not pw:
            continue
        if role == "Demo" and _demo_config() is None:
            continue
        users[name] = {"password": str(pw), "role": role}
    return users


USERS = _load_users()


# ======================================
# BMKG FUNCTIONS
# ======================================
@st.cache_data(ttl=300, show_spinner=False)
def get_bmkg_realtime_quake():
    """Mengambil data gempa real-time dari BMKG (di-cache 5 menit — sebelumnya
    fetch HTTP baru dilakukan di SETIAP interaksi widget di seluruh aplikasi,
    karena fungsi ini dipanggil ulang tiap kali main_app() jalan)."""
    url = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "Infogempa" in data and "gempa" in data["Infogempa"]:
            return data["Infogempa"]["gempa"]

        return None

    except Exception:
        # Tidak lagi memanggil st.sidebar.error() di sini — pemanggilan elemen UI
        # dari dalam fungsi ber-cache bisa berperilaku tidak konsisten antara
        # cache-hit dan cache-miss. Kegagalan cukup direpresentasikan sebagai None;
        # tampilan sisi pemanggil sudah punya fallback "No recent data".
        return None

# ======================================
# NEWS TICKER
# ======================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_news(limit=10):
    """Sebelumnya melakukan 3x HTTP request ke Google News di SETIAP interaksi
    widget di seluruh aplikasi (karena dipanggil ulang tiap main_app() jalan,
    dan main_app() jalan ulang di setiap klik). Sekarang di-cache 5 menit —
    berita tidak perlu ambil ulang secepat itu."""
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
        border-top: 3px solid #1BA0E2;
    }
    .fa-title {
        font-size: 1.25em;
        font-weight: 700;
        color: #1BA0E2;
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
        background: #f0f8ff;
        border: 1px solid #b8d9f0;
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
        background: #1BA0E2;
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
        background: #f0f8ff; border: 1px solid #b8d9f0;
        border-radius: 20px; padding: 5px 16px;
        font-size: 0.78em; color: #1BA0E2;
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
    uri          = _provisioning_uri(secret, username)
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
  --accent: #1BA0E2;
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
  border-radius:10px;background:#1BA0E2;
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
.save-body .key{{color:#1BA0E2;}}
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
          <div class="st">Nama akun: <strong>MTRAX:{username}</strong>, tipe: <em>Time-based</em>.</div>
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
        border: 1.5px solid #cce4f4 !important;
        font-size: 1.4em !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.40em !important;
        text-align: center !important;
        padding: 14px 12px !important;
        background: #f0f8ff !important;
        color: #3a1a3a !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
        background: #ffffff !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: #d0b8d0 !important; letter-spacing: 0.30em !important; }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    /* Tombol Verifikasi: merah #FF4B4B */
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background-color: #FF3333 !important;
        border-color: #FF3333 !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] p,
    button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #FFFFFF !important; }
    div[data-testid="InputInstructions"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Dibungkus st.form: tekan Enter setelah mengetik OTP langsung memverifikasi & masuk.
    with st.form("setup_otp_form", clear_on_submit=False, border=False):
        otp_input = st.text_input(
            "Kode OTP dari Google Authenticator",
            placeholder="● ● ● ● ● ●",
            max_chars=6,
            key="setup_otp_field"
        )
        # Enter = tombol submit PERTAMA (Verifikasi), jadi Kembali diletakkan setelahnya.
        col_v, col_c = st.columns([3, 2])
        with col_v:
            _verify = st.form_submit_button("✅  Verifikasi & Masuk", use_container_width=True, type="primary")
        with col_c:
            _back = st.form_submit_button("← Kembali", use_container_width=True)

    if _back:
        st.session_state.pending_2fa  = False
        st.session_state.pending_user = ""
        st.rerun()
    elif _verify:
        if _totp_valid(secret, otp_input):
            st.session_state.totp_enrolled[username] = True
            _finish_login(username)
        else:
            st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")


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
  background: #eef6fc;
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
  color: #1BA0E2;
}}
.step-pill {{
  background: #e6f4fb; color: #1BA0E2;
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
  background: linear-gradient(135deg, #062440 0%, #0D7FCC 55%, #1BA0E2 100%);
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
  background: #f0f8ff;
  border: 1px solid #cce4f4;
  border-radius: 12px;
  padding: 12px 18px;
  margin-bottom: 20px;
  gap: 12px;
}}
.timer-label {{
  font-size: 0.76em;
  color: #6a8fa0;
  font-weight: 500;
}}
.timer-val {{
  font-family: 'DM Mono', monospace;
  font-size: 1.0em;
  font-weight: 600;
  color: #1BA0E2;
  min-width: 28px;
  text-align: right;
}}
.timer-bar-wrap {{
  flex: 1;
  height: 4px;
  background: #cce4f4;
  border-radius: 2px;
  overflow: hidden;
}}
.timer-bar {{
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, #1BA0E2, #75caf0);
  width: {progress_pct}%;
  transition: width 1s linear;
}}

/* ── Hint ──────────────────────────────────── */
.hint-box {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: #eef6fc;
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
.hint-text em {{ color: #1BA0E2; font-style: normal; font-weight: 500; }}

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
        border: 1.5px solid #cce4f4 !important;
        font-size: 1.3em !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.35em !important;
        text-align: center !important;
        padding: 14px !important;
        background: #f0f8ff !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(156,87,137,0.12) !important;
    }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
    /* Tombol Verifikasi: merah #FF4B4B */
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
        background-color: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background-color: #FF3333 !important;
        border-color: #FF3333 !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] p,
    button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #FFFFFF !important; }
    div[data-testid="InputInstructions"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Dibungkus st.form: tekan Enter setelah mengetik OTP langsung memverifikasi & masuk.
    with st.form("verify_otp_form", clear_on_submit=False, border=False):
        otp_input = st.text_input(
            "Kode OTP (6 digit)",
            placeholder="● ● ● ● ● ●",
            max_chars=6,
            key="verify_otp_field"
        )
        # Enter = tombol submit PERTAMA (Verifikasi), jadi Kembali diletakkan setelahnya.
        col_v, col_c = st.columns([3, 2])
        with col_v:
            _verify = st.form_submit_button("✅  Verifikasi & Masuk", use_container_width=True, type="primary")
        with col_c:
            _back = st.form_submit_button("← Kembali", use_container_width=True)

    if _back:
        st.session_state.pending_2fa  = False
        st.session_state.pending_user = ""
        st.rerun()
    elif _verify:
        if _totp_valid(secret, otp_input):
            _finish_login(username)
        else:
            st.error("❌ Kode salah atau sudah kedaluwarsa. Coba lagi.")
            time.sleep(1)


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
    audit("login", role=USERS[username]["role"])
    # Simpan token sesi di cookie browser supaya login tetap bertahan saat halaman di-reload (F5)
    _token = _create_session_token(username, USERS[username]["role"])
    st.session_state._session_token = _token
    _set_session_cookie(_token)
    st.success("✅ Login berhasil! Selamat datang.")
    # PENTING: `st.session_state.authenticated` sudah True di atas, jadi
    # `st.rerun()` di bawah ini SUDAH CUKUP untuk langsung masuk ke dashboard
    # dalam sesi yang sama — tidak bergantung pada cookie/JS sama sekali.
    # Jeda diberi lebih panjang (1 detik, sebelumnya 0.5 detik) hanya supaya
    # browser sempat memuat iframe & benar-benar menulis cookie SEBELUM
    # DOM diganti oleh hasil rerun — ini yang membuat sesi tetap bertahan
    # kalau nanti user benar-benar me-refresh browser (F5).
    time.sleep(1)
    st.rerun()


# ======================================
# LOGIN FUNCTION (SECURE + ANTI BRUTE FORCE)
# ── DIMODIFIKASI: setelah password OK → redirect ke 2FA ──
# ======================================
def login_page():
    st.set_page_config(
        page_title="Login | MTRAX",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    current_time = time.time()
    if current_time < st.session_state.lockout_until:
        remaining_lock = int(st.session_state.lockout_until - current_time)
        st.error(f"Too many failed attempts. Try again in {remaining_lock} seconds.")
        st.stop()

    # Load logo
    def _b64_img(p):
        try:
            with open(p, "rb") as f: return base64.b64encode(f.read()).decode()
        except Exception: return ""

    logo_b64 = _b64_img(os.path.join("assets", "LOGO M FIX.png"))
    logo_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        f'style="width:44px;display:block;margin:0 auto 16px;" />'
        if logo_b64 else
        '<div style="width:44px;height:44px;border-radius:9px;'
        'background:linear-gradient(135deg,#1BA0E2,#0D7FCC);'
        'display:flex;align-items:center;justify-content:center;margin:0 auto 16px;'
        'font-size:0.9em;font-weight:800;color:#ffffff;letter-spacing:.04em;">MTX</div>'
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── RESET & BASE ── */
    *, *::before, *::after { font-family:'Inter',sans-serif !important; box-sizing:border-box; margin:0; padding:0; }
    html, body { height:100%; overflow:hidden; }

    /* ── FIX: kembalikan font Material Symbols untuk ikon bawaan Streamlit
         (mis. tombol show/hide password) — tanpa ini, wildcard font-family di atas
         merusak ligature ikon sehingga muncul teks literal "visibility" alih-alih ikon mata ── */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-size: 1.2rem !important;
    }

    .stApp                              { background:#0D7FCC !important; overflow:hidden; }
    [data-testid="stHeader"]            { display:none !important; }
    [data-testid="stToolbar"]           { display:none !important; }
    [data-testid="stDecoration"]        { display:none !important; }
    [data-testid="stSidebar"]           { display:none !important; }
    [data-testid="stAppViewContainer"]  { padding:0 !important; overflow:hidden !important; }
    section[data-testid="stMain"]       { overflow:hidden !important; }
    #MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
    .block-container                    { padding:0 !important; max-width:100% !important; overflow:hidden !important; }
    ::-webkit-scrollbar                 { display:none !important; }

    /* ── HIDE DUMMY LEFT COL ── */
    [data-testid="stHorizontalBlock"]                        { gap:0 !important; overflow:hidden !important; }
    [data-testid="stHorizontalBlock"] > div                  { padding:0 !important; }
    [data-testid="stHorizontalBlock"] > div:first-child      { display:none !important; }

    /* ══════════════════════════════════════════════════════
       CANVAS — full screen gradient background
    ══════════════════════════════════════════════════════ */
    .lp-canvas {
        position: fixed; top:0; left:0;
        width: 100%; height: 100vh; z-index: 1;
        background: linear-gradient(135deg,
            #e2f871 0%,
            #c8ec50 12%,
            #8dd4a8 26%,
            #4ab8d4 38%,
            #1BA0E2 50%,
            #1494C6 65%,
            #0D7FCC 80%,
            #07395f 100%
        );
        overflow: hidden;
    }

    /* animated mesh radial blobs */
    .lp-canvas::before {
        content:'';
        position:absolute; inset:0;
        background:
            radial-gradient(ellipse at 8% 10%,  rgba(226,248,113,0.55) 0%, transparent 38%),
            radial-gradient(ellipse at 90% 5%,  rgba(13,127,204,0.45)  0%, transparent 40%),
            radial-gradient(ellipse at 75% 90%, rgba(20,148,198,0.40)  0%, transparent 45%),
            radial-gradient(ellipse at 15% 88%, rgba(226,248,113,0.30) 0%, transparent 35%);
        animation: lp-meshShift 12s ease-in-out infinite alternate;
    }
    @keyframes lp-meshShift {
        0%   { opacity:1; transform:scale(1) translateX(0); }
        100% { opacity:0.85; transform:scale(1.04) translateX(12px); }
    }

    /* dot grid overlay */
    .lp-canvas::after {
        content:''; position:absolute; inset:0; pointer-events:none;
        background-image: radial-gradient(circle, rgba(255,255,255,0.18) 1px, transparent 1px);
        background-size: 30px 30px;
        -webkit-mask-image: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.1) 60%, transparent 100%);
        mask-image: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.1) 60%, transparent 100%);
    }

    /* ── BLOBS ── */
    .lp-blob {
        position:absolute; border-radius:50%; pointer-events:none;
    }
    .lp-blob-1 {
        width:520px; height:520px; top:-160px; left:-140px;
        background: radial-gradient(circle, rgba(226,248,113,0.40) 0%, transparent 68%);
        animation: lp-floatA 8s ease-in-out infinite alternate;
    }
    .lp-blob-2 {
        width:480px; height:480px; top:30%; right:-100px;
        background: radial-gradient(circle, rgba(13,127,204,0.32) 0%, transparent 65%);
        animation: lp-floatB 10s ease-in-out infinite alternate;
    }
    .lp-blob-3 {
        width:380px; height:380px; bottom:-120px; left:20%;
        background: radial-gradient(circle, rgba(7,57,95,0.28) 0%, transparent 60%);
        animation: lp-floatA 9s ease-in-out infinite alternate-reverse;
    }
    .lp-blob-4 {
        width:260px; height:260px; top:-60px; right:18%;
        background: radial-gradient(circle, rgba(200,236,80,0.28) 0%, transparent 65%);
        animation: lp-floatB 7s ease-in-out infinite alternate;
    }
    @keyframes lp-floatA {
        0%   { transform:translate(0,0) scale(1); }
        100% { transform:translate(20px,30px) scale(1.06); }
    }
    @keyframes lp-floatB {
        0%   { transform:translate(0,0) scale(1); }
        100% { transform:translate(-15px,20px) scale(0.95); }
    }

    /* ── RINGS ── */
    .lp-ring {
        position:absolute; border-radius:50%; pointer-events:none;
    }
    .lp-ring-1 { width:600px; height:600px; top:-180px; left:-180px; border:1.5px solid rgba(255,255,255,0.12); }
    .lp-ring-2 { width:380px; height:380px; top:60px; left:60px; border:1px solid rgba(255,255,255,0.07); }
    .lp-ring-3 { width:320px; height:320px; bottom:-80px; right:10%; border:1px solid rgba(255,255,255,0.08); }
    .lp-ring-4 { width:180px; height:180px; top:15%; right:28%; border:1px solid rgba(226,248,113,0.25); }

    /* ── PARTICLES ── */
    .lp-particle { position:absolute; border-radius:50%; pointer-events:none; }
    .lp-p1 { width:5px; height:5px; background:#e2f871; top:12%; left:8%;
              box-shadow:0 0 14px rgba(226,248,113,0.8); animation:lp-pulse 3s ease-in-out infinite; }
    .lp-p2 { width:4px; height:4px; background:#fff; top:6%; right:12%;
              opacity:0.6; animation:lp-pulse 4s ease-in-out infinite 1s; }
    .lp-p3 { width:3px; height:3px; background:#e2f871; bottom:22%; left:18%;
              box-shadow:0 0 10px rgba(226,248,113,0.6); animation:lp-pulse 5s ease-in-out infinite 0.5s; }
    .lp-p4 { width:6px; height:6px; background:rgba(255,255,255,0.5); bottom:35%; right:18%;
              animation:lp-pulse 3.5s ease-in-out infinite 2s; }
    .lp-p5 { width:3px; height:3px; background:#1BA0E2; top:45%; left:38%;
              box-shadow:0 0 10px rgba(27,160,226,0.7); animation:lp-pulse 4s ease-in-out infinite 1.5s; }
    @keyframes lp-pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%     { opacity:0.35; transform:scale(1.5); }
    }

    /* ── LEFT CONTENT: brand + hero ── */
    .lp-inner {
        position:fixed; z-index:10;
        top:0; left:0; bottom:0; width:50%;
        display:flex; flex-direction:column;
        justify-content:space-between;
        padding:40px 52px 48px;
    }
    .lp-brand { display:flex; align-items:center; gap:12px; }
    .lp-brand-icon {
        width:36px; height:36px; border-radius:8px;
        background:rgba(255,255,255,0.18);
        border:1px solid rgba(255,255,255,0.28);
        display:flex; align-items:center; justify-content:center;
    }
    .lp-brand-icon-sq {
        display:block; width:10px; height:10px;
        background:white; border-radius:1px; opacity:0.95;
    }
    .lp-brand-name { font-size:0.80em; font-weight:800; color:#fff; letter-spacing:0.20em;
                     text-shadow:0 1px 8px rgba(0,0,0,0.18); }
    .lp-divider {
        width:34px; height:3px;
        background:linear-gradient(90deg, #e2f871 0%, rgba(226,248,113,0.30) 100%);
        border-radius:2px; margin-bottom:18px;
        box-shadow:0 0 12px rgba(226,248,113,0.55);
    }
    .lp-headline {
        font-size:2.40em; font-weight:900; color:#fff;
        line-height:1.15; letter-spacing:-0.02em; margin-bottom:16px;
        text-shadow:0 2px 20px rgba(0,0,0,0.20);
    }
    .lp-sub { font-size:0.80em; color:rgba(255,255,255,0.60); font-weight:400; line-height:1.7; }

    /* ── COPYRIGHT ── */
    .lp-copy {
        position:fixed; bottom:20px; right:28px; z-index:10;
        font-size:0.60em; color:rgba(255,255,255,0.40);
        font-weight:500; letter-spacing:0.04em;
    }
    .lp-copy a { color:inherit; text-decoration:none; }
    .lp-copy a:hover { color:rgba(226,248,113,0.70); }

    /* ── FLOATING LOGIN CARD — glass white ── */
    [data-testid="stHorizontalBlock"] > div:last-child {
        position: fixed !important;
        top: 50% !important;
        left: 72% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1000 !important;
        width: 340px !important;
        max-width: 340px !important;
        background: rgba(255,255,255,0.96) !important;
        border-radius: 20px !important;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.30),
            0 8px 32px rgba(7,57,95,0.22),
            0 32px 80px rgba(7,57,95,0.16) !important;
        padding: 40px 36px 36px !important;
        border: none !important;
        overflow: visible !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }

    /* shimmer top border on card */
    [data-testid="stHorizontalBlock"] > div:last-child::before {
        content:'';
        position:absolute; top:0; left:20px; right:20px; height:1px;
        background:linear-gradient(90deg,
            transparent 0%, rgba(226,248,113,0.60) 30%,
            rgba(27,160,226,0.40) 70%, transparent 100%);
        border-radius:1px;
    }

    /* ── INPUT FIELDS ── */
    .stTextInput label { display:none !important; }
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #dde5ee !important;
        background: #f7fafd !important;
        padding: 12px 14px !important;
        font-size: 0.86em !important;
        color: #1a2a3a !important;
        transition: border-color .2s, box-shadow .2s !important;
    }
    .stTextInput > div > div > input::placeholder { color: #b0bec8 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #1BA0E2 !important;
        box-shadow: 0 0 0 3px rgba(27,160,226,0.13) !important;
        background: #ffffff !important;
        outline: none !important;
    }
    .stTextInput > div > div > div > button {
        background: transparent !important; border: none !important; color: #b0bec8 !important;
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
        min-width: 32px !important; height: 32px !important; padding: 0 !important;
    }
    .stTextInput > div > div > div > button:hover { color: #0D7FCC !important; }

    /* ── LOG IN BUTTON — lime gradient ── */
    /* (mencakup stButton dan stFormSubmitButton — form dipakai agar tombol Enter bisa submit login) */
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #e2f871 0%, #cce84a 50%, #b8d930 100%) !important;
        color: #07395f !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 13px !important;
        font-weight: 900 !important;
        font-size: 0.78em !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all .18s ease !important;
        margin-top: 6px !important;
        box-shadow: 0 4px 18px rgba(226,248,113,0.55), 0 1px 4px rgba(7,57,95,0.12) !important;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #d4e860 0%, #bdd938 50%, #a8c820 100%) !important;
        box-shadow: 0 6px 24px rgba(226,248,113,0.70), 0 2px 8px rgba(7,57,95,0.15) !important;
        transform: translateY(-1.5px) !important;
    }
    div[data-testid="stButton"] > button:active,
    div[data-testid="stFormSubmitButton"] > button:active { transform: translateY(0) !important; }

    /* Streamlit's st.form menambahkan border bawaan di sekeliling form — dihilangkan
       supaya tetap menyatu dengan desain kartu login yang sudah ada */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    .stAlert { border-radius: 8px !important; margin-top: 8px !important; font-size: 0.78em !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── RENDER BACKGROUND CANVAS
    st.markdown("""
    <div class="lp-canvas">
        <div class="lp-blob lp-blob-1"></div>
        <div class="lp-blob lp-blob-2"></div>
        <div class="lp-blob lp-blob-3"></div>
        <div class="lp-blob lp-blob-4"></div>
        <div class="lp-ring lp-ring-1"></div>
        <div class="lp-ring lp-ring-2"></div>
        <div class="lp-ring lp-ring-3"></div>
        <div class="lp-ring lp-ring-4"></div>
        <div class="lp-particle lp-p1"></div>
        <div class="lp-particle lp-p2"></div>
        <div class="lp-particle lp-p3"></div>
        <div class="lp-particle lp-p4"></div>
        <div class="lp-particle lp-p5"></div>
    </div>
    <div class="lp-inner">
        <div class="lp-brand">
            <div class="lp-brand-icon"><span class="lp-brand-icon-sq"></span></div>
            <span class="lp-brand-name">MTRAX</span>
        </div>
        <div>
            <div class="lp-divider"></div>
            <div class="lp-headline">Welcome to<br><b>MTRAX</b><br>Analytics</div>
            <div class="lp-sub">Corporate Travel Analytics Platform<br>for Business Travel</div>
        </div>
    </div>
    <div class="lp-copy"><a href="https://www.linkedin.com/in/rifyalt/" target="_blank">© 2026 MTRAX.</a> All rights reserved.</div>
    """, unsafe_allow_html=True)

    # ── FLOATING CARD: dummy col (hidden) + form col
    _dummy, col_form = st.columns(2)
    with col_form:
        st.markdown(f"""
        {logo_tag}
        <div style="text-align:center;font-size:1.15em;font-weight:800;color:#1a1a1a;
             letter-spacing:0.01em;margin-bottom:5px;">Welcome</div>
        <div style="text-align:center;font-size:0.70em;color:#666;margin-bottom:28px;
             font-weight:400;letter-spacing:0.02em;">Log in to your account</div>
        <div style="font-size:0.70em;font-weight:700;color:#07395f;
             letter-spacing:0.10em;text-transform:uppercase;margin-bottom:6px;">Username</div>
        """, unsafe_allow_html=True)

        # Dibungkus st.form supaya menekan Enter di kolom Username/Password
        # langsung men-submit login (form_submit_button otomatis ter-trigger oleh Enter),
        # tanpa harus mengklik tombol LOG IN secara manual.
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("_u", placeholder="Enter Username ...",
                                     label_visibility="collapsed", key="login_username")

            st.markdown("""
            <div style="font-size:0.70em;font-weight:700;color:#07395f;
                 letter-spacing:0.10em;text-transform:uppercase;margin:14px 0 6px;">Password</div>
            """, unsafe_allow_html=True)

            password = st.text_input("_p", type="password", placeholder="Enter Password ...",
                                     label_visibility="collapsed", key="login_password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("LOG IN", use_container_width=True)

        if submitted:
            if not validate_username(username):
                st.error("Invalid username format.")
                return
            _wait = _user_locked(username)
            if _wait > 0:
                st.error(f"Akun dikunci sementara. Coba lagi dalam {int(_wait)} detik.")
                return
            _is_demo_login = username in USERS and USERS[username]["role"] == "Demo"
            if _is_demo_login and demo_is_locked():
                st.error("Akun demo sedang dikunci sementara. Coba lagi nanti.")
                return
            if username in USERS:
                if secure_compare(password, USERS[username]["password"]):
                    _user_ok(username)
                    st.session_state.login_attempts = 0
                    st.session_state.lockout_until  = 0
                    if _is_demo_login:
                        demo_register_success()
                    # Akun demo: 2FA hanya jika punya secret sendiri ([totp][demo]).
                    if _is_demo_login and not _own_totp_secret(username):
                        _demo_audit("login")
                        _finish_login(username)
                    else:
                        st.session_state.pending_2fa    = True
                        st.session_state.pending_user   = username
                        st.session_state.pending_role   = USERS[username]["role"]
                        st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    _user_fail(username)
                    audit("login_gagal", user=username)
                    if _is_demo_login:
                        demo_register_failure()
                        _demo_audit("failed login")
            else:
                st.session_state.login_attempts += 1

            if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                st.session_state.lockout_until  = time.time() + LOCKOUT_SECONDS
                st.session_state.login_attempts = 0
                st.error("Too many failed attempts. Account temporarily locked.")
            else:
                remaining_attempts = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"Login failed. {remaining_attempts} attempt(s) remaining.")

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

    # ── BUG FIX: sliding expiry di server (_validate_session_token) memperpanjang
    # masa berlaku token, tapi cookie di BROWSER tetap memakai max-age lama sejak
    # login pertama. Kalau tidak disegarkan, cookie browser akan tetap kedaluwarsa
    # sesuai waktu login awal walau token server masih valid -> user bisa
    # ter-logout otomatis meski masih aktif memakai aplikasi. Segarkan cookie
    # (tanpa reload) setiap kali main_app dimuat, selama masih ada token sesi.
    if st.session_state.get("_session_token"):
        _set_session_cookie(st.session_state._session_token)

    # Custom CSS - MTRAX Theme v2 (Blue/Grey Palette)
    st.markdown("""
        <style>
        /* ============================================================
           MTRAX DESIGN SYSTEM
           Primary:  #1BA0E2  #1494C6  #0D7FCC
           Neutral:  #f0f0f0  #dedede  #434343
           Accent:   #98ea16  #d61b54  #ff5e1f  #ffdc00  #e2f871  #e13230
        ============================================================ */

        :root {
            --clr-primary:      #1BA0E2;
            --clr-primary-mid:  #1494C6;
            --clr-primary-dark: #0D7FCC;
            --clr-bg:           #f0f0f0;
            --clr-surface:      #ffffff;
            --clr-border:       #dedede;
            --clr-text-main:    #434343;
            --clr-text-muted:   #7a8a96;
            --clr-accent-green: #98ea16;
            --clr-accent-pink:  #d61b54;
            --clr-accent-orange:#ff5e1f;
            --clr-accent-yellow:#ffdc00;
            --clr-accent-lime:  #e2f871;
            --clr-accent-red:   #e13230;
        }

        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', sans-serif;
        }

        /* ── Proteksi tambahan: pastikan ikon Material Symbols bawaan Streamlit
             (selectbox, multiselect, date picker, expander, dsb.) tidak ikut
             ter-override font-family-nya oleh rule '*' di atas ── */
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: 'liga' !important;
            font-feature-settings: 'liga' !important;
            -webkit-font-smoothing: antialiased !important;
        }

        .stApp {
            background: var(--clr-bg);
        }

        /* Sembunyikan hanya header bawaan Streamlit (bukan semua tag <header>,
           agar judul bulan/tahun di kalender date picker tetap tampil) */
        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
        div[data-baseweb="calendar"],
        div[data-baseweb="calendar"] * { visibility: visible !important; }
        /* Tombol untuk membuka kembali sidebar ada di dalam header, jadi tetap ditampilkan */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stSidebarCollapsedControl"] *,
        [data-testid="collapsedControl"] * { visibility: visible !important; }

        /* Cegah tampilan tergeser ke samping: wadah utama (sidebar + konten) tidak boleh
           ikut tergulir horizontal saat isi konten sedikit lebih lebar dari layar
           (gejala: sidebar terpotong di kiri & konten terpotong di kanan). */
        .stApp, [data-testid="stAppViewContainer"] { overflow: clip !important; }
        section[data-testid="stMain"], section.main { min-width: 0 !important; }
        [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] { overflow-x: hidden !important; }

        /* ── NEWS TICKER ─────────────────────────────── */
        .news-ticker {
            background: linear-gradient(90deg, #d71c58 0%, #d71c58 100%);
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

        /* ── HEADER ──────────────────────────────────── */
        .header {
            background: var(--clr-surface);
            padding: 25px 40px;
            border-bottom: 2px solid var(--clr-primary);
            margin: 0 -60px 30px -60px;
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--clr-primary-dark);
            letter-spacing: -0.02em;
        }

        .header-subtitle {
            font-size: 0.9em;
            color: var(--clr-text-muted);
            margin-top: 4px;
        }

        .header-user { text-align: right; }

        .user-name {
            font-size: 0.95em;
            color: var(--clr-text-main);
            font-weight: 500;
        }

        .user-role {
            font-size: 0.85em;
            color: var(--clr-text-muted);
            margin-top: 2px;
        }

        /* ── GLOBAL FILTER BAR ───────────────────────── */
        .global-filter-bar {
            background: var(--clr-surface);
            border: 1px solid var(--clr-border);
            border-left: 4px solid var(--clr-primary);
            border-radius: 6px;
            padding: 14px 20px 10px 20px;
            margin-bottom: 20px;
        }
        .global-filter-label {
            font-size: 0.72em;
            font-weight: 700;
            color: var(--clr-primary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }

        /* ── METRIC BOX ──────────────────────────────── */
        .metric-box {
            background: var(--clr-surface);
            padding: 20px;
            border-radius: 6px;
            border-left: 3px solid var(--clr-primary);
        }

        .metric-label {
            font-size: 0.8em;
            color: var(--clr-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: 600;
            color: var(--clr-text-main);
        }

        /* ── STATS CARDS ─────────────────────────────── */
        .stats-card {
            background: var(--clr-surface);
            padding: 25px;
            border-radius: 6px;
            text-align: center;
            height: 100%;
        }

        .stats-number {
            font-size: 2.5em;
            font-weight: 700;
            color: var(--clr-primary);
            margin: 15px 0;
        }

        .stats-label {
            font-size: 0.85em;
            color: var(--clr-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stats-detail {
            font-size: 0.9em;
            color: var(--clr-text-muted);
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--clr-bg);
        }

        /* ── SECTION TITLE ───────────────────────────── */
        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            color: var(--clr-text-main);
            margin: 30px 0 20px 0;
        }

        /* ── TABS ────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--clr-surface);
            border-bottom: 2px solid var(--clr-border);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            font-weight: 500;
            color: var(--clr-text-muted);
            border-bottom: 2px solid transparent;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--clr-primary);
        }

        .stTabs [aria-selected="true"] {
            color: var(--clr-primary);
            border-bottom-color: var(--clr-primary);
            background: transparent;
        }

        /* ── BUTTONS ─────────────────────────────────── */
        .stButton > button {
            border-radius: 4px;
            font-weight: 500;
            padding: 10px 20px;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1BA0E2 0%, #0D7FCC 100%);
            color: white;
            border: none;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #a6ce39 !important;  # ← ganti warna sesuai keinginan
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1494C6 0%, #0D7FCC 100%);
        }

        /* ── SIDEBAR ─────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: var(--clr-surface);
            border-right: 1px solid var(--clr-border);
        }

        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--clr-primary-dark);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ── STREAMLIT METRIC ────────────────────────── */
        .stMetric {
            background: var(--clr-surface);
            padding: 18px;
            border-radius: 6px;
            border-left: 3px solid var(--clr-primary);
        }

        .stMetric label {
            color: var(--clr-text-muted) !important;
            font-size: 0.8em !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: var(--clr-text-main) !important;
            font-size: 1.8em !important;
        }

        /* ── PROGRESS ────────────────────────────────── */
        .stProgress > div > div {
            background: linear-gradient(90deg, #1BA0E2, #0D7FCC);
        }

        /* ── DIVIDER ─────────────────────────────────── */
        .divider {
            height: 1px;
            background: var(--clr-border);
            margin: 30px 0;
        }

        /* ── SELECTBOX ───────────────────────────────── */
        .stSelectbox > div > div {
            border-radius: 4px;
        }

        /* ── ACCENT BADGE HELPERS ────────────────────── */
        .badge-green  { background:#98ea16; color:#1e4000; padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-red    { background:#e13230; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-orange { background:#ff5e1f; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-yellow { background:#ffdc00; color:#433000; padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-pink   { background:#d61b54; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
        .badge-blue   { background:#1BA0E2; color:#fff;    padding:2px 8px; border-radius:20px; font-size:0.75em; font-weight:600; }
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
    # SIDEBAR MODE STATE
    # ======================================
    if "sidebar_mode" not in st.session_state:
        st.session_state.sidebar_mode = "expanded"

    # ======================================
    # SIDEBAR
    # ======================================
    def load_logo_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    logo_base64 = load_logo_base64("assets/LOGO M FIX.png")
    # ===== LOGO =====
#    if logo_base64:
#        logo_size = 60 if st.session_state.sidebar_mode == "expanded" else 35

#        st.markdown(f"""
#            <div style='text-align:center;padding:10px 0 15px 0;'>
#                <img src='data:image/png;base64,{logo_base64}' width='{logo_size}'/>
#            </div>
#        """, unsafe_allow_html=True)

    # ======================================
    # SIDEBAR WIDTH CONTROL (DYNAMIC)
    # ======================================

    sidebar_width = 260 if st.session_state.sidebar_mode == "expanded" else 90

    st.markdown(f"""
    <style>

    /* Sidebar width: lebar dikunci HANYA di mode compact. Di mode expanded lebar
       diatur Streamlit sendiri — memaksa width/min-width !important membuat
       sidebar tidak bisa collapse/expand dengan benar (terpotong & tombol buka hilang). */
    {f"section[data-testid='stSidebar'] {{ width:{sidebar_width}px !important; min-width:{sidebar_width}px !important; max-width:{sidebar_width}px !important; }}" if st.session_state.sidebar_mode == "compact" else ""}

    /* Hide labels in compact mode */
    {"section[data-testid='stSidebar'] label { display:none !important; }" if st.session_state.sidebar_mode == "compact" else ""}

    /* Hide select text in compact mode */
    {"section[data-testid='stSidebar'] .stSelectbox div div div { font-size:0 !important; }" if st.session_state.sidebar_mode == "compact" else ""}

    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # ===============================
        # TOGGLE BUTTON
        # ===============================
#        col1, col2 = st.columns([1,1])

#        with col1:
#            if st.button("☰", use_container_width=True):
#                if st.session_state.sidebar_mode == "expanded":
#                    st.session_state.sidebar_mode = "compact"
#                else:
#                    st.session_state.sidebar_mode = "expanded"
#                st.rerun()

#        with col2:
#            if st.session_state.sidebar_mode == "expanded":
#                st.markdown("<div style='font-size:12px;color:gray;padding-top:8px;'>Menu</div>", unsafe_allow_html=True)

#        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)


        st.markdown(f"""
            <div style='text-align:center;padding:20px 0;'>
                <img src='data:image/png;base64,{logo_base64}' width='60'/>
            </div>
        """, unsafe_allow_html=True)

        if not login_required():
            st.warning(f"🔓 Mode tanpa login (sementara). Semua pengunjung masuk sebagai "
                       f"'{GUEST_USERNAME}' dengan role {guest_role()}. Aktifkan lagi login di "
                       f"secrets.toml: [access] login_required = true.")

        # ── 2FA Status Badge di Sidebar ──────────────────────────────
        _uname_side   = st.session_state.get("username", "")
        _enrolled     = st.session_state.totp_enrolled.get(_uname_side, False)
        # BUG FIX: secret TOTP bersifat shared milik 'admin' (lihat _load_totp_secrets),
        # bukan per-username. Sebelumnya kode di sini mengecek
        # st.secrets["totp"][_uname_side] — hampir selalu False untuk user selain
        # 'admin' walau 2FA sebenarnya sudah aktif lewat secret bersama, sehingga
        # badge salah menampilkan "Setup Required" padahal 2FA sudah aktif.
        _from_toml = bool(_own_totp_secret(_uname_side))
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

        if is_demo_user():
            # Akun demo: data riil yang disamarkan, periode terbatas, tanpa upload/refresh.
            st.info("🧪 Mode demo — nomor transaksi & nominal disamarkan. Download dinonaktifkan.")
            _dcfg = _demo_config() or {"periods": []}
            _all_folders = dict(st.secrets["drive_folders"])
            _demo_periods = [p for p in _all_folders if not _dcfg["periods"] or p in _dcfg["periods"]]
            if not _demo_periods:
                st.warning("Tidak ada periode yang diizinkan untuk demo.")
            else:
                _demo_period = st.selectbox("Select Data Period", options=_demo_periods)
                if st.button("Get Data", use_container_width=True, type="primary"):
                    with st.spinner(f"Loading {_demo_period} data..."):
                        try:
                            _raw = load_drive_data(_all_folders[_demo_period],
                                                   ["Unnamed: 0", "Travel Request Number.1"])
                            st.session_state.df_all = prepare_demo_dataset(_raw)
                            del _raw
                            st.session_state.data_loaded = True
                            st.session_state.data_period = _demo_period
                            st.session_state._demo_masked = True
                            st.session_state._id_masked = False
                            st.session_state._data_ver = st.session_state.get("_data_ver", 0) + 1
                            _demo_audit(f"loaded period={_demo_period}")
                            st.rerun()
                        except Exception:
                            st.error("Gagal memuat data demo.")
        else:
            # Drive options
            drive_options = dict(st.secrets["drive_folders"])  # ID folder disimpan di Secrets, bukan di kode
        
            selected_period = st.selectbox(
                "Select Data Period",
                options=list(drive_options.keys()),
                help="Choose which year's data to load"
            )
        
            _btn_col1, _btn_col2 = st.columns([2, 1])
            with _btn_col1:
                _get_data_clicked = st.button("Get Data", use_container_width=True, type="primary")
            with _btn_col2:
                _refresh_clicked = st.button("🔄", use_container_width=True,
                                              help="Bersihkan cache & ambil ulang data terbaru dari Drive (pakai ini kalau ada file baru yang baru diupload ke folder Drive)")

            if _refresh_clicked:
                load_drive_data.clear()  # hapus cache lama supaya file baru di folder Drive ikut terbaca
                st.toast("Cache dibersihkan. Klik 'Get Data' lagi untuk ambil data terbaru.", icon="🔄")

            if _get_data_clicked:
                with st.spinner(f"Loading {selected_period} data..."):
                    progress_bar = st.progress(0)
                    try:
                        folder_id = drive_options[selected_period]
                        drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                    
                        progress_bar.progress(20)
                        loaded_data = load_drive_data(folder_id, drop_cols)
                    
                        progress_bar.progress(100)
                        st.session_state.df_all = loaded_data
                        st.session_state._id_masked = False
                        st.session_state._data_ver = st.session_state.get("_data_ver", 0) + 1
                        st.session_state.data_loaded = True
                        st.session_state.data_period = selected_period
                    
                        st.success(f"✅ Loaded {len(loaded_data):,} records")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            uploaded_files = st.file_uploader(
                "Upload Excel Files (Admin)",
                type=["xlsx", "xls"],
                accept_multiple_files=True,
                help="Hanya 13 kolom transaksi hotel yang dipakai; kolom lain otomatis dibuang."
            ) if is_admin() else None

            if uploaded_files:
                if st.button("Process Uploaded Files", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            dfs = []
                            drop_cols = ["Unnamed: 0", "Travel Request Number.1"]
                        
                            for file in uploaded_files:
                                df = pd.read_excel(file)
                                df = select_columns(df)[0]
                                dfs.append(trim_string_columns(df))

                            st.session_state.df_all = mask_dataframe(pd.concat(dfs, ignore_index=True))
                            audit("upload_manual", files=len(uploaded_files))
                            st.session_state._id_masked = False
                            st.session_state._data_ver = st.session_state.get("_data_ver", 0) + 1
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
            _fx = st.session_state.get("_fx_info") or {}
            if _fx.get("missing"):
                st.warning(f"{_fx['missing']:,} booking dengan mata uang tanpa kurs "
                           f"({', '.join(_fx.get('missing_currencies', []))}). Nilainya dikosongkan.")
        else:
            st.info("ℹ️ No data loaded")
        st.caption("Sumber: 13 kolom transaksi hotel tanpa identitas pekerja. No. TR & Booking ID "
                   "berupa kode acak yang berganti setiap data ditarik.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # BMKG Info
        st.markdown("### 🌏 BMKG INFO")
        
        quake_data = get_bmkg_realtime_quake()
        
        if quake_data:
            st.markdown(f"""
                <div style='background:white;padding:12px;border-radius:4px;border-left:3px solid #1BA0E2;'>
                    <div style='font-weight:600;color:#1494C6;font-size:0.85em;'>Latest Earthquake</div>
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
            _revoke_session_token(st.session_state.get("_session_token"))
            _clear_session_cookie()
            time.sleep(0.5)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ======================================
    # MAIN CONTENT
    # ======================================
    df_all = st.session_state.df_all

    # Penjaga: sesi demo hanya boleh memegang data yang sudah disamarkan.
    if is_demo_user() and not st.session_state.get("_demo_masked"):
        st.session_state.df_all = pd.DataFrame()
        st.session_state.data_loaded = False
        df_all = st.session_state.df_all
    if is_demo_user():
        # Sembunyikan tombol unduh/fullscreen bawaan tabel (lapisan tambahan, bukan kontrol utama)
        st.markdown("<style>[data-testid='stElementToolbar'],[data-testid='stDownloadButton'],"
                    ".stDownloadButton,.modebar,.modebar-container{display:none !important;}</style>",
                    unsafe_allow_html=True)

    if not df_all.empty:
        # Helper functions
        def prepare_monthly_trend(df, date_col, value_col, agg="count"):
            if date_col not in df.columns:
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce", dayfirst=True)
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
        # ======================================
        # PEMROSESAN DATA SEKALI PER PEMUATAN
        # Pembersihan, penyeragaman nama hotel, normalisasi kota, mapping
        # perusahaan & peta penggabungan properti hanya dijalankan saat data
        # baru dimuat (atau user berganti). Hasilnya disimpan di session,
        # jadi klik filter/dropdown tidak lagi mengulang proses berat ini.
        # ======================================
        _prep_key = (st.session_state.get("_data_ver", 0), st.session_state.get("username", ""))
        _prep = st.session_state.get("_prep")
        if _prep is not None and _prep.get("key") == _prep_key:
            df_all = st.session_state.df_all
            hotel_mapping = _prep["hotel_mapping"]
            property_merge_map = _prep["property_merge_map"]
        else:
            # Bersihkan Invoice Amount lalu konversi ke Rupiah (kolom Currency)
            if "Invoice Amount" in df_all.columns:
                df_all["Invoice Amount"] = pd.to_numeric(
                    df_all["Invoice Amount"].astype(str).str.replace("$", "").str.replace(",", ""),
                    errors="coerce"
                )
                df_all, st.session_state["_fx_info"] = convert_currency_to_idr(df_all)
        
            # Bersihkan Number of Rooms Night
            if "Number of Rooms Night" in df_all.columns:
                df_all["Number of Rooms Night"] = pd.to_numeric(df_all["Number of Rooms Night"], errors="coerce")
        
            # Convert dates
            if "Check in Date" in df_all.columns:
                df_all["Check in Date"] = pd.to_datetime(df_all["Check in Date"], errors="coerce", dayfirst=True)
        
            if "Check out Date" in df_all.columns:
                df_all["Check out Date"] = pd.to_datetime(df_all["Check out Date"], errors="coerce", dayfirst=True)
        
            if "Issue Time" in df_all.columns:
                df_all["Issue Time"] = pd.to_datetime(df_all["Issue Time"], errors="coerce", dayfirst=True)


            # ======================================
            # PENYERAGAMAN NAMA HOTEL & SUPPLIER
            # Aturan merek (Meliá, Mövenpick, éL Hotel, Marriott) + kamus standar
            # data/unique_hotels.xlsx. Idempoten, aman dijalankan setiap rerun.
            # ======================================
            try:
                df_all = standardize_hotel_columns(df_all)
                st.session_state.df_all = df_all
            except Exception as _e_std:
                st.warning(f"Penyeragaman nama hotel dilewati: {_e_std}")

            # ======================================
            # AUTO-NORMALISASI
            # ======================================
            hotel_mapping = pd.DataFrame()  # default agar tidak NameError jika kolom "Hotel Name" tidak ada
            if "Hotel Name" in df_all.columns:
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
            # PETA PENGGABUNGAN PROPERTI HOTEL
            # Dihitung dari data penuh (setelah standardisasi City & masking demo,
            # sebelum filter global) agar pengelompokan stabil. Dipakai di tab
            # Price Intelligence & Room Intelligence.
            # ======================================
            property_merge_map = get_property_merge_map(df_all, hotel_col="Hotel Name", city_col="City")

            st.session_state.df_all = df_all
            st.session_state._prep = {"key": _prep_key, "hotel_mapping": hotel_mapping,
                                      "property_merge_map": property_merge_map}

        # ======================================
        # GLOBAL FILTERS
        # ======================================
        st.markdown("""
            <div class='global-filter-label'>🔍 Global Filters — berlaku di seluruh tab</div>
        """, unsafe_allow_html=True)

        gcol1, gcol2, gcol3 = st.columns([2, 1.5, 1.5])

        selected_companies = []  # kolom perusahaan tidak ada di sumber data
        with gcol1:
            st.caption("Rentang tanggal di samping berlaku di seluruh tab. Nilai uang dalam Rupiah "
                       "(invoice non-Rupiah sudah dikonversi).")

        # Tentukan kolom tanggal untuk filter (From / To)
        _date_col_for_filter = None
        for _candidate_col in ["Issue Time", "Check in Date", "Check out Date"]:
            if _candidate_col in df_all.columns:
                _date_col_for_filter = _candidate_col
                break

        if _date_col_for_filter:
            _valid_dates = pd.to_datetime(df_all[_date_col_for_filter], errors="coerce", dayfirst=True).dropna()
            _min_date = _valid_dates.min().date() if not _valid_dates.empty else None
            _max_date = _valid_dates.max().date() if not _valid_dates.empty else None
        else:
            _min_date = _max_date = None

        # Inisialisasi nilai default di session_state agar tidak reset tiap rerun
        if _min_date and "gf_date_from_init" not in st.session_state:
            st.session_state["gf_date_from_init"] = _min_date
        if _max_date and "gf_date_to_init" not in st.session_state:
            st.session_state["gf_date_to_init"] = _max_date

        # Reset init jika data baru dimuat (periode berubah)
        _current_period = st.session_state.get("data_period", "")
        if st.session_state.get("_last_data_period") != _current_period:
            st.session_state["_last_data_period"] = _current_period
            if _min_date:
                st.session_state["gf_date_from_init"] = _min_date
            if _max_date:
                st.session_state["gf_date_to_init"] = _max_date

        with gcol2:
            if _min_date:
                # Gunakan nilai dari session_state sebagai default, bukan langsung _min_date
                _default_from = st.session_state.get("gf_date_from_init", _min_date)
                # Pastikan default_from masih dalam range yang valid
                if _default_from < _min_date:
                    _default_from = _min_date
                if _default_from > _max_date:
                    _default_from = _min_date
                filter_date_from = st.date_input(
                    "📅 From",
                    value=_default_from,
                    min_value=_min_date,
                    max_value=_max_date,
                    key="gf_date_from",
                    format="YYYY/MM/DD"
                )
                # Simpan pilihan user ke session_state agar persisten
                st.session_state["gf_date_from_init"] = filter_date_from
            else:
                filter_date_from = None
                st.info("Kolom tanggal tidak ditemukan.")

        with gcol3:
            if _max_date:
                _default_to = st.session_state.get("gf_date_to_init", _max_date)
                # Pastikan default_to masih dalam range yang valid
                if _default_to > _max_date:
                    _default_to = _max_date
                if _default_to < _min_date:
                    _default_to = _max_date
                filter_date_to = st.date_input(
                    "📅 To",
                    value=_default_to,
                    min_value=_min_date,
                    max_value=_max_date,
                    key="gf_date_to",
                    format="YYYY/MM/DD"
                )
                # Simpan pilihan user ke session_state agar persisten
                st.session_state["gf_date_to_init"] = filter_date_to
            else:
                filter_date_to = None

        # Validasi: pastikan from <= to
        if filter_date_from and filter_date_to and filter_date_from > filter_date_to:
            st.warning("⚠️ Tanggal 'From' tidak boleh lebih besar dari 'To'. Filter tanggal diabaikan.")
            filter_date_from = _min_date
            filter_date_to = _max_date

        # Terapkan filter ke df_all
        df_filtered = df_all.copy()

        if selected_companies:
            df_filtered = df_filtered[df_filtered["Nama Perusahaan"].isin(selected_companies)]

        if _date_col_for_filter and filter_date_from and filter_date_to:
            _date_series = pd.to_datetime(df_filtered[_date_col_for_filter], errors="coerce", dayfirst=True)
            df_filtered = df_filtered[
                (_date_series.dt.date >= filter_date_from) &
                (_date_series.dt.date <= filter_date_to)
            ]

        # Tampilkan info jumlah data setelah filter
        _total = len(df_all)
        _filtered = len(df_filtered)
        _pct = (_filtered / _total * 100) if _total > 0 else 0
        _filter_active = bool(selected_companies) or (filter_date_from != _min_date or filter_date_to != _max_date) if _min_date else bool(selected_companies)

        if _filter_active:
            st.markdown(f"""
            <div style='background:#e6f4fb;border:1px solid #b8d9f0;border-left:4px solid #1BA0E2;
                        border-radius:6px;padding:8px 16px;margin:8px 0 18px 0;font-size:0.83em;color:#0D7FCC;
                        display:flex;gap:20px;align-items:center;'>
                🔎 <b>Filter aktif</b> &nbsp;|&nbsp; Menampilkan 
                <b>{_filtered:,}</b> dari <b>{_total:,}</b> records 
                <span style='background:#1BA0E2;color:white;border-radius:20px;padding:1px 10px;font-size:0.9em;'>{_pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        
        # Gunakan df_filtered di seluruh tab (menggantikan df_all)
        df_all = df_filtered

        st.markdown("<div class='divider' style='margin: 4px 0 18px 0;'></div>", unsafe_allow_html=True)

        # ======================================
        # ======================================
        # TAB
        # ======================================
        tab2, tab6, tab_cost, tab_room, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "Dashboard",
            "Price Intelligence",
            "Cost Avoidance",
            "Room Intelligence",
            "Sankey Flow",
            "Top Hotel/City",
            "Dendrogram",
            "Patra Jasa",
            "Laporan",
        ])

        # ======================================
        # TAB 2: DASHBOARD
        # ======================================
        with tab2:

        # ======================================
        # FILTER OVERVIEW — NAMA PERUSAHAAN
        # ======================================
            df_overview = df_all.copy()

            # ======================================
            # FILTER DOMESTIK & INTERNASIONAL
            # ======================================
            if "Country" in df_overview.columns:
                _country_upper = df_overview["Country"].astype(str).str.strip().str.upper()
                _has_domestic   = (_country_upper == "INDONESIA").any()
                _has_intl       = (_country_upper != "INDONESIA").any()

                st.markdown("""
                <style>
                div[data-testid="stRadio"] > label {
                    font-size: 0.85em !important;
                    color: #555 !important;
                    font-weight: 500 !important;
                }
                div[data-testid="stRadio"] [data-baseweb="radio"] label {
                    font-size: 0.875em !important;
                }
                </style>
                """, unsafe_allow_html=True)

                _market_options = ["Semua", "Domestik 🇮🇩", "Internasional 🌍"]
                _market_filter = st.radio(
                    "🌐 Market",
                    options=_market_options,
                    index=0,
                    horizontal=True,
                    key="db_market_filter"
                )

                if _market_filter == "Domestik 🇮🇩":
                    df_overview = df_overview[
                        df_overview["Country"].astype(str).str.strip().str.upper() == "INDONESIA"
                    ]
                    _market_badge = (
                        "<span style='background:#e6f4fb;color:#0D7FCC;border:1px solid #b8d9f0;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🇮🇩 Domestik</span>"
                    )
                elif _market_filter == "Internasional 🌍":
                    df_overview = df_overview[
                        df_overview["Country"].astype(str).str.strip().str.upper() != "INDONESIA"
                    ]
                    _market_badge = (
                        "<span style='background:#fff4e6;color:#b05a00;border:1px solid #f5c891;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🌍 Internasional</span>"
                    )
                else:
                    _market_badge = (
                        "<span style='background:#f0f5e9;color:#3a6b00;border:1px solid #bdd99a;"
                        "border-radius:20px;padding:2px 12px;font-size:0.8em;font-weight:600;"
                        "margin-left:8px;'>🌐 Semua Market</span>"
                    )

                # Info badge jumlah data setelah filter market
                _n_market = len(df_overview)
                st.markdown(
                    f"<div style='margin:4px 0 14px 0;font-size:0.82em;color:#666;'>"
                    f"Menampilkan <b>{_n_market:,}</b> records{_market_badge}</div>",
                    unsafe_allow_html=True
                )


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
                if "Invoice Amount" in df_overview.columns and "Number of Rooms Night" in df_overview.columns:
                    _rn_sum = pd.to_numeric(df_overview["Number of Rooms Night"], errors="coerce").sum()
                    _avg_rate = df_overview["Invoice Amount"].sum() / _rn_sum if _rn_sum else 0
                    st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Avg Rate / Night</div>
                            <div class='metric-value'>{_avg_rate/1000:,.0f}k</div>
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
                if "Meal Plan" in df_overview.columns and len(df_overview):
                    _bf = df_overview["Meal Plan"].astype(str).str.lower().str.contains("breakfast").mean() * 100
                    st.metric("Breakfast Included", f"{_bf:.0f}%")

            with col2:
                if "Currency" in df_overview.columns:
                    _non_idr = (df_overview["Currency"].astype(str).str.upper().str.strip() != "IDR").sum()
                    st.metric("Booking non-IDR", f"{_non_idr:,}")

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
                            df_overview[col] = pd.to_datetime(df_overview[col], errors="coerce", dayfirst=True)

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
                    color_continuous_scale=["#ffffff", "#ddd", "#1BA0E2"]
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
                _city_c = "City Destination" if "City Destination" in df_overview.columns else "City"
                if _city_c in df_overview.columns:
                    top_cities = df_overview[_city_c].value_counts().head(10)

                    fig_cities = px.bar(
                        x=top_cities.values,
                        y=top_cities.index,
                        orientation="h",
                        text=top_cities.values
                    )

                    fig_cities.update_traces(
                        marker_color="#1BA0E2",
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
                _ctry_c = "Country Destination" if "Country Destination" in df_overview.columns else "Country"
                if _ctry_c in df_overview.columns:
                    top_countries = df_overview[_ctry_c].value_counts().head(10)

                    fig_countries = px.pie(
                        values=top_countries.values,
                        names=top_countries.index,
                        hole=0.4,
                        color_discrete_sequence=["#1BA0E2", "#47b5e8", "#75caf0", "#a3dcf6", "#d1eefb"]
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
                        line=dict(color="#1BA0E2", width=3), 
                        marker=dict(size=8, color="#1BA0E2")
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

                    if can_download():
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_tr_trend,
                            file_name="monthly_travel_request_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_monthly_tr_trend")
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download tidak tersedia untuk akun ini</span>
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
                        color_discrete_sequence=["#1BA0E2", "#cccccc"]
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
                        line=dict(color="#1BA0E2", width=3), 
                        marker=dict(size=8, color="#1BA0E2")
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

                    if can_download():
                        st.download_button(
                            label="⬇️ Download Data",
                            data=output_rn_trend,
                            file_name="monthly_room_nights_trend.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_monthly_rn_trend")
                    else:
                        st.markdown("""
                        <div style='background:#f9f9f9;border:1px solid #b8d9f0;border-left:3px solid #1BA0E2;
                        border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;
                        display:flex;align-items:center;gap:8px;'>
                            <span>🔒</span><span>Download tidak tersedia untuk akun ini</span>
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
        # TAB 6: PRICE INTELLIGENCE — Global Filters
        # ======================================
        with tab6:
            def _render_tab6():

                st.markdown("""
                <style>
                :root {
                    --pi-blue:      #1BA0E2;
                    --pi-blue-mid:  #1494C6;
                    --pi-blue-dark: #0D7FCC;
                    --pi-navy:      #062440;
                    --pi-lime:      #e2f871;
                    --pi-bg:        #f0f8ff;
                    --pi-border:    #cce4f4;
                    --pi-surface:   #ffffff;
                    --pi-muted:     #6a8fa0;
                }

                .pi-header {
                    background: linear-gradient(135deg, #062440 0%, #0D7FCC 50%, #1BA0E2 100%);
                    border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
                    position: relative; overflow: hidden;
                    box-shadow: 0 8px 32px rgba(13,127,204,0.26);
                }
                .pi-header::before {
                    content: ''; position: absolute; top: -70px; right: -50px;
                    width: 240px; height: 240px; border-radius: 50%;
                    background: rgba(255,255,255,0.05); pointer-events: none;
                }
                .pi-header::after {
                    content: ''; position: absolute; bottom: -50px; left: 25%;
                    width: 200px; height: 200px; border-radius: 50%;
                    background: rgba(226,248,113,0.07); pointer-events: none;
                }
                .pi-header-inner {
                    display: flex; align-items: center; gap: 16px;
                    position: relative; z-index: 1;
                }
                .pi-header-icon {
                    width: 52px; height: 52px; border-radius: 12px;
                    background: rgba(255,255,255,0.14);
                    border: 1px solid rgba(255,255,255,0.22);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.6em; flex-shrink: 0;
                }
                .pi-header-title { color: #fff; font-size: 1.45em; font-weight: 700; margin: 0; }
                .pi-header-sub   { color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px; }
                .pi-header-badge {
                    margin-left: auto;
                    display: inline-flex; align-items: center; gap: 7px;
                    background: rgba(255,255,255,0.12);
                    border: 1px solid rgba(255,255,255,0.22);
                    border-radius: 100px; padding: 6px 16px;
                    font-size: 0.72em; font-weight: 600;
                    color: rgba(255,255,255,0.85); letter-spacing: 0.06em;
                    white-space: nowrap;
                }
                .pi-badge-dot {
                    width: 6px; height: 6px; border-radius: 50%;
                    background: var(--pi-lime); box-shadow: 0 0 6px var(--pi-lime);
                    animation: pi-blink 2s ease infinite;
                }
                @keyframes pi-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

                /* ── Global Filter Bar ── */
                .pi-global-filter {
                    background: var(--pi-surface);
                    border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue);
                    border-radius: 10px; padding: 16px 22px 14px;
                    margin-bottom: 24px;
                    box-shadow: 0 2px 10px rgba(13,127,204,0.07);
                }
                .pi-global-filter-title {
                    font-size: 0.70em; font-weight: 700; color: var(--pi-blue);
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;
                    display: flex; align-items: center; gap: 8px;
                }
                .pi-filter-active-badge {
                    display: inline-flex; align-items: center; gap: 5px;
                    background: var(--pi-blue); color: white;
                    border-radius: 100px; padding: 2px 10px;
                    font-size: 0.85em; font-weight: 600; letter-spacing: 0.04em;
                }
                .pi-filter-dot {
                    width: 5px; height: 5px; border-radius: 50%;
                    background: var(--pi-lime); display: inline-block;
                }

                /* ── Section Divider ── */
                .pi-section-divider {
                    display: flex; align-items: center; gap: 14px;
                    margin: 28px 0 20px;
                    font-size: 0.70em; font-weight: 700;
                    color: var(--pi-muted); letter-spacing: 0.14em;
                    text-transform: uppercase;
                }
                .pi-section-divider::before,
                .pi-section-divider::after {
                    content: ''; flex: 1; height: 1px; background: var(--pi-border);
                }

                /* ── KPI Cards ── */
                .pi-kpi-grid {
                    display: grid; grid-template-columns: repeat(4,1fr);
                    gap: 14px; margin-bottom: 20px;
                }
                .pi-kpi {
                    background: var(--pi-surface); border-radius: 10px;
                    padding: 18px 16px 14px;
                    border-top: 3px solid var(--pi-blue);
                    box-shadow: 0 2px 10px rgba(13,127,204,0.08);
                    transition: transform .18s, box-shadow .18s;
                }
                .pi-kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,127,204,0.14); }
                .pi-kpi-label {
                    font-size: 0.68em; font-weight: 600; color: var(--pi-muted);
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 8px;
                }
                .pi-kpi-value { font-size: 1.30em; font-weight: 700; color: var(--pi-navy); line-height: 1.2; }
                .pi-kpi-sub   { font-size: 0.72em; color: var(--pi-muted); margin-top: 4px; }

                /* ── Hotel KPI Cards ── */
                .pi-hotel-kpi-grid {
                    display: grid; grid-template-columns: repeat(5,1fr);
                    gap: 12px; margin-bottom: 22px;
                }
                .pi-hotel-kpi {
                    background: var(--pi-surface); border-radius: 10px;
                    padding: 16px 14px 12px;
                    border-top: 3px solid var(--pi-color, #1BA0E2);
                    box-shadow: 0 2px 8px rgba(13,127,204,0.07);
                    text-align: center;
                    transition: transform .18s, box-shadow .18s;
                }
                .pi-hotel-kpi:hover { transform: translateY(-2px); box-shadow: 0 5px 16px rgba(13,127,204,0.13); }
                .pi-hotel-kpi-icon  { font-size: 1.35em; margin-bottom: 6px; display: block; }
                .pi-hotel-kpi-label {
                    font-size: 0.65em; font-weight: 600; color: var(--pi-muted);
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px;
                }
                .pi-hotel-kpi-value {
                    font-size: 1.05em; font-weight: 700;
                    color: var(--pi-color, #1BA0E2); line-height: 1.25;
                }

                /* ── Saving Sim Cards ── */
                .pi-sim-grid {
                    display: grid; grid-template-columns: repeat(3,1fr);
                    gap: 14px; margin: 16px 0;
                }
                .pi-sim-card {
                    border-radius: 10px; padding: 20px 18px;
                    border: 1px solid var(--pi-border);
                }
                .pi-sim-card-label {
                    font-size: 0.68em; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 10px;
                }
                .pi-sim-card-value { font-size: 1.30em; font-weight: 700; line-height: 1.2; }
                .pi-sim-card-sub   { font-size: 0.74em; margin-top: 5px; }

                .pi-sim-current { background: var(--pi-bg); border-color: var(--pi-border); }
                .pi-sim-current .pi-sim-card-label { color: var(--pi-blue); }
                .pi-sim-current .pi-sim-card-value { color: var(--pi-navy); }
                .pi-sim-current .pi-sim-card-sub   { color: var(--pi-muted); }

                .pi-sim-neg { background: #f0fbf4; border-color: #b8e8cc; }
                .pi-sim-neg .pi-sim-card-label { color: #2e8a57; }
                .pi-sim-neg .pi-sim-card-value { color: #1a5c38; }
                .pi-sim-neg .pi-sim-card-sub   { color: #5aaa7e; }

                .pi-sim-saving {
                    background: linear-gradient(135deg, #062440 0%, #0D7FCC 55%, #1BA0E2 100%);
                    border-color: transparent; position: relative; overflow: hidden;
                }
                .pi-sim-saving::after {
                    content: ''; position: absolute; top: -30px; right: -30px;
                    width: 120px; height: 120px; border-radius: 50%;
                    background: rgba(226,248,113,0.12);
                }
                .pi-sim-saving .pi-sim-card-label { color: rgba(255,255,255,0.65); position: relative; z-index: 1; }
                .pi-sim-saving .pi-sim-card-value { color: #fff; font-size: 1.45em; position: relative; z-index: 1; }
                .pi-sim-saving .pi-sim-card-sub   { color: rgba(255,255,255,0.60); position: relative; z-index: 1; }
                .pi-sim-saving-icon { display: block; font-size: 1.4em; margin-bottom: 6px; position: relative; z-index: 1; }

                /* ── Insight Box ── */
                .pi-insight {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue); border-radius: 8px;
                    padding: 16px 20px; font-size: 0.88em;
                    color: #2a3a4a; line-height: 1.7;
                }
                .pi-insight strong { color: var(--pi-navy); }
                .pi-insight em     { color: var(--pi-blue); font-style: normal; font-weight: 600; }

                .pi-locked {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 3px solid var(--pi-blue); border-radius: 6px;
                    padding: 10px 16px; font-size: 0.82em; color: var(--pi-blue);
                    display: flex; align-items: center; gap: 8px; margin-top: 8px;
                }

                /* ── Empty state ── */
                .pi-empty {
                    background: var(--pi-bg); border: 1px solid var(--pi-border);
                    border-left: 4px solid var(--pi-blue); border-radius: 8px;
                    padding: 20px 24px; font-size: 0.88em; color: var(--pi-navy);
                    text-align: center;
                }

                /* ── Radio pill override ── */
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] {
                    display:inline-flex!important; background:#1BA0E2;
                    border-radius:50px; padding:3px; gap:0;
                }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label {
                    cursor:pointer; padding:5px 18px!important; border-radius:50px!important;
                    font-size:0.78em!important; font-weight:500!important;
                    color:rgba(255,255,255,0.80)!important; margin:0!important;
                }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label > div:first-child { display:none!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) { background:white!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color:#1BA0E2!important; font-weight:700!important; }
                div[data-testid="stRadio"].st-key-pi_country_radio > label { display:none!important; }
                </style>

                <!-- Header -->
                <div class="pi-header">
                    <div class="pi-header-inner">
                        <div class="pi-header-icon">💹</div>
                        <div>
                            <div class="pi-header-title">Price Intelligence</div>
                            <div class="pi-header-sub">Pareto 80/20 Spend Concentration · Hotel Rate Benchmarking · Negotiation Simulator</div>
                        </div>
                        <div class="pi-header-badge">
                            <span class="pi-badge-dot"></span>LIVE ANALYSIS
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════
                #  GLOBAL FILTERS — berlaku untuk SEMUA section di tab6
                # ══════════════════════════════════════════════════════
                st.markdown("""
                <div class="pi-global-filter">
                    <div class="pi-global-filter-title">
                        🔍 Global Filters
                        <span style="color:var(--pi-muted);font-weight:400;letter-spacing:0;text-transform:none;font-size:1.1em;">
                            — berlaku untuk seluruh analisis di tab ini
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                gf_col1, gf_col2, gf_col3 = st.columns([1.2, 2, 2])
                gd_col1, gd_col2, gd_col3 = st.columns([1.2, 1.2, 2.8])

                # ── Filter 1: Wilayah ────────────────────────────────────────
                with gf_col1:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Wilayah</div>", unsafe_allow_html=True)
                    if "Country" in df_all.columns:
                        pi_country_filter = st.radio(
                            label="pi_wilayah",
                            options=["Indonesia", "Internasional"],
                            index=0, horizontal=True,
                            label_visibility="collapsed",
                            key="pi_country_radio"
                        )
                    else:
                        pi_country_filter = "Semua"
                        st.info("Kolom Country tidak tersedia")

                # ── Apply wilayah filter ke base dataframe ───────────────────
                df_pi_base = df_all.copy()

                if "Country" in df_pi_base.columns and pi_country_filter in ["Indonesia", "Internasional"]:
                    df_pi_base["_cu"] = df_pi_base["Country"].astype(str).str.strip().str.upper()
                    domestic_alias    = ["INDONESIA", "ID", "IDN"]
                    if pi_country_filter == "Indonesia":
                        df_pi_base    = df_pi_base[df_pi_base["_cu"].isin(domestic_alias)]
                        filter_label  = "🇮🇩 Indonesia"
                    else:
                        df_pi_base    = df_pi_base[~df_pi_base["_cu"].isin(domestic_alias)]
                        filter_label  = "🌐 Internasional"
                    df_pi_base.drop(columns=["_cu"], inplace=True)
                else:
                    filter_label = "🌏 Semua Wilayah"

                # ── Filter tanggal From / To (khusus tab ini) ────────────────
                df_pi_base, pi_period_label = tab_date_range_filter(
                    df_pi_base, _date_col_for_filter, "pi", gd_col1, gd_col2)

                # ── Filter 2: Nama Hotel (multiselect) ───────────────────────
                with gf_col2:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Filter Hotel (opsional)</div>", unsafe_allow_html=True)
                    if "Hotel Name" in df_pi_base.columns:
                        hotel_options = sorted(df_pi_base["Hotel Name"].dropna().unique().tolist())
                        pi_hotel_filter = st.multiselect(
                            label="pi_hotel_ms",
                            options=hotel_options,
                            default=[],
                            placeholder="Semua hotel (pilih untuk filter spesifik)…",
                            label_visibility="collapsed",
                            key="pi_hotel_multiselect"
                        )
                    else:
                        pi_hotel_filter = []

                # ── Filter 3: Dimensi Pareto ─────────────────────────────────
                with gf_col3:
                    st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Dimensi Pareto</div>", unsafe_allow_html=True)
                    dimension_options = [c for c in ["Hotel Name", "City", "Supplier Name"] if c in df_pi_base.columns]
                    if dimension_options:
                        pi_dimension = st.selectbox(
                            label="pi_dim",
                            options=dimension_options,
                            label_visibility="collapsed",
                            key="pi_dimension_select"
                        )
                    else:
                        pi_dimension = None

                # ── Apply hotel filter ───────────────────────────────────────
                if pi_hotel_filter:
                    df_pi_base = df_pi_base[df_pi_base["Hotel Name"].isin(pi_hotel_filter)]

                # ── Toggle penggabungan properti (hanya untuk dimensi Hotel Name) ──
                if pi_dimension == "Hotel Name":
                    _toggle = getattr(st, "toggle", st.checkbox)
                    pi_merge_property = _toggle(
                        "🔗 Gabungkan properti yang sama (nama hotel mirip di kota yang sama)",
                        value=True, key="pi_merge_toggle",
                        help="KPI, grafik Pareto, dan tabel Detail Data akan memakai data properti gabungan."
                    )
                else:
                    pi_merge_property = False

                # ── Filter summary badge ─────────────────────────────────────
                total_records  = len(df_all)
                filtered_records = len(df_pi_base)
                filter_pct     = (filtered_records / total_records * 100) if total_records > 0 else 0
                hotel_label    = f"{len(pi_hotel_filter)} hotel dipilih" if pi_hotel_filter else "Semua hotel"

                st.markdown(f"""
                <div style='background:#e6f4fb;border:1px solid #cce4f4;border-left:4px solid #1BA0E2;
                            border-radius:8px;padding:10px 18px;margin:4px 0 22px 0;
                            font-size:0.82em;color:#0D7FCC;display:flex;gap:20px;align-items:center;flex-wrap:wrap;'>
                    <span>🔎 <b>Filter aktif</b></span>
                    <span>Wilayah: <b>{filter_label}</b></span>
                    <span>Periode: <b>{pi_period_label}</b></span>
                    <span>Hotel: <b>{hotel_label}</b></span>
                    <span>Menampilkan <b>{filtered_records:,}</b> dari <b>{total_records:,}</b> records
                        <span style='background:#1BA0E2;color:white;border-radius:20px;
                                     padding:1px 10px;font-size:0.88em;margin-left:4px;'>{filter_pct:.1f}%</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                if df_pi_base.empty:
                    st.markdown('<div class="pi-empty">⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.</div>',
                                unsafe_allow_html=True)
                    return  # keluar dari tab ini saja, tidak menghentikan tab lain

                # ══════════════════════════════════════════════════════
                #  SECTION 1 — PARETO SPEND CONCENTRATION
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Spend Concentration — Pareto 80/20</div>',
                            unsafe_allow_html=True)

                if "Invoice Amount" not in df_pi_base.columns or pi_dimension is None:
                    st.markdown('<div class="pi-insight">⚠️ Kolom <strong>Invoice Amount</strong> atau dimensi tidak tersedia.</div>',
                                unsafe_allow_html=True)
                else:
                    df_sc = df_pi_base.dropna(subset=["Invoice Amount"]).copy()

                    if df_sc.empty:
                        st.markdown('<div class="pi-insight">⚠️ Tidak ada data setelah filter.</div>',
                                    unsafe_allow_html=True)
                    else:
                        # ── Kunci pengelompokan: properti gabungan atau nama asli ──
                        is_hotel_dim = (pi_dimension == "Hotel Name")
                        if is_hotel_dim and pi_merge_property:
                            df_sc = apply_property_merge(df_sc, property_merge_map,
                                                         hotel_col="Hotel Name", city_col="City")
                            df_sc["_pareto_key"] = df_sc["Hotel / Properti"]
                        else:
                            df_sc["_pareto_key"] = df_sc[pi_dimension].astype(str).str.strip()
                        # buang baris tanpa nama (setara perilaku groupby lama yang mengabaikan NaN)
                        df_sc = df_sc[df_sc["_pareto_key"].notna()
                                      & df_sc["_pareto_key"].astype(str).str.strip().ne("")
                                      & df_sc["_pareto_key"].astype(str).str.lower().ne("nan")]

                        has_rn  = "Number of Rooms Night" in df_sc.columns
                        has_trn = "Travel Request Number" in df_sc.columns
                        agg_map = {"Invoice Amount": ("Invoice Amount", "sum")}
                        if has_rn:
                            agg_map["Room Nights"] = ("Number of Rooms Night", "sum")
                        # Basis rata-rata: booking unik (Travel Request Number),
                        # fallback ke jumlah baris jika kolom tidak tersedia
                        agg_map["Bookings"] = (("Travel Request Number", "nunique") if has_trn
                                               else ("Invoice Amount", "size"))

                        pareto_df = (df_sc.groupby("_pareto_key").agg(**agg_map)
                                     .reset_index()
                                     .rename(columns={"_pareto_key": pi_dimension})
                                     .sort_values("Invoice Amount", ascending=False)
                                     .reset_index(drop=True))
                        if not has_rn:
                            pareto_df["Room Nights"] = np.nan

                        _bk = pareto_df["Bookings"].replace(0, np.nan)
                        pareto_df["Avg Spend / Booking"]      = pareto_df["Invoice Amount"] / _bk
                        pareto_df["Avg Room Night / Booking"] = pareto_df["Room Nights"] / _bk
                        pareto_df["Avg Rate/Night"] = pareto_df["Invoice Amount"] / pareto_df["Room Nights"].replace(0, np.nan)

                        # Daftar nama hotel asli per properti (untuk tabel, Excel & tooltip)
                        if is_hotel_dim:
                            members = build_member_names(df_sc, "_pareto_key", hotel_col="Hotel Name",
                                                         weight_col="Invoice Amount")
                            pareto_df["Nama Hotel Tergabung"] = (
                                pareto_df[pi_dimension].map(members).fillna(pareto_df[pi_dimension]))
                            pareto_df["_n_members"] = pareto_df["Nama Hotel Tergabung"].str.count(";") + 1
                        else:
                            pareto_df["Nama Hotel Tergabung"] = pareto_df[pi_dimension]
                            pareto_df["_n_members"] = 1

                        total_spend           = pareto_df["Invoice Amount"].sum()
                        pareto_df["Spend %"]  = pareto_df["Invoice Amount"] / total_spend * 100
                        pareto_df["Cumulative %"] = pareto_df["Spend %"].cumsum().clip(upper=100)
                        pareto_df["Rank"]     = range(1, len(pareto_df) + 1)

                        top_20_pct_count  = max(1, int(len(pareto_df) * 0.2))
                        top_contributors  = pareto_df.head(top_20_pct_count)
                        top_spend         = top_contributors["Invoice Amount"].sum()
                        top_spend_pct     = top_spend / total_spend * 100

                        # Info penggabungan
                        if is_hotel_dim and pi_merge_property:
                            n_raw    = df_sc["Hotel Name"].astype(str).str.strip().nunique()
                            n_prop   = len(pareto_df)
                            n_merged = int((pareto_df["_n_members"] > 1).sum())
                            st.markdown(f"""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                                        border-radius:6px;padding:8px 16px;margin-bottom:14px;
                                        font-size:0.82em;color:#0D7FCC;'>
                                🔗 <b>{n_raw:,} nama hotel</b> digabung menjadi <b>{n_prop:,} properti</b>
                                ({n_merged:,} properti memiliki lebih dari satu nama) · KPI, grafik &amp; tabel
                                memakai data gabungan
                            </div>
                            """, unsafe_allow_html=True)

                        # KPI Cards
                        st.markdown(f"""
                        <div class="pi-kpi-grid">
                            <div class="pi-kpi" style="border-top-color:#1BA0E2;">
                                <div class="pi-kpi-label">Total Spend · {filter_label}</div>
                                <div class="pi-kpi-value">Rp{total_spend:,.0f}</div>
                                <div class="pi-kpi-sub">keseluruhan periode</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#1494C6;">
                                <div class="pi-kpi-label">Top 20% Count</div>
                                <div class="pi-kpi-value">{top_20_pct_count}</div>
                                <div class="pi-kpi-sub">{"properti" if is_hotel_dim and pi_merge_property else pi_dimension} teratas</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#0D7FCC;">
                                <div class="pi-kpi-label">Top 20% Contribution</div>
                                <div class="pi-kpi-value" style="color:#0D7FCC;">{top_spend_pct:.1f}%</div>
                                <div class="pi-kpi-sub">dari total spend</div>
                            </div>
                            <div class="pi-kpi" style="border-top-color:#062440;">
                                <div class="pi-kpi-label">Bottom 80% Spend</div>
                                <div class="pi-kpi-value">Rp{(total_spend - top_spend):,.0f}</div>
                                <div class="pi-kpi-sub">sisa {len(pareto_df) - top_20_pct_count} entitas</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Pareto Chart (mengikuti data gabungan)
                        colors = ["#1BA0E2" if i < top_20_pct_count else "#d4e8f8"
                                  for i in range(len(pareto_df))]
                        _hover_members = pareto_df.apply(
                            lambda r: ("<br><i>Tergabung dari:</i><br>• " + r["Nama Hotel Tergabung"].replace("; ", "<br>• "))
                            if r["_n_members"] > 1 else "", axis=1)
                        _hover_rn = pareto_df["Room Nights"].apply(
                            lambda v: f"{v:,.0f} room nights" if pd.notna(v) else "-")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=pareto_df[pi_dimension], y=pareto_df["Invoice Amount"],
                            name="Spend", marker=dict(color=colors),
                            customdata=np.stack([_hover_rn, _hover_members], axis=-1),
                            hovertemplate="<b>%{x}</b><br>Rp%{y:,.0f} · %{customdata[0]}%{customdata[1]}<extra></extra>"
                        ))
                        fig.add_trace(go.Scatter(
                            x=pareto_df[pi_dimension], y=pareto_df["Cumulative %"],
                            name="Cumulative %", yaxis="y2", mode="lines+markers",
                            line=dict(color="#062440", width=2.5),
                            marker=dict(size=5, color="#062440"),
                            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
                        ))
                        fig.add_hline(y=80, yref="y2", line_dash="dash",
                                      line_color="#1BA0E2", opacity=0.5,
                                      annotation_text="80%", annotation_position="right",
                                      annotation_font_color="#1BA0E2")
                        fig.update_layout(
                            template="plotly_white",
                            yaxis=dict(title="Spend (Rp)", gridcolor="#e8f4fd"),
                            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                        range=[0, 100], showgrid=False, ticksuffix="%"),
                            height=460, plot_bgcolor="white", paper_bgcolor="white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="right", x=1,
                                        bgcolor="rgba(240,248,255,0.9)",
                                        bordercolor="#cce4f4", borderwidth=1),
                            margin=dict(l=60, r=80, t=40, b=100),
                            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Sub-tabs
                        tab_sim, tab_detail, tab_insight = st.tabs([
                            "💰  Saving Simulation", "📋  Detail Data", "💡  Insight"
                        ])

                        with tab_sim:
                            renegotiation_rate = st.slider(
                                "Target diskon pada Top 20% contributors (%)", 0, 25, 5, 1,
                                key="pi_pareto_disc"
                            )
                            potential_saving = top_spend * (renegotiation_rate / 100)
                            st.markdown(f"""
                            <div class="pi-sim-grid">
                                <div class="pi-sim-card pi-sim-current">
                                    <div class="pi-sim-card-label">Top 20% Spend</div>
                                    <div class="pi-sim-card-value">Rp{top_spend:,.0f}</div>
                                    <div class="pi-sim-card-sub">{top_20_pct_count} {pi_dimension} · {top_spend_pct:.1f}% of all</div>
                                </div>
                                <div class="pi-sim-card pi-sim-neg">
                                    <div class="pi-sim-card-label">Setelah Diskon {renegotiation_rate}%</div>
                                    <div class="pi-sim-card-value">Rp{top_spend*(1-renegotiation_rate/100):,.0f}</div>
                                    <div class="pi-sim-card-sub">spend setelah renegosiasi</div>
                                </div>
                                <div class="pi-sim-card pi-sim-saving">
                                    <span class="pi-sim-saving-icon">💰</span>
                                    <div class="pi-sim-card-label">Potensi Saving</div>
                                    <div class="pi-sim-card-value">Rp{potential_saving:,.0f}</div>
                                    <div class="pi-sim-card-sub">dengan target diskon {renegotiation_rate}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with tab_detail:
                            name_col = "Hotel / Properti" if is_hotel_dim else pi_dimension
                            detail_df = top_contributors.rename(columns={pi_dimension: name_col})
                            detail_cols = [name_col]
                            if is_hotel_dim:
                                detail_cols.append("Nama Hotel Tergabung")
                            detail_cols += ["Invoice Amount", "Room Nights", "Bookings",
                                            "Avg Spend / Booking", "Avg Room Night / Booking",
                                            "Spend %", "Cumulative %", "Rank"]
                            detail_df = detail_df[detail_cols].reset_index(drop=True)

                            st.dataframe(
                                detail_df.style
                                .format({"Invoice Amount": "Rp{:,.0f}",
                                         "Room Nights": "{:,.0f}",
                                         "Bookings": "{:,.0f}",
                                         "Avg Spend / Booking": "Rp{:,.0f}",
                                         "Avg Room Night / Booking": "{:.2f}",
                                         "Avg Rate/Night": "Rp{:,.0f}",                                         
                                         "Spend %": "{:.2f}%",
                                         "Cumulative %": "{:.2f}%"}, na_rep="-")
                                .background_gradient(subset=["Spend %"], cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )
                            if not has_trn:
                                st.caption("Bookings dihitung dari jumlah baris data "
                                           "(kolom Travel Request Number tidak tersedia).")

                            output_excel = excel_bytes_with_formats(
                                detail_df, "Top Contributors",
                                pct_cols=("Spend %", "Cumulative %"),
                                money_cols=("Invoice Amount", "Avg Spend / Booking", "Avg Rate/Night"),
                                int_cols=("Room Nights", "Bookings", "Rank"),
                                dec_cols=("Avg Room Night / Booking",))
                            if can_download():
                                st.download_button(
                                    label="⬇️ Download Excel", data=output_excel,
                                    file_name=f"pareto_{pi_dimension.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_pareto_contributors")
                            else:
                                st.markdown('<div class="pi-locked">🔒 Download tidak tersedia untuk akun ini</div>',
                                            unsafe_allow_html=True)

                        with tab_insight:
                            st.markdown(f"""
                            <div class="pi-insight">
                                <strong>Konsentrasi Spending · {filter_label} · {hotel_label}</strong><br>
                                Top 20% (<em>{top_20_pct_count} {pi_dimension}</em>) menyumbang
                                <em>{top_spend_pct:.1f}%</em> dari total pengeluaran
                                <strong>Rp{total_spend:,.0f}</strong>.<br><br>
                                Fokus renegosiasi pada kelompok ini memberikan dampak finansial terbesar.
                                Dengan target diskon 10%, estimasi saving yang dapat dicapai adalah
                                <em>Rp{top_spend * 0.10:,.0f}</em>.
                            </div>
                            """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════
                #  SECTION 2 — HOTEL PRICE INTELLIGENCE
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Hotel Price Intelligence</div>',
                            unsafe_allow_html=True)

                # Build hotel display name dari df_pi_base (sudah terfilter wilayah + hotel)
                df_pi_hotel = df_pi_base.copy()

                if "Canonical Hotel Name" in df_pi_hotel.columns:
                    df_pi_hotel["Hotel Display"] = (df_pi_hotel["Canonical Hotel Name"].astype(str)
                                                    .str.strip().str.replace(r"\s+", " ", regex=True).str.title())
                elif "Hotel Name" in df_pi_hotel.columns:
                    df_pi_hotel["Hotel Display"] = (df_pi_hotel["Hotel Name"].astype(str)
                                                    .str.strip().str.replace(r"\s+", " ", regex=True).str.title())
                else:
                    st.markdown('<div class="pi-insight">⚠️ Kolom Hotel Name tidak tersedia.</div>',
                                unsafe_allow_html=True)
                    return  # keluar dari tab ini saja, tidak menghentikan tab lain

                hotel_list_intel = sorted(df_pi_hotel["Hotel Display"].dropna().unique())

                if not hotel_list_intel:
                    st.markdown(f"""
                    <div class="pi-insight">
                        ⚠️ Tidak ada data hotel untuk filter:
                        <strong>{filter_label}</strong>
                        {f"· <strong>{hotel_label}</strong>" if pi_hotel_filter else ""}.
                        Coba ubah filter di atas.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Tampilkan info berapa hotel tersedia
                    st.markdown(f"""
                    <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                                border-radius:6px;padding:10px 16px;margin-bottom:14px;
                                font-size:0.82em;color:#0D7FCC;'>
                        🏨 <b>{len(hotel_list_intel):,} hotel</b> tersedia untuk
                        <b>{filter_label}</b>
                        {f"· filter: <b>{hotel_label}</b>" if pi_hotel_filter else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    selected_hotel = st.selectbox(
                        "Pilih hotel untuk analisis detail",
                        hotel_list_intel,
                        label_visibility="collapsed",
                        key="pi_hotel_detail_select"
                    )

                    df_hotel = df_pi_hotel[df_pi_hotel["Hotel Display"] == selected_hotel].copy()

                    if not df_hotel.empty and "Invoice Amount" in df_hotel.columns and "Number of Rooms Night" in df_hotel.columns:
                        df_valid = df_hotel[
                            df_hotel["Invoice Amount"].notna() &
                            (df_hotel["Number of Rooms Night"] > 0)
                        ].copy()
                        df_valid["Price Per Night"] = df_valid["Invoice Amount"] / df_valid["Number of Rooms Night"]

                        if df_valid.empty:
                            st.markdown('<div class="pi-insight">⚠️ Tidak ada data harga yang valid untuk hotel ini.</div>',
                                        unsafe_allow_html=True)
                        else:
                            avg_rate          = df_valid["Price Per Night"].mean()
                            median_rate       = df_valid["Price Per Night"].median()
                            max_rate          = df_valid["Price Per Night"].max()
                            min_rate          = df_valid["Price Per Night"].min()
                            total_room_nights = df_valid["Number of Rooms Night"].sum()

                            # Hotel KPI Cards
                            st.markdown(f"""
                            <div class="pi-hotel-kpi-grid">
                                <div class="pi-hotel-kpi" style="--pi-color:#1BA0E2;">
                                    <span class="pi-hotel-kpi-icon">📊</span>
                                    <div class="pi-hotel-kpi-label">Avg Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {avg_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#1494C6;">
                                    <span class="pi-hotel-kpi-icon">📍</span>
                                    <div class="pi-hotel-kpi-label">Median Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {median_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#d9534f;">
                                    <span class="pi-hotel-kpi-icon">🔺</span>
                                    <div class="pi-hotel-kpi-label">Max Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {max_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#2e8a57;">
                                    <span class="pi-hotel-kpi-icon">🔻</span>
                                    <div class="pi-hotel-kpi-label">Min Rate</div>
                                    <div class="pi-hotel-kpi-value">Rp {min_rate:,.0f}</div>
                                </div>
                                <div class="pi-hotel-kpi" style="--pi-color:#0D7FCC;">
                                    <span class="pi-hotel-kpi-icon">🌙</span>
                                    <div class="pi-hotel-kpi-label">Total Room Nights</div>
                                    <div class="pi-hotel-kpi-value">{total_room_nights:,.0f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Negotiation Simulator
                            st.markdown('<div class="pi-section-divider">Negotiation Simulator</div>',
                                        unsafe_allow_html=True)

                            target_discount  = st.slider("🎯 Target Discount (%)", 0, 30, 10, key="pi_hotel_disc")
                            negotiated_rate  = avg_rate * (1 - target_discount / 100)
                            estimated_saving = (avg_rate - negotiated_rate) * total_room_nights

                            st.markdown(f"""
                            <div class="pi-sim-grid">
                                <div class="pi-sim-card pi-sim-current">
                                    <div class="pi-sim-card-label">Current Avg Rate / Malam</div>
                                    <div class="pi-sim-card-value">Rp {avg_rate:,.0f}</div>
                                    <div class="pi-sim-card-sub">basis {total_room_nights:,.0f} room nights</div>
                                </div>
                                <div class="pi-sim-card pi-sim-neg">
                                    <div class="pi-sim-card-label">Negotiated Rate (–{target_discount}%)</div>
                                    <div class="pi-sim-card-value">Rp {negotiated_rate:,.0f}</div>
                                    <div class="pi-sim-card-sub">target rate kontrak</div>
                                </div>
                                <div class="pi-sim-card pi-sim-saving">
                                    <span class="pi-sim-saving-icon">💰</span>
                                    <div class="pi-sim-card-label">Estimated Saving</div>
                                    <div class="pi-sim-card-value">Rp {estimated_saving:,.0f}</div>
                                    <div class="pi-sim-card-sub">total potensi hemat</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander("📘 Cara Perhitungan Estimated Saving", expanded=False):
                                st.markdown(f"""
                                <div class="pi-insight">
                                    <strong>Formula:</strong><br>
                                    Saving = (Avg Rate − Negotiated Rate) × Total Room Nights<br><br>
                                    = (Rp {avg_rate:,.0f} − Rp {negotiated_rate:,.0f})
                                    × <em>{total_room_nights:,.0f} malam</em><br>
                                    = <strong>Rp {estimated_saving:,.0f}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pi-insight">⚠️ Data tidak cukup untuk hotel yang dipilih.</div>',
                                    unsafe_allow_html=True)

            # ======================================
            # TAB 7: SANKEY FLOW — MTRAX Blue Theme
            # ======================================
            _render_tab6()

        # ======================================
        # TAB ROOM INTELLIGENCE — konsep sama dengan Price Intelligence,
        # parameter utama: Room Nights
        # ======================================
        with tab_room:
            def _render_tab_room():
                # Kelas CSS pi-* sudah didefinisikan di tab Price Intelligence (global
                # di halaman). Di sini hanya ditambah style untuk radio & header tab ini.
                st.markdown("""
                <style>
                div[data-testid="stRadio"].st-key-ri_country_radio > div[role="radiogroup"] {
                    display:inline-flex!important; background:#1BA0E2;
                    border-radius:50px; padding:3px; gap:0;
                }
                div[data-testid="stRadio"].st-key-ri_country_radio > div[role="radiogroup"] > label {
                    cursor:pointer; padding:5px 18px!important; border-radius:50px!important;
                    font-size:0.78em!important; font-weight:500!important;
                    color:rgba(255,255,255,0.80)!important; margin:0!important;
                }
                div[data-testid="stRadio"].st-key-ri_country_radio > div[role="radiogroup"] > label > div:first-child { display:none!important; }
                div[data-testid="stRadio"].st-key-ri_country_radio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) { background:white!important; }
                div[data-testid="stRadio"].st-key-ri_country_radio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color:#1BA0E2!important; font-weight:700!important; }
                div[data-testid="stRadio"].st-key-ri_country_radio > label { display:none!important; }
                </style>

                <div class="pi-header">
                    <div class="pi-header-inner">
                        <div class="pi-header-icon">🛏️</div>
                        <div>
                            <div class="pi-header-title">Room Intelligence</div>
                            <div class="pi-header-sub">Pareto 80/20 Room Night Concentration · Hotel Room Night Benchmarking · Volume Leverage Simulator</div>
                        </div>
                        <div class="pi-header-badge">
                            <span class="pi-badge-dot"></span>LIVE ANALYSIS
                        </div>
                    </div>
                </div>

                <div class="pi-global-filter">
                    <div class="pi-global-filter-title">
                        🔍 Global Filters
                        <span style="color:var(--pi-muted);font-weight:400;letter-spacing:0;text-transform:none;font-size:1.1em;">
                            — berlaku untuk seluruh analisis di tab ini
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                RN_COL = "Number of Rooms Night"
                if RN_COL not in df_all.columns:
                    st.markdown('<div class="pi-empty">⚠️ Kolom <b>Number of Rooms Night</b> tidak tersedia pada data.</div>',
                                unsafe_allow_html=True)
                    return

                _lbl = ("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;"
                        "letter-spacing:0.08em;margin-bottom:6px;'>{}</div>")
                rf_col1, rf_col2, rf_col3 = st.columns([1.2, 2, 2])
                rd_col1, rd_col2, rd_col3 = st.columns([1.2, 1.2, 2.8])

                # ── Filter 1: Wilayah ──
                with rf_col1:
                    st.markdown(_lbl.format("Wilayah"), unsafe_allow_html=True)
                    if "Country" in df_all.columns:
                        ri_country_filter = st.radio(
                            label="ri_wilayah", options=["Indonesia", "Internasional"],
                            index=0, horizontal=True, label_visibility="collapsed",
                            key="ri_country_radio")
                    else:
                        ri_country_filter = "Semua"
                        st.info("Kolom Country tidak tersedia")

                df_ri_base = df_all.copy()
                if "Country" in df_ri_base.columns and ri_country_filter in ["Indonesia", "Internasional"]:
                    _cu = df_ri_base["Country"].astype(str).str.strip().str.upper()
                    _dom = _cu.isin(["INDONESIA", "ID", "IDN"])
                    if ri_country_filter == "Indonesia":
                        df_ri_base, ri_filter_label = df_ri_base[_dom], "🇮🇩 Indonesia"
                    else:
                        df_ri_base, ri_filter_label = df_ri_base[~_dom], "🌐 Internasional"
                else:
                    ri_filter_label = "🌏 Semua Wilayah"

                # ── Filter tanggal From / To (khusus tab ini) ──
                df_ri_base, ri_period_label = tab_date_range_filter(
                    df_ri_base, _date_col_for_filter, "ri", rd_col1, rd_col2)

                # ── Filter 2: Hotel ──
                with rf_col2:
                    st.markdown(_lbl.format("Filter Hotel (opsional)"), unsafe_allow_html=True)
                    if "Hotel Name" in df_ri_base.columns:
                        ri_hotel_filter = st.multiselect(
                            label="ri_hotel_ms",
                            options=sorted(df_ri_base["Hotel Name"].dropna().unique().tolist()),
                            default=[], placeholder="Semua hotel (pilih untuk filter spesifik)…",
                            label_visibility="collapsed", key="ri_hotel_multiselect")
                    else:
                        ri_hotel_filter = []

                # ── Filter 3: Dimensi Pareto ──
                with rf_col3:
                    st.markdown(_lbl.format("Dimensi Pareto"), unsafe_allow_html=True)
                    ri_dim_options = [c for c in ["Hotel Name", "City", "Supplier Name"] if c in df_ri_base.columns]
                    ri_dimension = (st.selectbox("ri_dim", ri_dim_options, label_visibility="collapsed",
                                                 key="ri_dimension_select") if ri_dim_options else None)

                if ri_hotel_filter:
                    df_ri_base = df_ri_base[df_ri_base["Hotel Name"].isin(ri_hotel_filter)]

                if ri_dimension == "Hotel Name":
                    _toggle = getattr(st, "toggle", st.checkbox)
                    ri_merge_property = _toggle(
                        "🔗 Gabungkan properti yang sama (nama hotel mirip di kota yang sama)",
                        value=True, key="ri_merge_toggle",
                        help="KPI, grafik Pareto, tabel, dan analisis per properti memakai data properti gabungan.")
                else:
                    ri_merge_property = False

                # Hanya baris dengan room night valid
                df_ri_base = df_ri_base.copy()
                df_ri_base[RN_COL] = pd.to_numeric(df_ri_base[RN_COL], errors="coerce")
                df_ri_base = df_ri_base[df_ri_base[RN_COL] > 0]

                ri_total_records = len(df_all)
                ri_filtered      = len(df_ri_base)
                ri_pct           = (ri_filtered / ri_total_records * 100) if ri_total_records else 0
                ri_hotel_label   = f"{len(ri_hotel_filter)} hotel dipilih" if ri_hotel_filter else "Semua hotel"

                st.markdown(f"""
                <div style='background:#e6f4fb;border:1px solid #cce4f4;border-left:4px solid #1BA0E2;
                            border-radius:8px;padding:10px 18px;margin:4px 0 22px 0;
                            font-size:0.82em;color:#0D7FCC;display:flex;gap:20px;align-items:center;flex-wrap:wrap;'>
                    <span>🔎 <b>Filter aktif</b></span>
                    <span>Wilayah: <b>{ri_filter_label}</b></span>
                    <span>Periode: <b>{ri_period_label}</b></span>
                    <span>Hotel: <b>{ri_hotel_label}</b></span>
                    <span>Menampilkan <b>{ri_filtered:,}</b> dari <b>{ri_total_records:,}</b> records (room night &gt; 0)
                        <span style='background:#1BA0E2;color:white;border-radius:20px;
                                     padding:1px 10px;font-size:0.88em;margin-left:4px;'>{ri_pct:.1f}%</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                if df_ri_base.empty or ri_dimension is None:
                    st.markdown('<div class="pi-empty">⚠️ Tidak ada data room night yang sesuai dengan filter.</div>',
                                unsafe_allow_html=True)
                    return

                # Properti gabungan dipakai di Section 1 & 2 agar konsisten
                ri_use_merge = ("Hotel Name" in df_ri_base.columns) and (ri_dimension != "Hotel Name" or ri_merge_property)
                if "Hotel Name" in df_ri_base.columns:
                    if ri_dimension == "Hotel Name" and not ri_merge_property:
                        df_ri_base["Hotel / Properti"] = df_ri_base["Hotel Name"].astype(str).str.strip()
                    else:
                        df_ri_base = apply_property_merge(df_ri_base, property_merge_map,
                                                          hotel_col="Hotel Name", city_col="City")

                has_inv = "Invoice Amount" in df_ri_base.columns
                has_trn = "Travel Request Number" in df_ri_base.columns

                # ══════════════════════════════════════════════════════
                #  SECTION 1 — PARETO ROOM NIGHT CONCENTRATION
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Room Night Concentration — Pareto 80/20</div>',
                            unsafe_allow_html=True)

                is_hotel_dim = (ri_dimension == "Hotel Name")
                df_rc = df_ri_base.copy()
                df_rc["_ri_key"] = (df_rc["Hotel / Properti"] if is_hotel_dim
                                    else df_rc[ri_dimension].astype(str).str.strip())
                df_rc = df_rc[df_rc["_ri_key"].notna()
                              & df_rc["_ri_key"].astype(str).str.strip().ne("")
                              & df_rc["_ri_key"].astype(str).str.lower().ne("nan")]
                if df_rc.empty:
                    st.markdown('<div class="pi-empty">⚠️ Tidak ada data untuk dimensi yang dipilih.</div>',
                                unsafe_allow_html=True)
                    return

                agg = {"Room Nights": (RN_COL, "sum")}
                agg["Bookings"] = ("Travel Request Number", "nunique") if has_trn else (RN_COL, "size")
                if has_inv:
                    agg["Invoice Amount"] = ("Invoice Amount", "sum")
                rp = (df_rc.groupby("_ri_key").agg(**agg).reset_index()
                        .rename(columns={"_ri_key": ri_dimension})
                        .sort_values("Room Nights", ascending=False).reset_index(drop=True))
                if not has_inv:
                    rp["Invoice Amount"] = np.nan

                if is_hotel_dim:
                    _mem = build_member_names(df_rc, "_ri_key", hotel_col="Hotel Name", weight_col=RN_COL)
                    rp["Nama Hotel Tergabung"] = rp[ri_dimension].map(_mem).fillna(rp[ri_dimension])
                    rp["_n_members"] = rp["Nama Hotel Tergabung"].str.count(";") + 1
                else:
                    rp["Nama Hotel Tergabung"] = rp[ri_dimension]
                    rp["_n_members"] = 1

                total_rn = rp["Room Nights"].sum()
                _rbk = rp["Bookings"].replace(0, np.nan)
                rp["Avg Room Night / Booking"] = rp["Room Nights"] / _rbk
                rp["Avg Spend / Booking"]      = rp["Invoice Amount"] / _rbk
                rp["Avg Rate/Night"]  = rp["Invoice Amount"] / rp["Room Nights"].replace(0, np.nan)
                rp["RN %"]            = rp["Room Nights"] / total_rn * 100
                rp["Cumulative %"]    = rp["RN %"].cumsum().clip(upper=100)
                rp["Rank"]            = range(1, len(rp) + 1)

                ri_top_n   = max(1, int(len(rp) * 0.2))
                ri_top     = rp.head(ri_top_n)
                top_rn     = ri_top["Room Nights"].sum()
                top_rn_pct = top_rn / total_rn * 100 if total_rn else 0
                bottom_rn  = total_rn - top_rn
                entity_lbl = "properti" if (is_hotel_dim and ri_merge_property) else ri_dimension

                if is_hotel_dim and ri_merge_property:
                    n_raw = df_rc["Hotel Name"].astype(str).str.strip().nunique()
                    st.markdown(f"""
                    <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                                border-radius:6px;padding:8px 16px;margin-bottom:14px;font-size:0.82em;color:#0D7FCC;'>
                        🔗 <b>{n_raw:,} nama hotel</b> digabung menjadi <b>{len(rp):,} properti</b>
                        ({int((rp["_n_members"] > 1).sum()):,} properti memiliki lebih dari satu nama)
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="pi-kpi-grid">
                    <div class="pi-kpi" style="border-top-color:#1BA0E2;">
                        <div class="pi-kpi-label">Total Room Nights · {ri_filter_label}</div>
                        <div class="pi-kpi-value">{total_rn:,.0f}</div>
                        <div class="pi-kpi-sub">keseluruhan periode</div>
                    </div>
                    <div class="pi-kpi" style="border-top-color:#1494C6;">
                        <div class="pi-kpi-label">Top 20% Count</div>
                        <div class="pi-kpi-value">{ri_top_n}</div>
                        <div class="pi-kpi-sub">{entity_lbl} teratas</div>
                    </div>
                    <div class="pi-kpi" style="border-top-color:#0D7FCC;">
                        <div class="pi-kpi-label">Top 20% Contribution</div>
                        <div class="pi-kpi-value" style="color:#0D7FCC;">{top_rn_pct:.1f}%</div>
                        <div class="pi-kpi-sub">dari total room nights</div>
                    </div>
                    <div class="pi-kpi" style="border-top-color:#062440;">
                        <div class="pi-kpi-label">Bottom 80% Room Nights</div>
                        <div class="pi-kpi-value">{bottom_rn:,.0f}</div>
                        <div class="pi-kpi-sub">sisa {len(rp) - ri_top_n} entitas</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                _colors = ["#1BA0E2" if i < ri_top_n else "#d4e8f8" for i in range(len(rp))]
                _hv_mem = rp.apply(
                    lambda r: ("<br><i>Tergabung dari:</i><br>• " + r["Nama Hotel Tergabung"].replace("; ", "<br>• "))
                    if r["_n_members"] > 1 else "", axis=1)
                _hv_inv = rp["Invoice Amount"].apply(lambda v: f"Rp{v:,.0f}" if pd.notna(v) else "-")
                fig_ri = go.Figure()
                fig_ri.add_trace(go.Bar(
                    x=rp[ri_dimension], y=rp["Room Nights"], name="Room Nights",
                    marker=dict(color=_colors),
                    customdata=np.stack([_hv_inv, _hv_mem], axis=-1),
                    hovertemplate="<b>%{x}</b><br>%{y:,.0f} room nights · %{customdata[0]}%{customdata[1]}<extra></extra>"))
                fig_ri.add_trace(go.Scatter(
                    x=rp[ri_dimension], y=rp["Cumulative %"], name="Cumulative %", yaxis="y2",
                    mode="lines+markers", line=dict(color="#062440", width=2.5),
                    marker=dict(size=5, color="#062440"),
                    hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"))
                fig_ri.add_hline(y=80, yref="y2", line_dash="dash", line_color="#1BA0E2", opacity=0.5,
                                 annotation_text="80%", annotation_position="right",
                                 annotation_font_color="#1BA0E2")
                fig_ri.update_layout(
                    template="plotly_white",
                    yaxis=dict(title="Room Nights", gridcolor="#e8f4fd"),
                    yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                range=[0, 100], showgrid=False, ticksuffix="%"),
                    height=460, plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(240,248,255,0.9)", bordercolor="#cce4f4", borderwidth=1),
                    margin=dict(l=60, r=80, t=40, b=100),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    hovermode="x unified")
                st.plotly_chart(fig_ri, use_container_width=True)

                ri_tab_sim, ri_tab_detail, ri_tab_insight = st.tabs([
                    "📦  Volume Leverage", "📋  Detail Data", "💡  Insight"])

                top_inv       = ri_top["Invoice Amount"].sum() if has_inv else np.nan
                top_avg_rate  = (top_inv / top_rn) if (has_inv and top_rn) else np.nan

                with ri_tab_sim:
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        shift_pct = st.slider("Konsolidasi: % room night Bottom 80% dialihkan ke Top 20%",
                                              0, 50, 10, 1, key="ri_shift_pct")
                    with sc2:
                        vol_disc = st.slider("Target diskon tier volume pada Top 20% (%)",
                                             0, 20, 5, 1, key="ri_vol_disc")
                    add_rn       = bottom_rn * shift_pct / 100
                    proj_top_rn  = top_rn + add_rn
                    vol_saving   = proj_top_rn * top_avg_rate * vol_disc / 100 if pd.notna(top_avg_rate) else np.nan
                    _saving_txt  = f"Rp{vol_saving:,.0f}" if pd.notna(vol_saving) else "-"
                    _rate_txt    = f"Rp{top_avg_rate:,.0f}" if pd.notna(top_avg_rate) else "-"

                    st.markdown(f"""
                    <div class="pi-sim-grid">
                        <div class="pi-sim-card pi-sim-current">
                            <div class="pi-sim-card-label">Top 20% Room Nights</div>
                            <div class="pi-sim-card-value">{top_rn:,.0f}</div>
                            <div class="pi-sim-card-sub">{ri_top_n} {entity_lbl} · avg rate {_rate_txt}/malam</div>
                        </div>
                        <div class="pi-sim-card pi-sim-neg">
                            <div class="pi-sim-card-label">Setelah Konsolidasi +{shift_pct}%</div>
                            <div class="pi-sim-card-value">{proj_top_rn:,.0f}</div>
                            <div class="pi-sim-card-sub">+{add_rn:,.0f} room nights dari Bottom 80%</div>
                        </div>
                        <div class="pi-sim-card pi-sim-saving">
                            <span class="pi-sim-saving-icon">💰</span>
                            <div class="pi-sim-card-label">Potensi Saving (simulasi)</div>
                            <div class="pi-sim-card-value">{_saving_txt}</div>
                            <div class="pi-sim-card-sub">diskon tier volume {vol_disc}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📘 Cara Perhitungan Potensi Saving", expanded=False):
                        st.markdown(f"""
                        <div class="pi-insight">
                            <strong>Formula:</strong><br>
                            Tambahan RN = Bottom 80% RN × % konsolidasi
                            = {bottom_rn:,.0f} × {shift_pct}% = <em>{add_rn:,.0f}</em><br>
                            Proyeksi RN Top 20% = {top_rn:,.0f} + {add_rn:,.0f} = <em>{proj_top_rn:,.0f}</em><br>
                            Saving = Proyeksi RN × Avg Rate Top 20% × Diskon tier<br>
                            = {proj_top_rn:,.0f} × {_rate_txt} × {vol_disc}% = <strong>{_saving_txt}</strong><br><br>
                            <em>Catatan:</em> ini simulasi berbasis asumsi. Diskon tier volume aktual
                            bergantung pada hasil negosiasi dengan masing-masing hotel.
                        </div>
                        """, unsafe_allow_html=True)

                with ri_tab_detail:
                    name_col = "Hotel / Properti" if is_hotel_dim else ri_dimension
                    ri_detail = ri_top.rename(columns={ri_dimension: name_col})
                    cols = [name_col] + (["Nama Hotel Tergabung"] if is_hotel_dim else []) + \
                           ["Room Nights", "Bookings", "Avg Room Night / Booking",
                            "Invoice Amount", "Avg Spend / Booking", "Avg Rate/Night",
                            "RN %", "Cumulative %", "Rank"]
                    ri_detail = ri_detail[cols].reset_index(drop=True)
                    st.dataframe(
                        ri_detail.style
                        .format({"Room Nights": "{:,.0f}", "Bookings": "{:,.0f}",
                                 "Avg Room Night / Booking": "{:.2f}",
                                 "Invoice Amount": "Rp{:,.0f}",
                                 "Avg Spend / Booking": "Rp{:,.0f}",
                                 "Avg Rate/Night": "Rp{:,.0f}",
                                 "RN %": "{:.2f}%", "Cumulative %": "{:.2f}%"}, na_rep="-")
                        .background_gradient(subset=["RN %"], cmap="Blues"),
                        use_container_width=True, hide_index=True)
                    if not has_trn:
                        st.caption("Bookings dihitung dari jumlah baris data (kolom Travel Request Number tidak tersedia).")
                    if can_download():
                        st.download_button(
                            label="⬇️ Download Excel",
                            data=excel_bytes_with_formats(
                                ri_detail, "Top Room Nights",
                                pct_cols=("RN %", "Cumulative %"),
                                money_cols=("Invoice Amount", "Avg Spend / Booking", "Avg Rate/Night"),
                                int_cols=("Room Nights", "Bookings", "Rank"),
                                dec_cols=("Avg Room Night / Booking",)),
                            file_name=f"room_pareto_{ri_dimension.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_room_pareto")
                    else:
                        st.markdown('<div class="pi-locked">🔒 Download tidak tersedia untuk akun ini</div>',
                                    unsafe_allow_html=True)

                with ri_tab_insight:
                    avg_los_all = total_rn / rp["Bookings"].sum() if rp["Bookings"].sum() else 0
                    st.markdown(f"""
                    <div class="pi-insight">
                        <strong>Konsentrasi Room Night · {ri_filter_label} · {ri_hotel_label}</strong><br>
                        Top 20% (<em>{ri_top_n} {entity_lbl}</em>) menyerap
                        <em>{top_rn_pct:.1f}%</em> dari total <strong>{total_rn:,.0f} room nights</strong>
                        dengan rata-rata lama menginap <em>{avg_los_all:.2f} malam</em> per booking.<br><br>
                        Volume yang terkonsentrasi ini menjadi posisi tawar untuk kontrak rate berbasis
                        volume (tiered/volume commitment). Mengalihkan sebagian room night dari Bottom 80%
                        ke hotel Top 20% memperkuat posisi negosiasi tersebut.
                    </div>
                    """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════════
                #  SECTION 2 — HOTEL ROOM NIGHT INTELLIGENCE
                # ══════════════════════════════════════════════════════
                st.markdown('<div class="pi-section-divider">Hotel Room Night Intelligence</div>',
                            unsafe_allow_html=True)

                if "Hotel / Properti" not in df_ri_base.columns:
                    st.markdown('<div class="pi-insight">⚠️ Kolom Hotel Name tidak tersedia.</div>',
                                unsafe_allow_html=True)
                    return

                df_ri_base = df_ri_base[df_ri_base["Hotel / Properti"].astype(str).str.strip().ne("")
                                        & df_ri_base["Hotel / Properti"].astype(str).str.lower().ne("nan")]
                prop_rank = (df_ri_base.groupby("Hotel / Properti")[RN_COL].sum()
                             .sort_values(ascending=False))
                st.markdown(f"""
                <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                            border-radius:6px;padding:10px 16px;margin-bottom:14px;font-size:0.82em;color:#0D7FCC;'>
                    🏨 <b>{len(prop_rank):,} {"properti" if ri_use_merge else "hotel"}</b> tersedia untuk
                    <b>{ri_filter_label}</b> · diurutkan dari room night terbesar
                </div>
                """, unsafe_allow_html=True)

                ri_selected = st.selectbox(
                    "Pilih hotel/properti untuk analisis detail", prop_rank.index.tolist(),
                    format_func=lambda x: f"{x}  ·  {prop_rank[x]:,.0f} RN",
                    label_visibility="collapsed", key="ri_hotel_detail_select")

                df_h = df_ri_base[df_ri_base["Hotel / Properti"] == ri_selected].copy()
                h_rn       = df_h[RN_COL].sum()
                h_book     = df_h["Travel Request Number"].nunique() if has_trn else len(df_h)
                h_los      = h_rn / h_book if h_book else 0
                h_share    = h_rn / df_ri_base[RN_COL].sum() * 100 if df_ri_base[RN_COL].sum() else 0
                h_rate     = (df_h["Invoice Amount"].sum() / h_rn) if (has_inv and h_rn) else np.nan
                h_rate_txt = f"Rp {h_rate:,.0f}" if pd.notna(h_rate) else "-"

                h_members = df_h["Hotel Name"].astype(str).str.strip().unique().tolist()
                if len(h_members) > 1:
                    st.markdown(
                        "<div style='font-size:0.80em;color:#6a8fa0;margin:-6px 0 12px;'>🔗 Tergabung dari: "
                        + " · ".join(h_members) + "</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="pi-hotel-kpi-grid">
                    <div class="pi-hotel-kpi" style="--pi-color:#1BA0E2;">
                        <span class="pi-hotel-kpi-icon">🌙</span>
                        <div class="pi-hotel-kpi-label">Total Room Nights</div>
                        <div class="pi-hotel-kpi-value">{h_rn:,.0f}</div>
                    </div>
                    <div class="pi-hotel-kpi" style="--pi-color:#1494C6;">
                        <span class="pi-hotel-kpi-icon">🧾</span>
                        <div class="pi-hotel-kpi-label">Bookings</div>
                        <div class="pi-hotel-kpi-value">{h_book:,.0f}</div>
                    </div>
                    <div class="pi-hotel-kpi" style="--pi-color:#0D7FCC;">
                        <span class="pi-hotel-kpi-icon">⏱️</span>
                        <div class="pi-hotel-kpi-label">Avg Length of Stay</div>
                        <div class="pi-hotel-kpi-value">{h_los:.2f} malam</div>
                    </div>
                    <div class="pi-hotel-kpi" style="--pi-color:#2e8a57;">
                        <span class="pi-hotel-kpi-icon">📊</span>
                        <div class="pi-hotel-kpi-label">Share Room Nights</div>
                        <div class="pi-hotel-kpi-value">{h_share:.2f}%</div>
                    </div>
                    <div class="pi-hotel-kpi" style="--pi-color:#062440;">
                        <span class="pi-hotel-kpi-icon">💵</span>
                        <div class="pi-hotel-kpi-label">Avg Rate / Night</div>
                        <div class="pi-hotel-kpi-value">{h_rate_txt}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tren room night bulanan
                if "Check in Date" in df_h.columns:
                    _d = pd.to_datetime(df_h["Check in Date"], errors="coerce")
                    trend = (df_h.assign(_m=_d.dt.to_period("M").dt.to_timestamp())
                             .dropna(subset=["_m"]).groupby("_m")[RN_COL].sum().reset_index())
                    if not trend.empty:
                        fig_tr = go.Figure(go.Bar(
                            x=trend["_m"], y=trend[RN_COL], marker_color="#1BA0E2",
                            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} room nights<extra></extra>"))
                        fig_tr.update_layout(
                            template="plotly_white", height=300, plot_bgcolor="white", paper_bgcolor="white",
                            title=dict(text="Tren Room Nights Bulanan (Check in Date)", font=dict(size=13)),
                            yaxis=dict(title="Room Nights", gridcolor="#e8f4fd"),
                            xaxis=dict(tickformat="%b %Y"), margin=dict(l=60, r=30, t=50, b=40))
                        st.plotly_chart(fig_tr, use_container_width=True)

                # Volume commitment simulator
                st.markdown('<div class="pi-section-divider">Volume Commitment Simulator</div>',
                            unsafe_allow_html=True)
                vc1, vc2 = st.columns(2)
                with vc1:
                    commit_growth = st.slider("📈 Komitmen kenaikan volume (%)", 0, 50, 20, key="ri_commit_growth")
                with vc2:
                    commit_disc = st.slider("🎯 Target diskon atas komitmen volume (%)", 0, 20, 5, key="ri_commit_disc")
                proj_rn     = h_rn * (1 + commit_growth / 100)
                commit_save = proj_rn * h_rate * commit_disc / 100 if pd.notna(h_rate) else np.nan
                _cs_txt     = f"Rp {commit_save:,.0f}" if pd.notna(commit_save) else "-"

                st.markdown(f"""
                <div class="pi-sim-grid">
                    <div class="pi-sim-card pi-sim-current">
                        <div class="pi-sim-card-label">Room Nights Saat Ini</div>
                        <div class="pi-sim-card-value">{h_rn:,.0f}</div>
                        <div class="pi-sim-card-sub">avg rate {h_rate_txt}/malam</div>
                    </div>
                    <div class="pi-sim-card pi-sim-neg">
                        <div class="pi-sim-card-label">Proyeksi Komitmen +{commit_growth}%</div>
                        <div class="pi-sim-card-value">{proj_rn:,.0f}</div>
                        <div class="pi-sim-card-sub">room nights yang ditawarkan ke hotel</div>
                    </div>
                    <div class="pi-sim-card pi-sim-saving">
                        <span class="pi-sim-saving-icon">💰</span>
                        <div class="pi-sim-card-label">Estimated Saving (simulasi)</div>
                        <div class="pi-sim-card-value">{_cs_txt}</div>
                        <div class="pi-sim-card-sub">dengan diskon {commit_disc}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📘 Cara Perhitungan Estimated Saving", expanded=False):
                    st.markdown(f"""
                    <div class="pi-insight">
                        <strong>Formula:</strong><br>
                        Proyeksi RN = RN saat ini × (1 + komitmen kenaikan)
                        = {h_rn:,.0f} × (1 + {commit_growth}%) = <em>{proj_rn:,.0f}</em><br>
                        Saving = Proyeksi RN × Avg Rate × Diskon
                        = {proj_rn:,.0f} × {h_rate_txt} × {commit_disc}% = <strong>{_cs_txt}</strong><br><br>
                        <em>Catatan:</em> simulasi ini mengasumsikan komitmen volume dapat dipenuhi
                        dan hotel menyetujui diskon yang ditargetkan.
                    </div>
                    """, unsafe_allow_html=True)

            _render_tab_room()


        # ======================================
        # TAB COST AVOIDANCE (modul cost_avoidance.py)
        # ======================================
        with tab_cost:
            render_cost_avoidance(
                df_all,
                can_download=can_download(),
                can_edit=is_admin(),
                username=st.session_state.get("username", ""),
                excel_fn=excel_bytes_with_formats,
                standardize_name=standardize_name,
                normalize_city=_normalize_city_key,
            )

        with tab7:

            st.markdown("""
            <style>
            :root {
                --sk-blue:      #1BA0E2;
                --sk-blue-mid:  #1494C6;
                --sk-blue-dark: #0D7FCC;
                --sk-navy:      #062440;
                --sk-lime:      #e2f871;
                --sk-lime-v:    #98ea16;
                --sk-bg:        #f0f8ff;
                --sk-border:    #cce4f4;
                --sk-surface:   #ffffff;
                --sk-muted:     #6a8fa0;
                --sk-text:      #1a2a3a;
            }

            /* ════════════ HEADER ════════════ */
            .sk-header {
                background: linear-gradient(135deg, #062440 0%, #0D7FCC 52%, #1BA0E2 100%);
                border-radius: 14px;
                padding: 28px 36px;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(13,127,204,0.28);
            }
            .sk-header::before {
                content: '';
                position: absolute; top: -70px; right: -50px;
                width: 240px; height: 240px; border-radius: 50%;
                background: rgba(255,255,255,0.05); pointer-events: none;
            }
            .sk-header::after {
                content: '';
                position: absolute; bottom: -50px; left: 30%;
                width: 200px; height: 200px; border-radius: 50%;
                background: rgba(226,248,113,0.08); pointer-events: none;
            }
            .sk-header-inner {
                display: flex; align-items: center; gap: 18px;
                position: relative; z-index: 1;
            }
            .sk-header-icon {
                width: 54px; height: 54px; border-radius: 13px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.24);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.7em; flex-shrink: 0;
            }
            .sk-header-body  { flex: 1; }
            .sk-header-title {
                color: #fff; font-size: 1.45em; font-weight: 700;
                margin: 0; letter-spacing: -0.01em;
            }
            .sk-header-sub {
                color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px;
            }
            .sk-header-badge {
                display: inline-flex; align-items: center; gap: 7px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 100px; padding: 6px 16px;
                font-size: 0.72em; font-weight: 600;
                color: rgba(255,255,255,0.88); letter-spacing: 0.06em;
                white-space: nowrap;
            }
            .sk-badge-dot {
                width: 6px; height: 6px; border-radius: 50%;
                background: var(--sk-lime);
                box-shadow: 0 0 6px var(--sk-lime);
                animation: sk-blink 2s ease infinite;
            }
            @keyframes sk-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

            /* ════════════ FLOW PATH INDICATOR ════════════ */
            .sk-flow-path {
                display: flex; align-items: center; justify-content: center;
                gap: 0; margin-bottom: 20px;
            }
            .sk-flow-node {
                background: var(--sk-surface);
                border: 1.5px solid var(--sk-border);
                border-radius: 8px; padding: 10px 20px;
                font-size: 0.82em; font-weight: 700;
                color: var(--sk-navy); letter-spacing: 0.02em;
                display: flex; align-items: center; gap: 8px;
                box-shadow: 0 2px 8px rgba(13,127,204,0.08);
            }
            .sk-flow-node-icon { font-size: 1.1em; }
            .sk-flow-arrow {
                display: flex; align-items: center;
                padding: 0 4px;
            }
            .sk-flow-arrow-line {
                width: 36px; height: 2px;
                background: linear-gradient(90deg, var(--sk-border), var(--sk-blue));
            }
            .sk-flow-arrow-head {
                width: 0; height: 0;
                border-top: 5px solid transparent;
                border-bottom: 5px solid transparent;
                border-left: 8px solid var(--sk-blue);
            }
            .sk-flow-node.active {
                background: linear-gradient(135deg, #f0f8ff, #e6f4fb);
                border-color: var(--sk-blue);
                color: var(--sk-blue-dark);
            }

            /* ════════════ FILTER PANEL ════════════ */
            .sk-filter-panel {
                background: var(--sk-surface);
                border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue);
                border-radius: 10px;
                padding: 16px 20px 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(13,127,204,0.07);
            }
            .sk-filter-label {
                font-size: 0.69em; font-weight: 700;
                color: var(--sk-blue); text-transform: uppercase;
                letter-spacing: 0.12em; margin-bottom: 10px;
                display: flex; align-items: center; gap: 8px;
            }
            .sk-filter-tag {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                color: var(--sk-blue); border-radius: 20px;
                padding: 2px 10px; font-size: 0.85em; font-weight: 600;
            }

            /* ════════════ SECTION DIVIDER ════════════ */
            .sk-divider {
                display: flex; align-items: center; gap: 14px;
                margin: 22px 0 16px;
                font-size: 0.69em; font-weight: 700;
                color: var(--sk-muted); letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .sk-divider::before,
            .sk-divider::after {
                content: ''; flex: 1; height: 1px; background: var(--sk-border);
            }

            /* ════════════ STAT CARDS ════════════ */
            .sk-stat-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 12px; margin-bottom: 18px;
            }
            .sk-stat {
                background: var(--sk-surface);
                border-radius: 10px; padding: 16px 14px 12px;
                border-top: 3px solid var(--sk-color, #1BA0E2);
                box-shadow: 0 2px 8px rgba(13,127,204,0.07);
                text-align: center;
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .sk-stat:hover { transform: translateY(-2px); box-shadow: 0 5px 16px rgba(13,127,204,0.13); }
            .sk-stat-icon  { font-size: 1.35em; margin-bottom: 6px; display: block; }
            .sk-stat-label {
                font-size: 0.66em; font-weight: 600; color: var(--sk-muted);
                text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 5px;
            }
            .sk-stat-value {
                font-size: 1.15em; font-weight: 700;
                color: var(--sk-color, #1BA0E2); line-height: 1.2;
            }

            /* ════════════ ACTIVE FILTER BADGE ════════════ */
            .sk-active-filter {
                background: #e6f4fb; border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue);
                border-radius: 8px; padding: 10px 18px;
                margin-bottom: 16px;
                font-size: 0.82em; color: var(--sk-blue-dark);
                display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
            }

            /* ════════════ DOWNLOAD / LOCKED ════════════ */
            .sk-locked {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                border-left: 3px solid var(--sk-blue); border-radius: 6px;
                padding: 10px 16px; font-size: 0.82em; color: var(--sk-blue);
                display: flex; align-items: center; gap: 8px; margin-top: 8px;
            }

            /* ════════════ EMPTY / WARN ════════════ */
            .sk-warn {
                background: var(--sk-bg); border: 1px solid var(--sk-border);
                border-left: 4px solid var(--sk-blue); border-radius: 8px;
                padding: 16px 20px; font-size: 0.88em; color: var(--sk-navy);
            }

            /* ════════════ RADIO PILL — WILAYAH ════════════ */
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] {
                display:inline-flex!important; background:#1BA0E2;
                border-radius:50px; padding:3px; gap:0;
            }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label {
                cursor:pointer; padding:5px 20px!important; border-radius:50px!important;
                font-size:0.78em!important; font-weight:500!important;
                color:rgba(255,255,255,0.80)!important; margin:0!important;
            }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label > div:first-child { display:none!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) { background:white!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color:#1BA0E2!important; font-weight:700!important; }
            div[data-testid="stRadio"].st-key-sankey_country_radio > label { display:none!important; }
            </style>

            <!-- ── Header ── -->
            <div class="sk-header">
                <div class="sk-header-inner">
                    <div class="sk-header-icon">🌊</div>
                    <div class="sk-header-body">
                        <div class="sk-header-title">Sankey Flow Analysis</div>
                        <div class="sk-header-sub">
                            Visualisasi alur pengeluaran travel ·
                            Negara → Kota → Hotel · berbasis Invoice Amount
                        </div>
                    </div>
                    <div class="sk-header-badge">
                        <span class="sk-badge-dot"></span>FLOW MAP
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Flow Path Indicator ─────────────────────────────────────────
            st.markdown("""
            <div class="sk-flow-path">
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🌏</span> Negara
                </div>
                <div class="sk-flow-arrow">
                    <div class="sk-flow-arrow-line"></div>
                    <div class="sk-flow-arrow-head"></div>
                </div>
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🏙️</span> Kota
                </div>
                <div class="sk-flow-arrow">
                    <div class="sk-flow-arrow-line"></div>
                    <div class="sk-flow-arrow-head"></div>
                </div>
                <div class="sk-flow-node active">
                    <span class="sk-flow-node-icon">🏨</span> Hotel
                </div>
            </div>
            """, unsafe_allow_html=True)

            required_sankey  = ["Country", "Hotel Name", "Invoice Amount"]
            city_sankey_col  = next(
                (c for c in ["City", "City Destination"] if c in df_overview.columns), None
            )

            if not all(c in df_overview.columns for c in required_sankey) or city_sankey_col is None:
                st.markdown(
                    '<div class="sk-warn">⚠️ Kolom yang dibutuhkan tidak lengkap untuk Sankey Flow.</div>',
                    unsafe_allow_html=True
                )
            else:
                # ── Filter Panel ─────────────────────────────────────────────
                st.markdown("""
                <div class="sk-filter-panel">
                    <div class="sk-filter-label">
                        ⚙ Filter &amp; Parameter
                        <span class="sk-filter-tag">Semua perubahan langsung memperbarui diagram</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                row_ctrl = st.columns([1.4, 1, 1, 1, 1])

                with row_ctrl[0]:
                    st.markdown(
                        "<div style='font-size:0.70em;font-weight:700;color:#6a8fa0;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>"
                        "Wilayah</div>",
                        unsafe_allow_html=True
                    )
                    sankey_country_filter = st.radio(
                        label="sankey_wilayah",
                        options=["Semua", "Domestik", "Internasional"],
                        index=0, horizontal=True,
                        label_visibility="collapsed",
                        key="sankey_country_radio"
                    )

                with row_ctrl[1]:
                    top_n_company = st.selectbox(
                        "🌏 Negara", [3, 5, 8, 10, 15],
                        index=1, key="sankey_top_company"
                    )
                with row_ctrl[2]:
                    top_n_city = st.selectbox(
                        "🏙️ Kota", [5, 8, 10, 15, 20, 30],
                        index=2, key="sankey_top_city"
                    )
                with row_ctrl[3]:
                    top_n_hotel = st.selectbox(
                        "🏨 Hotel", [10, 15, 20, 30, 50],
                        index=2, key="sankey_top_hotel"
                    )
                with row_ctrl[4]:
                    chart_height = st.selectbox(
                        "📐 Tinggi Chart", [600, 750, 900, 1100],
                        index=1, key="sankey_height"
                    )

                # ── Apply wilayah filter ──────────────────────────────────────
                if "Country" in df_overview.columns:
                    _df_sk      = df_overview.copy()
                    _df_sk["_cu"] = _df_sk["Country"].astype(str).str.strip().str.upper()
                    domestic_alias = ["INDONESIA", "ID", "IDN"]

                    if sankey_country_filter == "Semua":
                        df_sankey    = _df_sk.drop(columns=["_cu"])
                        sankey_label = "🌏 Semua"
                    elif sankey_country_filter == "Domestik":
                        df_sankey    = _df_sk[_df_sk["_cu"].isin(domestic_alias)].drop(columns=["_cu"])
                        sankey_label = "🇮🇩 Domestik"
                    else:
                        df_sankey    = _df_sk[~_df_sk["_cu"].isin(domestic_alias)].drop(columns=["_cu"])
                        sankey_label = "🌐 Internasional"
                else:
                    df_sankey    = df_overview.copy()
                    sankey_label = "🌏 Semua"

                if df_sankey.empty:
                    st.markdown(
                        f'<div class="sk-warn">⚠️ Tidak ada data untuk filter: <strong>{sankey_label}</strong></div>',
                        unsafe_allow_html=True
                    )
                else:
                    df_sk = df_sankey.dropna(
                        subset=["Country", city_sankey_col, "Hotel Name", "Invoice Amount"]
                    ).copy()

                    # Compute tops
                    top_companies = (df_sk.groupby("Country")["Invoice Amount"]
                                     .sum().nlargest(top_n_company).index)
                    top_cities    = (df_sk.groupby(city_sankey_col)["Invoice Amount"]
                                     .sum().nlargest(top_n_city).index)
                    top_hotels    = (df_sk.groupby("Hotel Name")["Invoice Amount"]
                                     .sum().nlargest(top_n_hotel).index)

                    df_sk = df_sk[
                        df_sk["Country"].isin(top_companies) &
                        df_sk[city_sankey_col].isin(top_cities) &
                        df_sk["Hotel Name"].isin(top_hotels)
                    ]

                    if df_sk.empty:
                        st.markdown(
                            '<div class="sk-warn">⚠️ Tidak ada data setelah filter diterapkan.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # ── Active filter badge ───────────────────────────────
                        total_flow_spend = df_sk["Invoice Amount"].sum()
                        total_raw_spend  = df_overview["Invoice Amount"].sum() if "Invoice Amount" in df_overview.columns else 0
                        coverage_pct     = (total_flow_spend / total_raw_spend * 100) if total_raw_spend > 0 else 0

                        st.markdown(f"""
                        <div class="sk-active-filter">
                            <span>🔎 <b>Filter aktif</b></span>
                            <span>Wilayah: <b>{sankey_label}</b></span>
                            <span>Top <b>{top_n_company}</b> Negara
                                · <b>{top_n_city}</b> Kota
                                · <b>{top_n_hotel}</b> Hotel</span>
                            <span>Total Spend:
                                <b>Rp{total_flow_spend:,.0f}</b>
                                <span style='background:#1BA0E2;color:white;border-radius:20px;
                                             padding:1px 10px;font-size:0.88em;margin-left:4px;'>
                                    {coverage_pct:.1f}% dari total
                                </span>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Stat Cards ────────────────────────────────────────
                        n_co = df_sk["Country"].nunique()
                        n_ci = df_sk[city_sankey_col].nunique()
                        n_ht = df_sk["Hotel Name"].nunique()
                        n_rn = df_sk["Number of Rooms Night"].sum() if "Number of Rooms Night" in df_sk.columns else 0

                        st.markdown(f"""
                        <div class="sk-stat-grid">
                            <div class="sk-stat" style="--sk-color:#1BA0E2;">
                                <span class="sk-stat-icon">🏢</span>
                                <div class="sk-stat-label">Negara</div>
                                <div class="sk-stat-value">{n_co}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#1494C6;">
                                <span class="sk-stat-icon">🏙️</span>
                                <div class="sk-stat-label">Kota Tujuan</div>
                                <div class="sk-stat-value">{n_ci}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#0D7FCC;">
                                <span class="sk-stat-icon">🏨</span>
                                <div class="sk-stat-label">Hotel</div>
                                <div class="sk-stat-value">{n_ht}</div>
                            </div>
                            <div class="sk-stat" style="--sk-color:#062440;">
                                <span class="sk-stat-icon">💰</span>
                                <div class="sk-stat-label">Total Spend</div>
                                <div class="sk-stat-value">Rp{total_flow_spend/1e9:.1f}B</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Build Sankey data ─────────────────────────────────
                        co_ci = (df_sk.groupby(["Country", city_sankey_col])
                                 ["Invoice Amount"].sum().reset_index())
                        ci_ho = (df_sk.groupby([city_sankey_col, "Hotel Name"])
                                 ["Invoice Amount"].sum().reset_index())

                        companies = list(co_ci["Country"].unique())
                        cities    = list(co_ci[city_sankey_col].unique())
                        hotels    = list(ci_ho["Hotel Name"].unique())
                        node_labels = companies + cities + hotels

                        co_idx = {c: i                              for i, c in enumerate(companies)}
                        ci_idx = {c: len(companies) + i              for i, c in enumerate(cities)}
                        ho_idx = {h: len(companies)+len(cities)+i    for i, h in enumerate(hotels)}

                        sources, targets, values = [], [], []
                        for _, row in co_ci.iterrows():
                            if row["Country"] in co_idx and row[city_sankey_col] in ci_idx:
                                sources.append(co_idx[row["Country"]])
                                targets.append(ci_idx[row[city_sankey_col]])
                                values.append(row["Invoice Amount"])
                        for _, row in ci_ho.iterrows():
                            if row[city_sankey_col] in ci_idx and row["Hotel Name"] in ho_idx:
                                sources.append(ci_idx[row[city_sankey_col]])
                                targets.append(ho_idx[row["Hotel Name"]])
                                values.append(row["Invoice Amount"])

                        # MTRAX blue-family node colors
                        # Companies → blue shades, Cities → teal/mid-blue, Hotels → lime+navy
                        CO_COLORS = [
                            "#1BA0E2", "#1494C6", "#0D7FCC", "#47b5e8",
                            "#75caf0", "#a3dcf6", "#cce4f4", "#e6f4fb",
                            "#5a8fc0", "#2e6fa0", "#062440", "#0a3d62",
                            "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"
                        ]
                        CI_COLORS = [
                            "#0D7FCC", "#1494C6", "#1BA0E2", "#47b5e8",
                            "#75caf0", "#a3dcf6", "#5a8fc0", "#2e6fa0",
                            "#0a4d7a", "#1565C0", "#1976D2", "#1E88E5",
                            "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"
                        ]
                        HO_COLORS = [
                            "#98ea16", "#e2f871", "#b8e040", "#cce84a",
                            "#a8d030", "#062440", "#0D7FCC", "#1BA0E2",
                            "#47b5e8", "#5a8fc0", "#2e6fa0", "#0a3d62",
                            "#1494C6", "#75caf0", "#a3dcf6", "#cce4f4"
                        ]

                        co_colors = [CO_COLORS[i % len(CO_COLORS)] for i in range(len(companies))]
                        ci_colors = [CI_COLORS[i % len(CI_COLORS)] for i in range(len(cities))]
                        ho_colors = [HO_COLORS[i % len(HO_COLORS)] for i in range(len(hotels))]
                        node_colors = co_colors + ci_colors + ho_colors

                        link_colors = ["rgba(27,160,226,0.28)"] * len(sources)

                        # ── Sankey figure ─────────────────────────────────────
                        st.markdown('<div class="sk-divider">Diagram Sankey</div>', unsafe_allow_html=True)

                        fig_sankey = go.Figure(go.Sankey(
                            arrangement="snap",
                            textfont=dict(
                                family="'Segoe UI', Arial, sans-serif",
                                size=11, color="#1a2a3a"
                            ),
                            node=dict(
                                pad=24, thickness=18,
                                line=dict(color="rgba(255,255,255,0.6)", width=0.6),
                                label=node_labels,
                                color=node_colors
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                color=link_colors
                            )
                        ))

                        fig_sankey.update_layout(
                            paper_bgcolor="#f0f8ff",
                            plot_bgcolor="#f0f8ff",
                            height=chart_height,
                            margin=dict(l=16, r=16, t=58, b=16),
                            title=dict(
                                text=(
                                    f"<b>Sankey Flow: Negara → Kota → Hotel</b>"
                                    f"<span style='color:#6a8fa0;font-size:11px;'>"
                                    f"  ·  Top {top_n_company} Negara"
                                    f"  ·  Top {top_n_city} Kota"
                                    f"  ·  Top {top_n_hotel} Hotel"
                                    f"  ·  {sankey_label}"
                                    f"</span>"
                                ),
                                x=0.01, xanchor="left",
                                font=dict(size=13, color="#1a2a3a")
                            )
                        )

                        st.plotly_chart(fig_sankey, use_container_width=True)

                        # ── Top Tables ────────────────────────────────────────
                        st.markdown('<div class="sk-divider">Tabel Ringkasan</div>', unsafe_allow_html=True)

                        tbl_col1, tbl_col2, tbl_col3 = st.columns(3)

                        top_co_tbl = (df_sk.groupby("Country")["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_company))
                        top_ci_tbl = (df_sk.groupby(city_sankey_col)["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_city))
                        top_ht_tbl = (df_sk.groupby("Hotel Name")["Invoice Amount"]
                                      .sum().sort_values(ascending=False)
                                      .reset_index().head(top_n_hotel))

                        with tbl_col1:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #1BA0E2;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#1BA0E2;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🌏 Top Negara
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_co_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )

                        with tbl_col2:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #1494C6;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#1494C6;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🏙️ Top Kota
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_ci_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="Blues"),
                                use_container_width=True, hide_index=True
                            )

                        with tbl_col3:
                            st.markdown("""
                            <div style='background:#f0f8ff;border:1px solid #cce4f4;
                                        border-left:3px solid #0D7FCC;border-radius:8px;
                                        padding:10px 16px 4px;margin-bottom:8px;'>
                                <div style='font-size:0.72em;font-weight:700;color:#0D7FCC;
                                            text-transform:uppercase;letter-spacing:0.08em;'>
                                    🏨 Top Hotel
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(
                                top_ht_tbl.style
                                .format({"Invoice Amount": "Rp{:,.0f}"})
                                .background_gradient(subset=["Invoice Amount"], cmap="PuBu"),
                                use_container_width=True, hide_index=True
                            )

                        # ── Download ──────────────────────────────────────────
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        output_sankey = BytesIO()
                        export_df = (
                            df_sk.groupby(["Country", city_sankey_col, "Hotel Name"])
                            ["Invoice Amount"].sum()
                            .reset_index()
                            .sort_values("Invoice Amount", ascending=False)
                        )
                        export_df.to_excel(output_sankey, index=False, sheet_name="Sankey Flow")
                        output_sankey.seek(0)

                        if can_download():
                            st.download_button(
                                label="⬇️ Download Data Sankey Flow (Excel)",
                                data=output_sankey,
                                file_name=(
                                    f"sankey_flow_{sankey_label.replace('🇮🇩','').replace('🌐','').strip()}_"
                                    f"{datetime.now().strftime('%Y%m%d')}.xlsx"
                                ),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_sankey_flow")
                        else:
                            st.markdown(
                                '<div class="sk-locked">🔒 Download tidak tersedia untuk akun ini</div>',
                                unsafe_allow_html=True
                            )

        # ======================================
        # TAB 8: DATA HOTEL
        # ======================================
        with tab8:
            def _render_tab8():
                st.markdown("<div class='section-title'>Data Hotel</div>", unsafe_allow_html=True)

                # =========================
                # FILTER DOMESTIK / INTERNATIONAL
                # =========================
                if "Country" in df_overview.columns:

                    st.markdown("""
                    <style>
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"]{
                        display:inline-flex!important;
                        background:#1BA0E2;
                        border-radius:50px;
                        padding:3px;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"] > label{
                        cursor:pointer;
                        padding:4px 16px!important;
                        border-radius:50px!important;
                        font-size:0.78em!important;
                        font-weight:500!important;
                        color:rgba(255,255,255,0.80)!important;
                        margin:0!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > div[role="radiogroup"] > label > div:first-child{
                        display:none!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio 
                    > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
                        background:white!important;
                        color:#1BA0E2!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio 
                    > div[role="radiogroup"] > label:has(input:checked) > div:last-child p{
                        color:#1BA0E2!important;
                    }
                    div[data-testid="stRadio"].st-key-tab8_country_radio > label{
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
                        return  # keluar dari tab ini saja, tidak menghentikan tab lain

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
                                "Top 20": "#1BA0E2",
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

                        # Download — tambah Average Room Rate per hotel
                        # (total Invoice Amount ÷ total Room Nights, sama dengan "Avg Rate/Night"
                        # di Price/Room Intelligence; hanya baris dengan RN > 0 & nominal valid)
                        dl_hotels = top_hotels_tab.drop(columns=["Rank", "Highlight"])
                        # Kolom City setelah Hotel Name (kota yang paling sering muncul per hotel)
                        _city_dl = next((c for c in ["City", "City Destination"] if c in df_tab8.columns), None)
                        if _city_dl:
                            _hc = df_tab8[["Hotel Name", _city_dl]].dropna()
                            _hc = _hc[_hc[_city_dl].astype(str).str.strip().str.lower() != "nan"]
                            _hotel_city = (_hc.groupby("Hotel Name")[_city_dl]
                                           .agg(lambda s: s.value_counts().index[0]))
                            dl_hotels.insert(1, "City", dl_hotels["Hotel Name"].map(_hotel_city))
                        if "Invoice Amount" in df_tab8.columns:
                            _rate_src = df_tab8[["Hotel Name", "Number of Rooms Night", "Invoice Amount"]].copy()
                            _rate_src["Number of Rooms Night"] = pd.to_numeric(_rate_src["Number of Rooms Night"], errors="coerce")
                            _rate_src["Invoice Amount"] = pd.to_numeric(_rate_src["Invoice Amount"], errors="coerce")
                            _rate_src = _rate_src[(_rate_src["Number of Rooms Night"] > 0) & _rate_src["Invoice Amount"].notna()]
                            _rate = (_rate_src.groupby("Hotel Name")[["Invoice Amount", "Number of Rooms Night"]].sum())
                            _rate["Average Room Rate"] = (_rate["Invoice Amount"] /
                                                         _rate["Number of Rooms Night"].replace(0, np.nan)).round(0)
                            dl_hotels = dl_hotels.merge(_rate[["Average Room Rate"]], how="left",
                                                        left_on="Hotel Name", right_index=True)
                        output_hotels = excel_bytes_with_formats(
                            dl_hotels, "Top 100 Hotels",
                            money_cols=("Average Room Rate",),
                            int_cols=("Number of Rooms Night",),
                        )

                        if can_download():
                            st.download_button(
                                label="⬇️ Download Data",
                                data=output_hotels,
                                file_name="top_100_hotels_by_room_nights.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_top100_hotels")
                        else:
                            st.markdown("""
                            <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;
                            border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>
                            🔒 Download tidak tersedia untuk akun ini
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
                                "Top 20": "#1BA0E2",
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

                        if can_download():
                            st.download_button(
                                label="⬇️ Download Data",
                                data=output_cities,
                                file_name="top_100_cities_by_room_nights.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="dl_top100_cities")
                        else:
                            st.markdown("""
                            <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;
                            border-radius:6px;padding:10px 16px;font-size:0.82em;color:#1BA0E2;'>
                            🔒 Download tidak tersedia untuk akun ini
                            </div>
                            """, unsafe_allow_html=True)

            # ======================================
            # TAB 9: DENDROGRAM CLUSTERING — MTRAX Blue Theme
            # ======================================
            _render_tab8()
        with tab9:
            from plotly.subplots import make_subplots as _make_subplots
            from scipy.cluster.hierarchy import linkage as sk_linkage, dendrogram as scipy_dendrogram, fcluster
            from sklearn.preprocessing import normalize as sk_normalize

            # ── CSS ─────────────────────────────────────────────────────────
            st.markdown("""
            <style>
            :root {
                --dnd-blue:      #1BA0E2;
                --dnd-blue-mid:  #1494C6;
                --dnd-blue-dark: #0D7FCC;
                --dnd-navy:      #062440;
                --dnd-lime:      #e2f871;
                --dnd-lime-v:    #98ea16;
                --dnd-bg:        #f0f8ff;
                --dnd-border:    #cce4f4;
                --dnd-surface:   #ffffff;
                --dnd-muted:     #6a8fa0;
                --dnd-text:      #1a2a3a;
            }

            /* ════════════ HEADER ════════════ */
            .dnd-header {
                background: linear-gradient(135deg, #062440 0%, #0D7FCC 52%, #1BA0E2 100%);
                border-radius: 14px;
                padding: 28px 36px;
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(13,127,204,0.28);
            }
            .dnd-header::before {
                content: '';
                position: absolute; top: -70px; right: -50px;
                width: 240px; height: 240px; border-radius: 50%;
                background: rgba(255,255,255,0.05); pointer-events: none;
            }
            .dnd-header::after {
                content: '';
                position: absolute; bottom: -50px; left: 28%;
                width: 200px; height: 200px; border-radius: 50%;
                background: rgba(226,248,113,0.08); pointer-events: none;
            }
            .dnd-header-inner {
                display: flex; align-items: center; gap: 18px;
                position: relative; z-index: 1;
            }
            .dnd-header-icon {
                width: 54px; height: 54px; border-radius: 13px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.24);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.7em; flex-shrink: 0;
            }
            .dnd-header-body  { flex: 1; }
            .dnd-header-title { color: #fff; font-size: 1.45em; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
            .dnd-header-sub   { color: rgba(255,255,255,0.58); font-size: 0.82em; margin-top: 4px; }
            .dnd-header-badge {
                display: inline-flex; align-items: center; gap: 7px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 100px; padding: 6px 16px;
                font-size: 0.72em; font-weight: 600;
                color: rgba(255,255,255,0.88); letter-spacing: 0.06em;
                white-space: nowrap;
            }
            .dnd-badge-dot {
                width: 6px; height: 6px; border-radius: 50%;
                background: var(--dnd-lime);
                box-shadow: 0 0 6px var(--dnd-lime);
                animation: dnd-blink 2s ease infinite;
            }
            @keyframes dnd-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

            /* ════════════ SECTION DIVIDER ════════════ */
            .dnd-divider {
                display: flex; align-items: center; gap: 14px;
                margin: 26px 0 18px;
                font-size: 0.69em; font-weight: 700;
                color: var(--dnd-muted); letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            .dnd-divider::before,
            .dnd-divider::after { content: ''; flex: 1; height: 1px; background: var(--dnd-border); }

            /* ════════════ FILTER PANEL ════════════ */
            .dnd-filter-panel {
                background: var(--dnd-surface);
                border: 1px solid var(--dnd-border);
                border-left: 4px solid var(--dnd-blue);
                border-radius: 10px;
                padding: 16px 20px 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(13,127,204,0.07);
            }
            .dnd-filter-label {
                font-size: 0.69em; font-weight: 700;
                color: var(--dnd-blue); text-transform: uppercase;
                letter-spacing: 0.12em; margin-bottom: 10px;
                display: flex; align-items: center; gap: 8px;
            }
            .dnd-filter-tag {
                background: var(--dnd-bg); border: 1px solid var(--dnd-border);
                color: var(--dnd-blue); border-radius: 20px;
                padding: 2px 10px; font-size: 0.85em; font-weight: 600;
            }

            /* ════════════ KPI CARDS ════════════ */
            .dnd-kpi-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 14px; margin-bottom: 20px;
            }
            .dnd-kpi {
                background: var(--dnd-surface);
                border-radius: 10px; padding: 18px 16px 14px;
                border-top: 3px solid var(--dnd-color, #1BA0E2);
                box-shadow: 0 2px 10px rgba(13,127,204,0.08);
                transition: transform .18s ease, box-shadow .18s ease;
            }
            .dnd-kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(13,127,204,0.15); }
            .dnd-kpi-label {
                font-size: 0.67em; font-weight: 600; color: var(--dnd-muted);
                text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 8px;
            }
            .dnd-kpi-value { font-size: 1.35em; font-weight: 700; color: var(--dnd-color, #1BA0E2); line-height: 1.2; }
            .dnd-kpi-sub   { font-size: 0.72em; color: var(--dnd-muted); margin-top: 4px; }

            /* ════════════ CLUSTER CARDS ════════════ */
            .dnd-cluster-grid {
                display: grid; grid-template-columns: repeat(4,1fr);
                gap: 14px; margin-top: 8px;
            }
            .dnd-cluster-card {
                background: var(--dnd-surface);
                border-radius: 12px; padding: 18px 18px 16px;
                border-top: 4px solid var(--cl-color, #1BA0E2);
                box-shadow: 0 2px 12px rgba(13,127,204,0.08);
                transition: transform .18s ease, box-shadow .18s ease;
                position: relative; overflow: hidden;
            }
            .dnd-cluster-card::before {
                content: '';
                position: absolute; top: -20px; right: -20px;
                width: 80px; height: 80px; border-radius: 50%;
                background: var(--cl-color, #1BA0E2);
                opacity: 0.06;
            }
            .dnd-cluster-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(13,127,204,0.14); }

            .dnd-cluster-head {
                display: flex; justify-content: space-between;
                align-items: center; margin-bottom: 12px;
            }
            .dnd-cluster-name {
                font-weight: 700; font-size: 0.92em;
                color: var(--cl-color, #1BA0E2);
            }
            .dnd-cluster-count {
                background: color-mix(in srgb, var(--cl-color, #1BA0E2) 12%, white);
                color: var(--cl-color, #1BA0E2);
                border: 1px solid color-mix(in srgb, var(--cl-color, #1BA0E2) 25%, white);
                border-radius: 100px; padding: 2px 10px;
                font-size: 0.70em; font-weight: 700;
            }
            .dnd-cluster-members {
                font-size: 0.74em; color: #4a5a6a;
                line-height: 1.6; margin-bottom: 14px;
                min-height: 48px;
            }
            .dnd-cluster-stat {
                background: var(--dnd-bg);
                border-radius: 8px; padding: 10px 12px;
                border-left: 3px solid var(--cl-color, #1BA0E2);
            }
            .dnd-cluster-stat-value {
                font-size: 1.0em; font-weight: 700;
                color: var(--cl-color, #1BA0E2); line-height: 1.2;
            }
            .dnd-cluster-stat-label {
                font-size: 0.68em; font-weight: 600;
                color: var(--dnd-muted); margin-top: 2px;
            }

            /* ════════════ INSIGHT BOX ════════════ */
            .dnd-insight {
                background: var(--dnd-bg);
                border: 1px solid var(--dnd-border);
                border-left: 4px solid var(--dnd-blue);
                border-radius: 8px; padding: 14px 18px;
                font-size: 0.84em; color: #2a3a4a; line-height: 1.65;
                margin-top: 4px;
            }
            .dnd-insight strong { color: var(--dnd-navy); }
            .dnd-insight em     { color: var(--dnd-blue); font-style: normal; font-weight: 600; }
            </style>

            <!-- ── Header ── -->
            <div class="dnd-header">
                <div class="dnd-header-inner">
                    <div class="dnd-header-icon">🌿</div>
                    <div class="dnd-header-body">
                        <div class="dnd-header-title">Hierarchical Clustering — Dendrogram</div>
                        <div class="dnd-header-sub">Segmentasi pola perjalanan berbasis kemiripan · Ward / Complete / Average / Single linkage</div>
                    </div>
                    <div class="dnd-header-badge">
                        <span class="dnd-badge-dot"></span>CLUSTERING
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Filter Panel ────────────────────────────────────────────────
            st.markdown("""
            <div class="dnd-filter-panel">
                <div class="dnd-filter-label">
                    ⚙ Parameter Clustering
                    <span class="dnd-filter-tag">Semua perubahan langsung memperbarui chart</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            dend_r1 = st.columns([2, 2, 2, 2])
            with dend_r1[0]:
                dend_entity = st.selectbox(
                    "🎯 Entitas", ["Kota", "Hotel"], key="dend_entity",
                    help="Pilih entitas yang akan dikelompokkan"
                )
            with dend_r1[1]:
                dend_metric_col = st.selectbox(
                    "📐 Metric",
                    ["Invoice Amount", "Number of Rooms Night"],
                    key="dend_metric",
                    help="Metric yang digunakan sebagai dasar clustering"
                )
            with dend_r1[2]:
                dend_method = st.selectbox(
                    "🔗 Linkage",
                    ["ward", "complete", "average", "single"],
                    key="dend_method",
                    help="Algoritma penggabungan klaster"
                )
            with dend_r1[3]:
                dend_top_n = st.selectbox(
                    "🔢 Top N", [10, 15, 20, 25, 30, 40, 50],
                    index=2, key="dend_top_n",
                    help="Jumlah entitas teratas yang dianalisis"
                )

            dend_r2 = st.columns([3, 2])
            with dend_r2[0]:
                n_clusters_dend = st.slider(
                    "🎨 Jumlah Klaster", 2, 8, 4, key="dend_clusters",
                    help="Jumlah kelompok yang terbentuk dari pemotongan dendrogram"
                )
            with dend_r2[1]:
                dend_show_bar = st.checkbox(
                    "Tampilkan bar spend chart", value=True, key="dend_bar"
                )

            # ── Data preparation ─────────────────────────────────────────────
            # Profil tiap entitas = pola bulanan (kolom perusahaan tidak ada di sumber data)
            _df_dend_src = df_all.copy()
            if "Issue Time" in _df_dend_src.columns:
                _df_dend_src["Bulan"] = pd.to_datetime(_df_dend_src["Issue Time"], errors="coerce",
                                                       dayfirst=True).dt.to_period("M").astype(str)
            _city_col = next((c for c in _df_dend_src.columns if c in ["City", "City Destination"]), None)
            entity_map_dend  = {"Hotel": "Hotel Name", "Kota": _city_col}
            pivot_rows_dend  = {"Hotel": "Bulan", "Kota": "Bulan"}

            entity_col_dend = entity_map_dend.get(dend_entity)
            row_col_dend    = pivot_rows_dend.get(dend_entity)
            metric_col_dend = next((c for c in _df_dend_src.columns if c == dend_metric_col), None)

            if (entity_col_dend and entity_col_dend in _df_dend_src.columns
                    and metric_col_dend
                    and row_col_dend and row_col_dend in _df_dend_src.columns):

                df_dend = _df_dend_src[[entity_col_dend, row_col_dend, metric_col_dend]].dropna()
                top_ent = df_dend.groupby(entity_col_dend)[metric_col_dend].sum().nlargest(dend_top_n).index
                df_dend = df_dend[df_dend[entity_col_dend].isin(top_ent)]
                pivot_dend = (df_dend.groupby([entity_col_dend, row_col_dend])[metric_col_dend]
                              .sum().unstack(fill_value=0))
                if pivot_dend.shape[0] >= 2 and pivot_dend.shape[1] >= 1:
                    X_dend = sk_normalize(pivot_dend.values, norm="l2")
                else:
                    X_dend = np.empty((0, 0))
                    st.info("ℹ️ Data belum cukup untuk membentuk dendrogram dengan pilihan ini "
                            "(minimal 2 entitas). Coba ganti Entitas/Metric atau longgarkan filter.")

                if X_dend.shape[0] >= 2:
                    Z_dend       = sk_linkage(X_dend, method=dend_method, metric="euclidean")
                    cluster_ids  = fcluster(Z_dend, t=n_clusters_dend, criterion="maxclust")

                    # MTRAX-aligned cluster palette — blue family + lime accents
                    DEND_PALETTE = [
                        "#1BA0E2",  # MTRAX blue
                        "#0D7FCC",  # blue dark
                        "#98ea16",  # lime vivid
                        "#062440",  # navy
                        "#1494C6",  # blue mid
                        "#e2f871",  # lime light
                        "#47b5e8",  # blue light
                        "#5a8fc0",  # steel blue
                    ]

                    def fmt_val(v, col):
                        return f"Rp{v:,.0f}" if col == "Invoice Amount" else f"{v:,.0f}"

                    labels_list     = pivot_dend.index.tolist()
                    total_spend_dend = df_dend[metric_col_dend].sum()
                    avg_spend_dend   = total_spend_dend / len(pivot_dend) if len(pivot_dend) else 0
                    leaf_colors_map  = {
                        lbl: DEND_PALETTE[(cid - 1) % len(DEND_PALETTE)]
                        for lbl, cid in zip(labels_list, cluster_ids)
                    }

                    # ── KPI Cards ──────────────────────────────────────────
                    st.markdown('<div class="dnd-divider">Ringkasan Analisis</div>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="dnd-kpi-grid">
                        <div class="dnd-kpi" style="--dnd-color:#1BA0E2;">
                            <div class="dnd-kpi-label">Total Entitas</div>
                            <div class="dnd-kpi-value">{len(pivot_dend)}</div>
                            <div class="dnd-kpi-sub">dari Top {dend_top_n} {dend_entity}</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#0D7FCC;">
                            <div class="dnd-kpi-label">Jumlah Klaster</div>
                            <div class="dnd-kpi-value">{n_clusters_dend}</div>
                            <div class="dnd-kpi-sub">linkage: {dend_method}</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#1494C6;">
                            <div class="dnd-kpi-label">Total {dend_metric_col[:16]}</div>
                            <div class="dnd-kpi-value">{fmt_val(total_spend_dend, metric_col_dend)}</div>
                            <div class="dnd-kpi-sub">keseluruhan entitas</div>
                        </div>
                        <div class="dnd-kpi" style="--dnd-color:#062440;">
                            <div class="dnd-kpi-label">Rata-rata per Entitas</div>
                            <div class="dnd-kpi-value">{fmt_val(avg_spend_dend, metric_col_dend)}</div>
                            <div class="dnd-kpi-sub">baseline clustering</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Build dendrogram ────────────────────────────────────
                    color_thresh_dend = Z_dend[-(n_clusters_dend - 1), 2] if n_clusters_dend > 1 else 0
                    dend_no_plot = scipy_dendrogram(
                        Z_dend, labels=labels_list, no_plot=True,
                        color_threshold=color_thresh_dend
                    )
                    leaves_order    = dend_no_plot["leaves"]
                    labels_ordered  = [labels_list[i] for i in leaves_order]
                    leaf_xs         = {lbl: 5 + 10 * i for i, lbl in enumerate(labels_ordered)}
                    spend_series    = (df_dend.groupby(entity_col_dend)[metric_col_dend]
                                       .sum().reindex(labels_ordered).fillna(0))

                    # ── Build Plotly figure ─────────────────────────────────
                    st.markdown('<div class="dnd-divider">Dendrogram Visualisasi</div>', unsafe_allow_html=True)

                    if dend_show_bar:
                        fig_ply = _make_subplots(
                            rows=1, cols=2,
                            column_widths=[0.68, 0.32],
                            horizontal_spacing=0.04,
                            shared_yaxes=True
                        )
                        dr, dc, br, bc = 1, 1, 1, 2
                    else:
                        fig_ply = go.Figure()
                        dr = dc = br = bc = None

                    def _add(trace):
                        if dend_show_bar:
                            fig_ply.add_trace(trace, row=dr, col=dc)
                        else:
                            fig_ply.add_trace(trace)

                    # Edge lines — subtle blue
                    for xi, yi in zip(dend_no_plot["icoord"], dend_no_plot["dcoord"]):
                        _add(go.Scatter(
                            x=yi, y=xi, mode="lines",
                            line=dict(color="rgba(27,160,226,0.32)", width=1.6),
                            hoverinfo="skip", showlegend=False
                        ))

                    # Leaf nodes
                    for lbl in labels_ordered:
                        xp  = leaf_xs[lbl]
                        nc  = leaf_colors_map.get(lbl, "#1BA0E2")
                        cid = cluster_ids[labels_list.index(lbl)]
                        sv  = spend_series.get(lbl, 0)
                        _add(go.Scatter(
                            x=[0], y=[xp],
                            mode="markers+text",
                            marker=dict(size=10, color=nc,
                                        line=dict(color="white", width=2),
                                        symbol="circle"),
                            text=[lbl],
                            textposition="middle left",
                            textfont=dict(size=9, color=nc),
                            hovertemplate=(
                                f"<b>{lbl}</b><br>"
                                f"Klaster: <b>{cid}</b><br>"
                                f"{dend_metric_col}: <b>{fmt_val(sv, metric_col_dend)}</b>"
                                "<extra></extra>"
                            ),
                            showlegend=False
                        ))

                    # Cut threshold line
                    if n_clusters_dend > 1:
                        y_range = [min(leaf_xs.values()) - 5, max(leaf_xs.values()) + 5]
                        _add(go.Scatter(
                            x=[color_thresh_dend, color_thresh_dend], y=y_range,
                            mode="lines",
                            line=dict(color="#d9534f", width=1.6, dash="dash"),
                            name=f"Cut @ {color_thresh_dend:.3f}",
                            showlegend=True
                        ))

                    # Bar chart (spend per entity)
                    if dend_show_bar:
                        bar_colors = [leaf_colors_map.get(l, "#1BA0E2") for l in labels_ordered]
                        fig_ply.add_trace(
                            go.Bar(
                                y=[leaf_xs[l] for l in labels_ordered],
                                x=spend_series.values,
                                orientation="h",
                                marker=dict(color=bar_colors, opacity=0.85,
                                            line=dict(color="white", width=0.5)),
                                showlegend=False,
                                name="Spend"
                            ),
                            row=br, col=bc
                        )

                    # Legend entries per cluster
                    for i in range(n_clusters_dend):
                        fig_ply.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=10, color=DEND_PALETTE[i % len(DEND_PALETTE)]),
                            name=f"Klaster {i + 1}",
                            showlegend=True
                        ))

                    # Layout
                    chart_h  = max(520, len(pivot_dend) * 26 + 80)
                    max_diss = max((max(d) for d in dend_no_plot["dcoord"]), default=1.0)
                    x_left   = -max(0.55 * max_diss, 0.35)

                    _ax = dict(
                        showgrid=True,
                        gridcolor="rgba(204,228,244,0.50)",
                        gridwidth=0.6,
                        zeroline=False,
                        tickfont=dict(size=8, color="#6a8fa0"),
                        linecolor="#cce4f4",
                        linewidth=1,
                        showline=True
                    )

                    fig_ply.update_layout(
                        height=chart_h,
                        paper_bgcolor="#f0f8ff",
                        plot_bgcolor="#f0f8ff",
                        font=dict(size=11, color="#1a2a3a"),
                        title=dict(
                            text=(
                                f"<b>Dendrogram — {dend_entity}</b>"
                                f"<span style='color:#6a8fa0;font-size:11px;'>"
                                f"  ·  Linkage: {dend_method}"
                                f"  ·  {n_clusters_dend} Klaster"
                                f"  ·  Metric: {dend_metric_col}"
                                f"</span>"
                            ),
                            x=0.01, xanchor="left",
                            font=dict(size=13, color="#1a2a3a")
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.01,
                            xanchor="right", x=1,
                            bgcolor="rgba(240,248,255,0.95)",
                            bordercolor="#cce4f4",
                            borderwidth=1,
                            font=dict(size=10, color="#1a2a3a")
                        ),
                        margin=dict(l=10, r=20, t=60, b=40),
                        hovermode="closest",
                        bargap=0.10
                    )

                    fig_ply.update_xaxes(**_ax)
                    fig_ply.update_yaxes(**_ax)
                    fig_ply.update_xaxes(
                        title_text="Dissimilarity",
                        range=[x_left, max_diss * 1.08],
                        row=dr, col=dc
                    )
                    fig_ply.update_yaxes(
                        showticklabels=False, showgrid=False,
                        row=dr, col=dc
                    )
                    if dend_show_bar:
                        fig_ply.update_xaxes(
                            title_text=dend_metric_col[:18],
                            row=br, col=bc
                        )
                        fig_ply.update_yaxes(
                            showticklabels=False, showgrid=False,
                            row=br, col=bc
                        )

                    st.plotly_chart(fig_ply, use_container_width=True)

                    # ── Cluster Summary Cards ──────────────────────────────
                    st.markdown('<div class="dnd-divider">Ringkasan Klaster</div>', unsafe_allow_html=True)

                    cluster_df_dend = pd.DataFrame({
                        dend_entity: labels_list,
                        "Klaster":   cluster_ids,
                        "Total": (
                            df_dend.groupby(entity_col_dend)[metric_col_dend]
                            .sum().reindex(labels_list).values
                        )
                    }).sort_values(["Klaster", "Total"], ascending=[True, False])

                    unique_clusters = sorted(cluster_df_dend["Klaster"].unique())

                    # Render cards in rows of 4
                    for row_start in range(0, len(unique_clusters), 4):
                        row_clusters = unique_clusters[row_start:row_start + 4]
                        cols_c = st.columns(len(row_clusters))

                        for col_c, cid in zip(cols_c, row_clusters):
                            c_color  = DEND_PALETTE[(cid - 1) % len(DEND_PALETTE)]
                            sub      = cluster_df_dend[cluster_df_dend["Klaster"] == cid]
                            total_c  = sub["Total"].sum()
                            pct_c    = total_c / total_spend_dend * 100 if total_spend_dend else 0
                            members  = sub[dend_entity].tolist()
                            shown    = members[:4]
                            more_n   = max(0, len(members) - 4)
                            members_html = "<br>".join(
                                [f"<span style='color:#4a5a6a;'>• {m}</span>" for m in shown]
                            )
                            if more_n > 0:
                                members_html += (
                                    f"<br><span style='color:#a0b0c0;font-size:0.88em;'>"
                                    f"+{more_n} lainnya…</span>"
                                )

                            with col_c:
                                st.markdown(f"""
                                <div class="dnd-cluster-card" style="--cl-color:{c_color};">
                                    <div class="dnd-cluster-head">
                                        <span class="dnd-cluster-name">Klaster {cid}</span>
                                        <span class="dnd-cluster-count">{len(sub)} entitas</span>
                                    </div>
                                    <div class="dnd-cluster-members">{members_html}</div>
                                    <div class="dnd-cluster-stat">
                                        <div class="dnd-cluster-stat-value">{fmt_val(total_c, metric_col_dend)}</div>
                                        <div class="dnd-cluster-stat-label">{pct_c:.1f}% dari total · {dend_metric_col[:18]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # ── Insight Box ────────────────────────────────────────
                    st.markdown('<div class="dnd-divider">Interpretasi</div>', unsafe_allow_html=True)

                    # Find dominant cluster
                    dominant = cluster_df_dend.groupby("Klaster")["Total"].sum().idxmax()
                    dominant_pct = (cluster_df_dend[cluster_df_dend["Klaster"] == dominant]["Total"].sum()
                                    / total_spend_dend * 100)

                    st.markdown(f"""
                    <div class="dnd-insight">
                        <strong>Hasil Clustering — {dend_entity} · {dend_method} linkage · {n_clusters_dend} klaster</strong><br><br>
                        Dari <em>{len(pivot_dend)} {dend_entity}</em> yang dianalisis,
                        terbentuk <em>{n_clusters_dend} klaster</em> berdasarkan kemiripan pola bulanan
                        <em>{dend_metric_col}</em>.<br><br>
                        <strong>Klaster {dominant}</strong> mendominasi dengan kontribusi
                        <em>{dominant_pct:.1f}%</em> dari total.
                        Entitas pada klaster yang sama memiliki profil perjalanan serupa dan berpotensi
                        untuk dikelola dengan <strong>strategi pengadaan bersama</strong> guna meningkatkan
                        leverage negosiasi.
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style='background:#f0f8ff;border:1px solid #cce4f4;
                            border-left:4px solid #1BA0E2;border-radius:8px;
                            padding:16px 20px;font-size:0.88em;color:#0D7FCC;'>
                    ⚠️ Data tidak cukup untuk membuat dendrogram dengan parameter yang dipilih.
                    Pastikan kolom <strong>Hotel Name</strong>, <strong>City</strong>,
                    <strong>Issue Time</strong> dan <strong>Invoice Amount</strong> tersedia.
                </div>
                """, unsafe_allow_html=True)


        # ======================================
        # TAB LAPORAN — hanya laporan agregat + audit log
        # ======================================
        with tab10:
            st.markdown("<div class='section-title'>Laporan</div>", unsafe_allow_html=True)
            st.caption("Semua unduhan berupa laporan agregat dan tercatat atas nama akun Anda. "
                       "Laporan Cost Avoidance (detail, kebocoran, target renegosiasi) ada di tab Cost Avoidance.")

            _rep = df_all.copy()
            _rep["Number of Rooms Night"] = pd.to_numeric(_rep.get("Number of Rooms Night"), errors="coerce")
            _has_tr = "Travel Request Number" in _rep.columns

            def _agg(df, keys):
                g = df.groupby(keys, dropna=False).agg(
                    **{"Room nights": ("Number of Rooms Night", "sum"),
                       "Spend (Rp)": ("Invoice Amount", "sum")})
                if _has_tr:
                    g["Travel requests"] = df.groupby(keys, dropna=False)["Travel Request Number"].nunique()
                g["Avg rate / malam (Rp)"] = (g["Spend (Rp)"] / g["Room nights"].replace(0, np.nan)).round(0)
                return g.reset_index().sort_values("Room nights", ascending=False)

            _reports = []
            if {"Hotel Name", "City"}.issubset(_rep.columns):
                _reports.append(("Ringkasan per hotel", "Hotel, kota, room nights, spend, rata-rata rate per malam",
                                 _agg(_rep, ["Hotel Name", "City"]), "ringkasan_per_hotel"))
            if "City" in _rep.columns:
                _reports.append(("Ringkasan per kota", "Kota, negara, room nights, spend, rata-rata rate per malam",
                                 _agg(_rep, ["Country", "City"] if "Country" in _rep.columns else ["City"]),
                                 "ringkasan_per_kota"))
            if "Issue Time" in _rep.columns:
                _m = _rep.copy()
                _m["Bulan"] = pd.to_datetime(_m["Issue Time"], errors="coerce", dayfirst=True).dt.to_period("M").astype(str)
                if "Country" in _m.columns:
                    _m["Market"] = np.where(_m["Country"].astype(str).str.strip().str.upper() == "INDONESIA",
                                            "Domestik", "Internasional")
                    _keys = ["Bulan", "Market"]
                else:
                    _keys = ["Bulan"]
                _reports.append(("Tren bulanan", "Travel request, room nights dan spend per bulan",
                                 _agg(_m, _keys).sort_values(_keys), "tren_bulanan"))

            _cols = st.columns(3)
            for _i, (_title, _desc, _df, _fname) in enumerate(_reports):
                with _cols[_i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{_title}**")
                        st.caption(f"{_desc} · {len(_df):,} baris".replace(",", "."))
                        if can_download():
                            st.download_button(
                                "⬇️ Unduh Excel",
                                excel_bytes_with_formats(_df, _title[:31],
                                                         money_cols=("Spend (Rp)", "Avg rate / malam (Rp)"),
                                                         int_cols=("Room nights", "Travel requests")),
                                file_name=f"{_fname}_{datetime.now():%Y%m%d}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_rep_{_fname}")
                        else:
                            st.caption("🔒 Unduhan tidak tersedia untuk akun ini.")

            st.info("**Data per transaksi tidak bisa diunduh dari dashboard.** Kebutuhan data per booking "
                    "diajukan lewat permintaan data dan disetujui atasan serta Pejabat Pelindungan Data Pribadi.")

            if is_super_admin():
                st.markdown("<div class='section-title'>Audit log</div>", unsafe_allow_html=True)
                _log = read_audit_tail(200)
                if _log.empty:
                    st.caption("Belum ada catatan.")
                else:
                    st.dataframe(_log, hide_index=True, use_container_width=True, height=360)

        # ======================================
        # TAB 11: PATRA JASA GROUP vs NON-PATRA JASA
        # ======================================
        with tab11:
            st.markdown("<div class='section-title'>Patra Jasa Group vs Non-Patra Jasa — Perbandingan Hotel Domestik</div>", unsafe_allow_html=True)

            # ── Kamus Hotel Patra Jasa Group ──
            PATRA_JASA_HOTELS = [
                "The Patra Bali Resort & Villas",
                "Patra Semarang Hotel & Convention",
                "Patra Cirebon Hotel & Convention",
                "Patra Malioboro Hotel",
                "Patra Dumai Hotel",
                "Patra Bandung Hotel",
                "Patra Jakarta Hotel",
                "Patra Anyer Hotel",
                "Patra Parapat Hotel",
            ]
            # Nama di data sudah diseragamkan, jadi kamus Patra juga dicocokkan
            # dalam bentuk aslinya maupun bentuk standarnya.
            PATRA_JASA_NORMALIZED = sorted(
                {h.lower().strip() for h in PATRA_JASA_HOTELS}
                | {str(standardize_name(h)).lower().strip() for h in PATRA_JASA_HOTELS}
            )

            # ── Validasi kolom minimum ──
            _req_cols = {"Hotel Name", "Country"}
            _missing = _req_cols - set(df_all.columns)
            if _missing:
                st.warning(f"⚠️ Kolom berikut tidak ditemukan di data: {', '.join(_missing)}. Tab ini membutuhkan kolom tersebut.")
            else:
                # ── Filter INDONESIA saja ──
                df_pj = df_all.copy()
                df_pj["_country_up"] = df_pj["Country"].astype(str).str.strip().str.upper()
                df_pj = df_pj[df_pj["_country_up"] == "INDONESIA"].copy()

                if df_pj.empty:
                    st.info("ℹ️ Tidak ada data domestik (Country = INDONESIA) setelah filter global diterapkan.")
                else:
                    # ── Klasifikasi Patra / Non-Patra ──
                    df_pj["_hotel_norm"] = df_pj["Hotel Name"].astype(str).str.lower().str.strip()
                    df_pj["Grup Hotel"] = df_pj["_hotel_norm"].apply(
                        lambda x: "Patra Jasa Group" if x in PATRA_JASA_NORMALIZED else "Non-Patra Jasa"
                    )

                    # ── Tambahkan kolom bulan & tahun ──
                    # PERBAIKAN: sebelumnya hanya .dt.month tanpa memperhitungkan tahun,
                    # sehingga bulan yang sama dari tahun berbeda bisa tercampur/menyembunyikan data.
                    # PERBAIKAN 2: tambah fallback antar-kolom tanggal per baris — kalau kolom utama
                    # gagal di-parse (NaT) untuk sebagian baris, baris itu tidak lagi hilang diam-diam
                    # dari agregasi bulanan, melainkan dicoba pakai kolom tanggal lain yang tersedia.
                    _date_col_pj = None
                    for _c in ["Issue Time", "Check in Date"]:
                        if _c in df_pj.columns:
                            _date_col_pj = _c
                            break
                    if _date_col_pj:
                        df_pj["_dt"] = pd.to_datetime(df_pj[_date_col_pj], errors="coerce", dayfirst=True)
                        for _fb_col in [c for c in ["Issue Time", "Check in Date", "Check out Date"]
                                        if c in df_pj.columns and c != _date_col_pj]:
                            _fb_dt = pd.to_datetime(df_pj[_fb_col], errors="coerce", dayfirst=True)
                            df_pj["_dt"] = df_pj["_dt"].fillna(_fb_dt)
                        df_pj["_month"] = df_pj["_dt"].dt.month
                        df_pj["_year"] = df_pj["_dt"].dt.year

                        _n_unparsed = df_pj["_dt"].isna().sum()
                        if _n_unparsed > 0:
                            st.caption(
                                f"⚠️ {_n_unparsed:,} baris tidak punya tanggal yang bisa dibaca "
                                f"(kolom {_date_col_pj} dan alternatifnya kosong/format tidak dikenali) "
                                f"— baris ini tidak masuk ke tabel bulanan di bawah."
                            )

                    # ── Filter Nama Perusahaan (opsional) ──
                    st.markdown("""
                    <div style='background:var(--clr-surface,#fff);border:1px solid #e8eaf0;
                                border-left:4px solid #1BA0E2;border-radius:6px;
                                padding:10px 16px 8px 16px;margin-bottom:16px;'>
                        <div style='font-size:0.72em;font-weight:700;color:#1BA0E2;
                                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>
                            🔍 Filter Tab Ini
                        </div>
                    </div>""", unsafe_allow_html=True)

                    _pj_col1, _pj_col2, _pj_col3 = st.columns([1.6, 1, 1])

                    with _pj_col1:
                        if "Nama Perusahaan" in df_pj.columns:
                            _pj_company_opts = sorted(df_pj["Nama Perusahaan"].dropna().unique().tolist())
                            _pj_selected_companies = st.multiselect(
                                "🏢 Filter Perusahaan",
                                options=_pj_company_opts,
                                default=[],
                                placeholder="Semua perusahaan",
                                key="pj_company_filter"
                            )
                            if _pj_selected_companies:
                                df_pj = df_pj[df_pj["Nama Perusahaan"].isin(_pj_selected_companies)]
                        else:
                            _pj_selected_companies = []

                    with _pj_col2:
                        # PERBAIKAN: selector tahun eksplisit agar tabel bulanan tidak pernah
                        # mencampur bulan yang sama dari tahun berbeda (mis. Jan 2025 + Jan 2026
                        # tergabung jadi satu kolom "Jan"). Default ke tahun terbaru yang ada di data.
                        if "_year" in df_pj.columns and df_pj["_year"].notna().any():
                            _pj_year_opts = sorted(df_pj["_year"].dropna().astype(int).unique().tolist(), reverse=True)
                            _pj_selected_year = st.selectbox(
                                "📅 Tahun (Tabel Bulanan)",
                                options=_pj_year_opts,
                                index=0,
                                key="pj_year_select"
                            )
                        else:
                            _pj_selected_year = None

                    with _pj_col3:
                        _pj_metric_opt = st.selectbox(
                            "📊 Metrik Tabel",
                            options=["Invoice (unique)", "Room Nights"],
                            key="pj_metric_select"
                        )

                    # ── Hitung aggregasi ──
                    # Invoice unique = jumlah baris unik per Travel Request Number (atau row count jika tidak ada)
                    if "Travel Request Number" in df_pj.columns:
                        _inv_col = "Travel Request Number"
                        pj_inv  = df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"][_inv_col].nunique()
                        npj_inv = df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"][_inv_col].nunique()
                    else:
                        pj_inv  = len(df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"])
                        npj_inv = len(df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"])

                    total_inv = pj_inv + npj_inv
                    pj_inv_pct  = (pj_inv  / total_inv * 100) if total_inv > 0 else 0
                    npj_inv_pct = (npj_inv / total_inv * 100) if total_inv > 0 else 0

                    # Room Nights
                    if "Number of Rooms Night" in df_pj.columns:
                        df_pj["Number of Rooms Night"] = pd.to_numeric(df_pj["Number of Rooms Night"], errors="coerce")
                        pj_rn  = df_pj[df_pj["Grup Hotel"] == "Patra Jasa Group"]["Number of Rooms Night"].sum()
                        npj_rn = df_pj[df_pj["Grup Hotel"] == "Non-Patra Jasa"]["Number of Rooms Night"].sum()
                    else:
                        pj_rn, npj_rn = 0, 0

                    total_rn = pj_rn + npj_rn
                    pj_rn_pct  = (pj_rn  / total_rn * 100) if total_rn > 0 else 0
                    npj_rn_pct = (npj_rn / total_rn * 100) if total_rn > 0 else 0

                    # ── Metric Cards ──
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    with _mc1:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Total Invoice (Domestik)</div>
                            <div class='metric-value'>{total_inv:,}</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc2:
                        st.markdown(f"""
                        <div class='metric-box' style='border-left-color:#1BA0E2;'>
                            <div class='metric-label'>Patra Jasa Group</div>
                            <div class='metric-value' style='color:#1BA0E2;'>{pj_inv:,}</div>
                            <div style='font-size:0.78em;color:#888;margin-top:4px;'>{pj_inv_pct:.1f}% dari total</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc3:
                        st.markdown(f"""
                        <div class='metric-box' style='border-left-color:#ff8c00;'>
                            <div class='metric-label'>Non-Patra Jasa</div>
                            <div class='metric-value' style='color:#ff8c00;'>{npj_inv:,}</div>
                            <div style='font-size:0.78em;color:#888;margin-top:4px;'>{npj_inv_pct:.1f}% dari total</div>
                        </div>""", unsafe_allow_html=True)
                    with _mc4:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-label'>Total Room Nights</div>
                            <div class='metric-value'>{total_rn:,.0f}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Pie Charts ──
                    st.markdown("<div class='section-title'>Proporsi Perbandingan</div>", unsafe_allow_html=True)
                    _pie1, _pie2 = st.columns(2)

                    with _pie1:
                        if total_inv > 0:
                            fig_pj_inv = go.Figure(data=[go.Pie(
                                labels=["Patra Jasa Group", "Non-Patra Jasa"],
                                values=[pj_inv, npj_inv],
                                hole=0.55,
                                marker=dict(colors=["#1BA0E2", "#ff8c00"],
                                            line=dict(color="white", width=2)),
                                textinfo="percent",
                                textfont=dict(size=13),
                                hovertemplate="<b>%{label}</b><br>Invoice: %{value:,}<br>Proporsi: %{percent}<extra></extra>"
                            )])
                            fig_pj_inv.update_layout(
                                title=dict(text="Invoice Unique — Patra vs Non-Patra", font=dict(size=14)),
                                height=340,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                            xanchor="center", x=0.5, font=dict(size=11)),
                                margin=dict(l=10, r=10, t=60, b=20),
                                annotations=[dict(
                                    text=f"<b>{total_inv:,}</b><br><span style='font-size:10px'>total</span>",
                                    x=0.5, y=0.5, font=dict(size=15), showarrow=False
                                )]
                            )
                            st.plotly_chart(fig_pj_inv, use_container_width=True)

                            # Tabel ringkasan invoice
                            st.markdown(f"""
                            <table style='width:100%;border-collapse:collapse;font-size:0.83em;'>
                              <thead>
                                <tr style='background:#f0f8ff;'>
                                  <th style='padding:6px 10px;text-align:left;border:1px solid #e0e0e0;color:#555;'>Grup</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>Invoice</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>%</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#1BA0E2;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Patra Jasa Group
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#1BA0E2;'>{pj_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{pj_inv_pct:.1f}%</td>
                                </tr>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#ff8c00;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Non-Patra Jasa
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#ff8c00;'>{npj_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{npj_inv_pct:.1f}%</td>
                                </tr>
                                <tr style='background:#f9f9f9;'>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;font-weight:600;'>Total</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>{total_inv:,}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>100%</td>
                                </tr>
                              </tbody>
                            </table>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Tidak ada data invoice untuk ditampilkan.")

                    with _pie2:
                        if total_rn > 0:
                            fig_pj_rn = go.Figure(data=[go.Pie(
                                labels=["Patra Jasa Group", "Non-Patra Jasa"],
                                values=[pj_rn, npj_rn],
                                hole=0.55,
                                marker=dict(colors=["#1BA0E2", "#ff8c00"],
                                            line=dict(color="white", width=2)),
                                textinfo="percent",
                                textfont=dict(size=13),
                                hovertemplate="<b>%{label}</b><br>Room Nights: %{value:,.0f}<br>Proporsi: %{percent}<extra></extra>"
                            )])
                            fig_pj_rn.update_layout(
                                title=dict(text="Room Nights — Patra vs Non-Patra", font=dict(size=14)),
                                height=340,
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                                            xanchor="center", x=0.5, font=dict(size=11)),
                                margin=dict(l=10, r=10, t=60, b=20),
                                annotations=[dict(
                                    text=f"<b>{total_rn:,.0f}</b><br><span style='font-size:10px'>total</span>",
                                    x=0.5, y=0.5, font=dict(size=15), showarrow=False
                                )]
                            )
                            st.plotly_chart(fig_pj_rn, use_container_width=True)

                            # Tabel ringkasan room nights
                            st.markdown(f"""
                            <table style='width:100%;border-collapse:collapse;font-size:0.83em;'>
                              <thead>
                                <tr style='background:#f0f8ff;'>
                                  <th style='padding:6px 10px;text-align:left;border:1px solid #e0e0e0;color:#555;'>Grup</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>Room Nights</th>
                                  <th style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;color:#555;'>%</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#1BA0E2;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Patra Jasa Group
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#1BA0E2;'>{pj_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{pj_rn_pct:.1f}%</td>
                                </tr>
                                <tr>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;'>
                                    <span style='display:inline-block;width:10px;height:10px;background:#ff8c00;border-radius:2px;margin-right:6px;vertical-align:middle;'></span>Non-Patra Jasa
                                  </td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;color:#ff8c00;'>{npj_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;'>{npj_rn_pct:.1f}%</td>
                                </tr>
                                <tr style='background:#f9f9f9;'>
                                  <td style='padding:6px 10px;border:1px solid #e0e0e0;font-weight:600;'>Total</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>{total_rn:,.0f}</td>
                                  <td style='padding:6px 10px;text-align:right;border:1px solid #e0e0e0;font-weight:600;'>100%</td>
                                </tr>
                              </tbody>
                            </table>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Kolom 'Number of Rooms Night' tidak ditemukan atau tidak ada data.")

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Tabel Bulanan ──
                    _pj_year_label = f" · Tahun {_pj_selected_year}" if _pj_selected_year is not None else ""
                    st.markdown(
                        f"<div class='section-title'>Tabel Perbandingan Bulanan — "
                        f"{'Invoice (unique)' if _pj_metric_opt == 'Invoice (unique)' else 'Room Nights'}"
                        f"{_pj_year_label}</div>",
                        unsafe_allow_html=True
                    )

                    # PERBAIKAN: scope tabel bulanan ke SATU tahun yang dipilih di atas —
                    # ini mencegah bulan yang sama dari tahun berbeda tergabung jadi satu kolom
                    # (mis. data Jan 2025 + Jan 2026 tidak lagi ikut menumpuk/menyembunyikan bulan lain).
                    df_pj_month_table = df_pj.copy()
                    if _pj_selected_year is not None and "_year" in df_pj_month_table.columns:
                        df_pj_month_table = df_pj_month_table[df_pj_month_table["_year"] == _pj_selected_year]

                    if _date_col_pj and "_month" in df_pj_month_table.columns and not df_pj_month_table.empty:
                        MONTH_NAMES = {
                            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                            7:"Jul",8:"Agt",9:"Sep",10:"Okt",11:"Nov",12:"Des"
                        }

                        if _pj_metric_opt == "Invoice (unique)":
                            if "Travel Request Number" in df_pj_month_table.columns:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])["Travel Request Number"]
                                    .nunique()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])
                                    .size()
                                    .reset_index(name="Nilai")
                                )
                        else:
                            if "Number of Rooms Night" in df_pj_month_table.columns:
                                _monthly_agg = (
                                    df_pj_month_table.groupby(["Grup Hotel","_month"])["Number of Rooms Night"]
                                    .sum()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                st.warning("Kolom 'Number of Rooms Night' tidak tersedia.")
                                _monthly_agg = pd.DataFrame()

                        if not _monthly_agg.empty:
                            _pivot = _monthly_agg.pivot_table(
                                index="Grup Hotel",
                                columns="_month",
                                values="Nilai",
                                fill_value=0
                            )
                            # PERBAIKAN: normalisasi label kolom ke int murni — pivot_table bisa
                            # menghasilkan kolom bertipe float (mis. 4.0) kalau ada NaN tercampur
                            # di data sumber sebelum di-groupby, sehingga pengecekan
                            # "if _m not in _pivot.columns" di bawah (memakai int biasa) gagal
                            # mengenali kolom yang sebenarnya sudah ada, lalu menimpanya dengan 0.
                            _pivot.columns = [int(c) for c in _pivot.columns]

                            # Pastikan semua 12 bulan ada
                            for _m in range(1, 13):
                                if _m not in _pivot.columns:
                                    _pivot[_m] = 0
                            _pivot = _pivot[[m for m in range(1, 13)]]
                            _pivot["Total"] = _pivot.sum(axis=1)
                            _pivot = _pivot.reset_index()

                            # Reorder rows: Patra Jasa dulu
                            _row_order = ["Patra Jasa Group", "Non-Patra Jasa"]
                            _pivot["_sort"] = _pivot["Grup Hotel"].map(
                                {r: i for i, r in enumerate(_row_order)}
                            ).fillna(99)
                            _pivot = _pivot.sort_values("_sort").drop(columns=["_sort"])

                            # Tambah row % share Patra
                            _totals_by_month = {m: _pivot[m].sum() for m in range(1, 13)}
                            _pj_row = _pivot[_pivot["Grup Hotel"] == "Patra Jasa Group"]
                            _share_row = {"Grup Hotel": "% Patra share"}
                            for _m in range(1, 13):
                                _denom = _totals_by_month[_m]
                                _num = _pj_row[_m].values[0] if not _pj_row.empty else 0
                                _share_row[_m] = f"{(_num/_denom*100):.0f}%" if _denom > 0 else "-"
                            _total_denom = _pivot["Total"].sum()
                            _pj_total = _pj_row["Total"].values[0] if not _pj_row.empty else 0
                            _share_row["Total"] = f"{(_pj_total/_total_denom*100):.0f}%" if _total_denom > 0 else "-"

                            # Build HTML table
                            _th_style = "padding:7px 10px;background:#f0f8ff;border:1px solid #d0dde8;font-size:0.8em;color:#444;text-align:center;white-space:nowrap;"
                            _th_left  = "padding:7px 10px;background:#f0f8ff;border:1px solid #d0dde8;font-size:0.8em;color:#444;text-align:left;white-space:nowrap;min-width:150px;"

                            _html_tbl = f"""
                            <div style='overflow-x:auto;'>
                            <table style='width:100%;border-collapse:collapse;font-size:0.82em;'>
                              <thead>
                                <tr>
                                  <th style='{_th_left}'>Grup Hotel</th>
                                  {"".join(f"<th style='{_th_style}'>{MONTH_NAMES[m]}</th>" for m in range(1,13))}
                                  <th style='{_th_style}font-weight:700;background:#daeaf8;'>Total</th>
                                </tr>
                              </thead>
                              <tbody>
                            """

                            for _, row in _pivot.iterrows():
                                _grup = row["Grup Hotel"]
                                if _grup == "Patra Jasa Group":
                                    _row_bg  = "background:rgba(27,160,226,0.07);"
                                    _val_col = "color:#1BA0E2;font-weight:600;"
                                    _badge   = "<span style='background:#e6f4fb;color:#0D7FCC;border-radius:20px;padding:1px 7px;font-size:0.78em;font-weight:600;margin-left:4px;'>PJ</span>"
                                else:
                                    _row_bg  = "background:rgba(255,140,0,0.05);"
                                    _val_col = "color:#cc6600;font-weight:600;"
                                    _badge   = "<span style='background:#fff4e6;color:#b05a00;border-radius:20px;padding:1px 7px;font-size:0.78em;font-weight:600;margin-left:4px;'>NPJ</span>"

                                _html_tbl += f"<tr style='{_row_bg}'>"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;font-weight:500;'>{_grup}{_badge}</td>"
                                for _m in range(1, 13):
                                    _v = int(row[_m]) if _pj_metric_opt == "Invoice (unique)" else f"{row[_m]:,.0f}"
                                    _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;{_val_col}'>{_v}</td>"
                                _total_v = int(row["Total"]) if _pj_metric_opt == "Invoice (unique)" else f"{row['Total']:,.0f}"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;background:#daeaf8;{_val_col}'>{_total_v}</td>"
                                _html_tbl += "</tr>"

                            # Row total gabungan
                            _html_tbl += "<tr style='background:#f5f5f5;font-weight:600;'>"
                            _html_tbl += "<td style='padding:7px 10px;border:1px solid #e0e0e0;'>Total Domestik</td>"
                            for _m in range(1, 13):
                                _col_total = _pivot[_m].sum()
                                _v = int(_col_total) if _pj_metric_opt == "Invoice (unique)" else f"{_col_total:,.0f}"
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;'>{_v}</td>"
                            _grand = _pivot["Total"].sum()
                            _grand_v = int(_grand) if _pj_metric_opt == "Invoice (unique)" else f"{_grand:,.0f}"
                            _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;background:#daeaf8;'>{_grand_v}</td>"
                            _html_tbl += "</tr>"

                            # Row % share Patra
                            _html_tbl += "<tr style='background:#fafafa;font-style:italic;'>"
                            _html_tbl += "<td style='padding:7px 10px;border:1px solid #e0e0e0;font-size:0.78em;color:#888;'>% Patra Jasa share</td>"
                            for _m in range(1, 13):
                                _html_tbl += f"<td style='padding:7px 10px;border:1px solid #e0e0e0;text-align:right;font-size:0.78em;color:#888;'>{_share_row[_m]}</td>"
                            _html_tbl += f"<td style='padding:7px 10px;border:1px solid #daeaf8;text-align:right;font-size:0.78em;color:#888;background:#daeaf8;'>{_share_row['Total']}</td>"
                            _html_tbl += "</tr>"

                            _html_tbl += "</tbody></table></div>"
                            st.markdown(_html_tbl, unsafe_allow_html=True)

                            # Download tabel
                            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                            _dl_pivot = _pivot.rename(columns={m: MONTH_NAMES[m] for m in range(1,13)})
                            _output_pj = BytesIO()
                            _dl_pivot.to_excel(_output_pj, index=False, sheet_name="Patra vs Non-Patra")
                            _output_pj.seek(0)
                            if can_download():
                                st.download_button(
                                    label="⬇️ Download Tabel",
                                    data=_output_pj,
                                    file_name=f"patra_jasa_comparison_{_pj_metric_opt.replace(' ','_')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="dl_patra_jasa_comparison")
                            else:
                                st.markdown("""
                                <div style='background:#f9f9f9;border-left:3px solid #1BA0E2;border-radius:6px;
                                padding:10px 16px;font-size:0.82em;color:#1BA0E2;display:flex;align-items:center;gap:8px;'>
                                    <span>🔒</span><span>Download tidak tersedia untuk akun ini</span>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.info(f"ℹ️ Tidak ada data bulanan untuk ditampilkan pada tahun {_pj_selected_year}.")
                    else:
                        if not _date_col_pj:
                            st.info("ℹ️ Kolom tanggal (Issue Time / Check in Date) tidak ditemukan. Tabel bulanan tidak dapat ditampilkan.")
                        elif df_pj_month_table.empty:
                            st.info(f"ℹ️ Tidak ada data untuk tahun {_pj_selected_year} setelah filter yang aktif diterapkan. Coba pilih tahun lain di atas.")
                        else:
                            st.info("ℹ️ Tabel bulanan tidak dapat ditampilkan untuk kombinasi filter saat ini.")

                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    # ── Kamus Referensi + Data Per Hotel ──
                    with st.expander(f"📋 Detail Per Hotel Patra Jasa Group — Invoice & Room Nights Bulanan{_pj_year_label}"):

                        MONTH_NAMES_KM = {
                            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                            7:"Jul",8:"Agt",9:"Sep",10:"Okt",11:"Nov",12:"Des"
                        }

                        # Data hanya hotel Patra Jasa — pakai df_pj_month_table (sudah di-scope ke tahun terpilih)
                        # agar konsisten dengan tabel perbandingan bulanan di atas dan tidak mencampur tahun.
                        df_kamus = df_pj_month_table[df_pj_month_table["Grup Hotel"] == "Patra Jasa Group"].copy()

                        # Normalisasi nama hotel ke nama kanonik dari kamus
                        _norm_to_canonical = {h.lower().strip(): h for h in PATRA_JASA_HOTELS}
                        _norm_to_canonical.update({str(standardize_name(h)).lower().strip(): h
                                                   for h in PATRA_JASA_HOTELS})
                        df_kamus["Nama Hotel Canonical"] = df_kamus["_hotel_norm"].map(_norm_to_canonical).fillna(df_kamus["Hotel Name"])

                        _km_th_h  = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:center;white-space:nowrap;"
                        _km_th_l  = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:left;white-space:nowrap;min-width:30px;"
                        _km_th_nm = "padding:6px 9px;background:#e6f4fb;border:1px solid #b8d9f0;font-size:0.78em;font-weight:600;color:#0D7FCC;text-align:left;white-space:nowrap;min-width:200px;"
                        _km_td    = "padding:5px 9px;border:1px solid #e0e0e0;text-align:right;font-size:0.8em;"
                        _km_td_l  = "padding:5px 9px;border:1px solid #e0e0e0;text-align:left;font-size:0.8em;"
                        _km_td_tot= "padding:5px 9px;border:1px solid #b8d9f0;text-align:right;font-size:0.8em;font-weight:600;background:#daeaf8;color:#0D7FCC;"

                        # ── TABEL 1: INVOICE UNIQUE PER HOTEL ──
                        st.markdown("<div style='font-size:0.85em;font-weight:600;color:#1BA0E2;margin:8px 0 6px 0;'>Invoice Unique per Hotel per Bulan</div>", unsafe_allow_html=True)

                        if _date_col_pj and "_month" in df_kamus.columns and not df_kamus.empty:
                            if "Travel Request Number" in df_kamus.columns:
                                _km_inv_agg = (
                                    df_kamus.groupby(["Nama Hotel Canonical", "_month"])["Travel Request Number"]
                                    .nunique()
                                    .reset_index(name="Nilai")
                                )
                            else:
                                _km_inv_agg = (
                                    df_kamus.groupby(["Nama Hotel Canonical", "_month"])
                                    .size().reset_index(name="Nilai")
                                )

                            _km_inv_pivot = _km_inv_agg.pivot_table(
                                index="Nama Hotel Canonical", columns="_month",
                                values="Nilai", fill_value=0
                            )
                            _km_inv_pivot.columns = [int(c) for c in _km_inv_pivot.columns]  # normalisasi tipe kolom
                            for _m in range(1, 13):
                                if _m not in _km_inv_pivot.columns:
                                    _km_inv_pivot[_m] = 0
                            _km_inv_pivot = _km_inv_pivot[[m for m in range(1, 13)]]
                            _km_inv_pivot["Total"] = _km_inv_pivot.sum(axis=1)
                            _km_inv_pivot = _km_inv_pivot.sort_values("Total", ascending=False).reset_index()

                            _html_km_inv = f"""
                            <div style='overflow-x:auto;margin-bottom:18px;'>
                            <table style='border-collapse:collapse;font-size:0.82em;min-width:100%;'>
                              <thead><tr>
                                <th style='{_km_th_l}'>No</th>
                                <th style='{_km_th_nm}'>Nama Hotel</th>
                                {"".join(f"<th style='{_km_th_h}'>{MONTH_NAMES_KM[m]}</th>" for m in range(1,13))}
                                <th style='{_km_th_h}background:#daeaf8;'>Total</th>
                              </tr></thead>
                              <tbody>
                            """
                            for _i, _row in _km_inv_pivot.iterrows():
                                _bg = "background:#fafeff;" if _i % 2 == 0 else "background:#f4faff;"
                                _html_km_inv += f"<tr style='{_bg}'>"
                                _html_km_inv += f"<td style='{_km_td_l}color:#999;'>{_i+1}</td>"
                                _html_km_inv += f"<td style='{_km_td_l}font-weight:500;'>{_row['Nama Hotel Canonical']}</td>"
                                for _m in range(1, 13):
                                    _v = int(_row[_m])
                                    _col_style = _km_td if _v > 0 else _km_td + "color:#ccc;"
                                    _html_km_inv += f"<td style='{_col_style}'>{_v if _v > 0 else '—'}</td>"
                                _html_km_inv += f"<td style='{_km_td_tot}'>{int(_row['Total']):,}</td>"
                                _html_km_inv += "</tr>"

                            # Baris total kolom
                            _html_km_inv += f"<tr style='background:#e6f4fb;font-weight:700;'>"
                            _html_km_inv += f"<td style='{_km_td_l}'></td>"
                            _html_km_inv += f"<td style='{_km_td_l}font-weight:700;color:#0D7FCC;'>Total</td>"
                            for _m in range(1, 13):
                                _col_sum = int(_km_inv_pivot[_m].sum())
                                _html_km_inv += f"<td style='{_km_th_h}'>{_col_sum:,}</td>"
                            _html_km_inv += f"<td style='{_km_th_h}background:#c8e0f5;'>{int(_km_inv_pivot['Total'].sum()):,}</td>"
                            _html_km_inv += "</tr>"

                            _html_km_inv += "</tbody></table></div>"
                            st.markdown(_html_km_inv, unsafe_allow_html=True)
                        else:
                            st.info("Data invoice per hotel tidak tersedia.")

                        # ── TABEL 2: ROOM NIGHTS PER HOTEL ──
                        st.markdown("<div style='font-size:0.85em;font-weight:600;color:#1BA0E2;margin:14px 0 6px 0;'>Room Nights per Hotel per Bulan</div>", unsafe_allow_html=True)

                        if _date_col_pj and "_month" in df_kamus.columns and "Number of Rooms Night" in df_kamus.columns and not df_kamus.empty:
                            _km_rn_agg = (
                                df_kamus.groupby(["Nama Hotel Canonical", "_month"])["Number of Rooms Night"]
                                .sum().reset_index(name="Nilai")
                            )

                            _km_rn_pivot = _km_rn_agg.pivot_table(
                                index="Nama Hotel Canonical", columns="_month",
                                values="Nilai", fill_value=0
                            )
                            _km_rn_pivot.columns = [int(c) for c in _km_rn_pivot.columns]  # normalisasi tipe kolom
                            for _m in range(1, 13):
                                if _m not in _km_rn_pivot.columns:
                                    _km_rn_pivot[_m] = 0
                            _km_rn_pivot = _km_rn_pivot[[m for m in range(1, 13)]]
                            _km_rn_pivot["Total"] = _km_rn_pivot.sum(axis=1)
                            _km_rn_pivot = _km_rn_pivot.sort_values("Total", ascending=False).reset_index()

                            _html_km_rn = f"""
                            <div style='overflow-x:auto;margin-bottom:12px;'>
                            <table style='border-collapse:collapse;font-size:0.82em;min-width:100%;'>
                              <thead><tr>
                                <th style='{_km_th_l}'>No</th>
                                <th style='{_km_th_nm}'>Nama Hotel</th>
                                {"".join(f"<th style='{_km_th_h}'>{MONTH_NAMES_KM[m]}</th>" for m in range(1,13))}
                                <th style='{_km_th_h}background:#daeaf8;'>Total</th>
                              </tr></thead>
                              <tbody>
                            """
                            for _i, _row in _km_rn_pivot.iterrows():
                                _bg = "background:#fafeff;" if _i % 2 == 0 else "background:#f4faff;"
                                _html_km_rn += f"<tr style='{_bg}'>"
                                _html_km_rn += f"<td style='{_km_td_l}color:#999;'>{_i+1}</td>"
                                _html_km_rn += f"<td style='{_km_td_l}font-weight:500;'>{_row['Nama Hotel Canonical']}</td>"
                                for _m in range(1, 13):
                                    _v = _row[_m]
                                    _v_fmt = f"{_v:,.0f}" if _v > 0 else "—"
                                    _col_style = _km_td if _v > 0 else _km_td + "color:#ccc;"
                                    _html_km_rn += f"<td style='{_col_style}'>{_v_fmt}</td>"
                                _html_km_rn += f"<td style='{_km_td_tot}'>{_row['Total']:,.0f}</td>"
                                _html_km_rn += "</tr>"

                            # Baris total kolom
                            _html_km_rn += f"<tr style='background:#e6f4fb;font-weight:700;'>"
                            _html_km_rn += f"<td style='{_km_td_l}'></td>"
                            _html_km_rn += f"<td style='{_km_td_l}font-weight:700;color:#0D7FCC;'>Total</td>"
                            for _m in range(1, 13):
                                _col_sum = _km_rn_pivot[_m].sum()
                                _html_km_rn += f"<td style='{_km_th_h}'>{_col_sum:,.0f}</td>"
                            _html_km_rn += f"<td style='{_km_th_h}background:#c8e0f5;'>{_km_rn_pivot['Total'].sum():,.0f}</td>"
                            _html_km_rn += "</tr>"

                            _html_km_rn += "</tbody></table></div>"
                            st.markdown(_html_km_rn, unsafe_allow_html=True)
                        else:
                            st.info("Data room nights per hotel tidak tersedia (kolom 'Number of Rooms Night' tidak ditemukan).")

    # ======================================
    # DISCLAIMER + FOOTER
    # ======================================
    st.markdown("""
    <div style="background:white;padding:20px;border-radius:4px;border-left:4px solid #1BA0E2;font-size:0.9em;">
        <b>Disclaimer &amp; Compliance Notice</b><br><br>
        Aplikasi ini disediakan untuk tujuan analisis internal. Output yang dihasilkan tidak bersifat final,
        tidak mengikat, dan harus melalui proses validasi serta persetujuan sesuai kebijakan perusahaan yang berlaku.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='divider' style='margin-top:50px;'></div>
    <div style='text-align:center;padding:25px;color:#888888;font-size:0.85em;'>
        © 2026 Dikembangkan oleh 
        <a href="https://www.linkedin.com/in/rifyalt/" target="_blank" style='color:#1BA0E2;text-decoration:none;font-weight:500;'>
            Rifyal Tumber
        </a> · MTRAX Travel Analytics
    </div>
    """, unsafe_allow_html=True)


# ===============================
# ROUTING — WAJIB PALING BAWAH
# ── DIMODIFIKASI: tambah pengecekan pending_2fa + pemulihan sesi dari cookie ──
# ===============================
_try_restore_session_from_cookie()  # pulihkan login kalau cookie sesi masih valid (mis. setelah reload halaman)

# Mode tanpa login (sementara): [access] login_required = false di secrets.toml
if not login_required() and not st.session_state.get("authenticated"):
    guest_login()
# Login diaktifkan lagi: sesi tamu yang masih terbuka langsung diakhiri
if login_required() and st.session_state.get("username") == GUEST_USERNAME:
    for _k in list(st.session_state.keys()):
        del st.session_state[_k]
    init_login_security()
    st.session_state.authenticated = False

if not st.session_state.get("authenticated"):

    if st.session_state.get("pending_2fa"):
        # ── Password sudah benar, tunggu OTP ─────────────────────────
        username = st.session_state.pending_user
        totp_secrets = _load_totp_secrets()

        # Setiap user punya secret TOTP sendiri di secrets.toml [totp][<username>]
        enrolled_via_toml = bool(_own_totp_secret(username))
        if USERS.get(username, {}).get("role") == "Demo" and not enrolled_via_toml:
            # Demo tanpa secret sendiri tidak memakai 2FA
            st.session_state.pending_2fa = False
            st.rerun()

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
    if login_required():
        check_session_timeout()
    main_app()
