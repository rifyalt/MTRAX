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


def _html_frame(html: str, height: int = 0):
    """Render HTML+JS di iframe. Pakai st.iframe (Streamlit baru); jatuh ke components.html bila belum ada.
    `st.components.v1.html` dijadwalkan dihapus oleh Streamlit, jadi semua pemanggilan lewat fungsi ini."""
    if hasattr(st, "iframe"):
        return st.iframe(html, height=max(int(height), 1))
    import streamlit.components.v1 as _components
    return _components.html(html, height=height, scrolling=False)

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
            st.warning("Sesi berakhir. Silakan login lagi.")
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
    max_age = int(SESSION_TIMEOUT)
    _html_frame(
        f"""<script>
        document.cookie = "{_SESSION_COOKIE_NAME}={token}; max-age={max_age}; path=/; SameSite=Lax";
        </script>""",
        height=0
    )

def _clear_session_cookie():
    """Hapus cookie sesi di browser (dipanggil saat logout / timeout)."""
    _html_frame(
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
        greet = "Selamat pagi"
    elif hour < 15:
        greet = "Selamat siang"
    elif hour < 18:
        greet = "Selamat sore"
    else:
        greet = "Selamat malam"

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

# ---------------------------------------------------------------------
# CORPORATE RATE DARI GOOGLE DRIVE
#   secrets.toml:
#     [corp_rate]
#     folder_id = "<ID folder Google Drive berisi MasterData corp rate>"
#   Folder harus dibagikan (Viewer) ke email service account di [gcp_service_account].client_email.
#   Dipakai file Excel / Google Sheet TERBARU di folder itu (utamakan yang namanya memuat
#   "corp" atau "master"). Dicek ulang tiap 10 menit; tombol "Muat ulang" untuk langsung.
# ---------------------------------------------------------------------
CORP_DRIVE_TTL = 10 * 60


def _corp_folder_id() -> str:
    try:
        return str(dict(st.secrets.get("corp_rate", {})).get("folder_id", "") or "").strip()
    except Exception:
        return ""


@st.cache_data(ttl=CORP_DRIVE_TTL, max_entries=4, show_spinner=False)
def _corp_from_drive(folder_id: str):
    import cost_avoidance as _ca
    if not _DRIVE_ID_RE.fullmatch(folder_id):
        raise ValueError("folder_id corp rate tidak valid.")
    items = _collect_excel_files(_drive_service(), folder_id, max_depth=1)
    if not items:
        raise FileNotFoundError("Folder Google Drive tidak berisi file Excel/Google Sheet, atau belum dibagikan "
                                "ke service account aplikasi.")

    def rank(it):
        n = it["name"].lower()
        return (("corp" in n) or ("master" in n), it.get("modifiedTime", ""))
    item = max(items, key=rank)
    df = _ca.read_corp_file(_download_to_memory(_drive_service(), item))
    updated = pd.to_datetime(item.get("modifiedTime"), utc=True, errors="coerce")
    if pd.notna(updated):
        updated = updated.tz_convert("Asia/Jakarta").tz_localize(None).to_pydatetime()
    else:
        updated = None
    return {"df": df, "name": item["name"], "updated": updated, "origin": "drive",
            "sig": f"drive:{item['id']}:{item.get('modifiedTime', '')}", "error": None}


def corp_source():
    """Sumber corp rate untuk Cost Avoidance & Laporan. None bila folder Drive tidak diatur."""
    fid = _corp_folder_id()
    if not fid:
        return None
    try:
        src = dict(_corp_from_drive(fid))
    except Exception as e:
        msg = str(e)
        if "404" in msg or "403" in msg or "notFound" in msg or "insufficient" in msg.lower():
            try:
                sa = dict(st.secrets["gcp_service_account"]).get("client_email", "service account aplikasi")
            except Exception:
                sa = "service account aplikasi"
            msg = f"Folder tidak bisa diakses. Bagikan folder (Viewer) ke {sa}."
        src = {"df": None, "origin": "drive", "error": msg}
    src["refresh"] = _corp_from_drive.clear
    return src


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
    df["_alias"] = df["_hotel_raw"].map(PROPERTY_MANUAL_ALIAS).astype(object).fillna(df["_hotel_raw"])
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


def _umap(series, func):
    """series.map(func) tetapi func dihitung sekali per nilai unik (jauh lebih cepat untuk puluhan ribu baris)."""
    from cost_avoidance import umap
    return umap(series, func)


def get_property_merge_map(df: pd.DataFrame, hotel_col: str = "Hotel Name",
                           city_col: str = "City") -> pd.DataFrame:
    """Bangun peta penggabungan properti dari df (hasil di-cache)."""
    if hotel_col not in df.columns:
        return pd.DataFrame(columns=["_hotel_raw", "_city_key", "Hotel / Properti"])
    tmp = pd.DataFrame({
        "_hotel_raw": df[hotel_col].fillna("").astype(str).str.strip(),
        "_city_key": (_umap(df[city_col], _normalize_city_key)
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
    out["_city_key"] = (_umap(out[city_col], _normalize_city_key)
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
        st.markdown(_lbl.format(f"Dari · {date_col}"), unsafe_allow_html=True)
        d_from = st.date_input("Dari", min_value=mn, max_value=mx, key=k_from,
                               format="YYYY/MM/DD", label_visibility="collapsed")
    with col_to:
        st.markdown(_lbl.format("Sampai"), unsafe_allow_html=True)
        d_to = st.date_input("Sampai", min_value=mn, max_value=mx, key=k_to,
                             format="YYYY/MM/DD", label_visibility="collapsed")

    if d_from > d_to:
        st.warning("Tanggal 'Dari' lebih besar dari 'Sampai'. Filter tanggal tab ini diabaikan.")
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
    """Super Admin & Admin: unggah Excel, kelola corp rate & target manual renegosiasi."""
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
    # Streamlit baru menerima data berupa fungsi (Excel dibuat saat tombol diklik). Versi lama: panggil langsung.
    _deferred_ok = "callable" in (getattr(_orig_dg, "__doc__", "") or "")

    def _prep_data(args, kwargs):
        if "data" in kwargs and callable(kwargs["data"]) and not _deferred_ok:
            kwargs["data"] = kwargs["data"]()
        elif len(args) > 1 and callable(args[1]) and not _deferred_ok:
            args = (args[0], args[1]()) + tuple(args[2:])
        return args, kwargs

    def _log(res, args, kwargs):
        if res:
            name = kwargs.get("file_name") or (args[2] if len(args) > 2 else "")
            audit("unduh", file=name)
        return res

    def _dg_guard(self, *args, **kwargs):
        if not can_download():
            return False
        kwargs.setdefault("icon", ":material/download:")
        args, kwargs = _prep_data(args, kwargs)
        return _log(_orig_dg(self, *args, **kwargs), args, kwargs)

    def _st_guard(*args, **kwargs):
        if not can_download():
            return False
        kwargs.setdefault("icon", ":material/download:")
        args, kwargs = _prep_data(args, kwargs)
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


def _as_bool(v, default=True) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("false", "0", "no", "off", "tidak", "n"):
        return False
    if s in ("true", "1", "yes", "on", "ya", "y"):
        return True
    return default


def login_required() -> bool:
    """Baca saklar login. Urutan: [access] login_required -> login_required di level teratas
    secrets.toml -> variabel lingkungan MTRAX_LOGIN_REQUIRED. Default: login wajib."""
    try:
        acc = dict(st.secrets.get("access", {}))
        if "login_required" in acc:
            return _as_bool(acc["login_required"])
    except Exception:
        pass
    try:
        if "login_required" in st.secrets:
            return _as_bool(st.secrets["login_required"])
    except Exception:
        pass
    try:  # baris yang tidak sengaja masuk ke bagian lain (mis. ditempel di akhir file di bawah [demo])
        for _name, _sec in st.secrets.items():
            if hasattr(_sec, "keys") and "login_required" in _sec:
                return _as_bool(_sec["login_required"])
    except Exception:
        pass
    env = os.environ.get("MTRAX_LOGIN_REQUIRED")
    if env is not None:
        return _as_bool(env)
    return True


def guest_role() -> str:
    r = ""
    try:
        r = str(dict(st.secrets.get("access", {})).get("guest_role", "") or "")
    except Exception:
        pass
    r = r or os.environ.get("MTRAX_GUEST_ROLE", "Viewer")
    match = {x.lower(): x for x in _VALID_ROLES}.get(r.strip().lower())
    return match or "Viewer"


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
        qr_note = '<div class="qr-missing">Install <code>qrcode[pil]</code> atau <code>segno</code> untuk QR code</div>'

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

    _html_frame(html_content, height=540)

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
            _verify = st.form_submit_button("Verifikasi & masuk", width="stretch", type="primary")
        with col_c:
            _back = st.form_submit_button("← Kembali", width="stretch")

    if _back:
        st.session_state.pending_2fa  = False
        st.session_state.pending_user = ""
        st.rerun()
    elif _verify:
        if _totp_valid(secret, otp_input):
            st.session_state.totp_enrolled[username] = True
            _finish_login(username)
        else:
            st.error("Kode salah atau sudah kedaluwarsa. Coba lagi.")


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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'Inter', system-ui, sans-serif;
  background: #F0F0F0;
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
  box-shadow: 0 4px 32px rgba(20,20,20,0.08), 0 1px 4px rgba(20,20,20,0.05);
  overflow: hidden;
}}

.card-header {{
  background: linear-gradient(135deg, #0A5D96 0%, #0D7FCC 55%, #1BA0E2 100%);
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
  background: #98EA16;
  box-shadow: none;
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
.header-sub em {{ color: #E2F871; font-style: normal; font-weight: 500; }}

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
  font-family: 'Roboto Mono', monospace;
  font-size: 1.0em;
  font-weight: 600;
  color: #0D7FCC;
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
  background: #F0F0F0;
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
        <div class="lock-circle"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg></div>
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
        
        <div class="hint-text">
          Buka <strong>Google Authenticator</strong> &rarr; cari akun <em>MTRAX:{username}</em> &rarr; masukkan <strong>6 digit kode</strong> yang ditampilkan.
        </div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""

    _html_frame(html_content, height=420)

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
        font-family: 'Roboto Mono', monospace !important;
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
            _verify = st.form_submit_button("Verifikasi & masuk", width="stretch", type="primary")
        with col_c:
            _back = st.form_submit_button("← Kembali", width="stretch")

    if _back:
        st.session_state.pending_2fa  = False
        st.session_state.pending_user = ""
        st.rerun()
    elif _verify:
        if _totp_valid(secret, otp_input):
            _finish_login(username)
        else:
            st.error("Kode salah atau sudah kedaluwarsa. Coba lagi.")
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
    st.success("Login berhasil. Selamat datang.")
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
        st.error(f"Terlalu banyak percobaan gagal. Coba lagi dalam {remaining_lock} detik.")
        st.stop()

    # Load logo
    def _b64_img(p):
        try:
            with open(p, "rb") as f: return base64.b64encode(f.read()).decode()
        except Exception: return ""

    wm_b64 = ui.wordmark_b64()
    wordmark = (f'<img class="lg-wordmark" src="data:image/png;base64,{wm_b64}" alt="MTRAX" />' if wm_b64
                else '<div class="lg-h" style="letter-spacing:.12em">MTRAX</div>')

    # Palet mengikuti identitas MTRAX: lime -> hijau toska -> biru -> biru tua
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --lime:#E2F871; --teal:#1494C6; --blue:#1BA0E2; --blue-d:#0C79C2; --navy:#0A5D96;
            --ink:#434343; --muted:#6B6B6B; --line:#DEDEDE; }
    *, *::before, *::after { font-family:'Inter',sans-serif !important; box-sizing:border-box; }
    [data-testid="stIconMaterial"] {
        font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
        font-weight:normal !important; letter-spacing:normal !important; text-transform:none !important;
        -webkit-font-feature-settings:'liga' !important; font-feature-settings:'liga' !important;
        font-size:1.15rem !important;
    }
    html, body { height:100%; overflow:hidden; }
    .stApp { background:#FFFFFF !important; overflow:hidden; }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stSidebar"], #MainMenu, footer { display:none !important; }
    [data-testid="stAppViewContainer"], section[data-testid="stMain"] { padding:0 !important; overflow:hidden !important; }
    .block-container { padding:0 !important; max-width:100% !important; overflow:hidden !important; }
    ::-webkit-scrollbar { display:none !important; }
    [data-testid="stHorizontalBlock"] { gap:0 !important; }
    [data-testid="stHorizontalBlock"] > div { padding:0 !important; }
    [data-testid="stHorizontalBlock"] > div:first-child { display:none !important; }

    /* ── Panel kiri: identitas, gradasi lembut ── */
    .lg-brand-panel {
        position:fixed; inset:0 auto 0 0; width:56%; z-index:1; overflow:hidden;
        background: linear-gradient(140deg, #E2F871 0%, #9FDCB0 24%, #1BA0E2 56%, #1494C6 74%, #0D7FCC 90%, #0A5D96 100%);
        display:flex; flex-direction:column; justify-content:space-between; padding:44px 56px 40px;
    }
    .lg-brand-panel::after {           /* satu bidang halus agar tidak datar, tanpa ornamen */
        content:''; position:absolute; right:-18%; bottom:-30%; width:70%; height:80%;
        background: radial-gradient(closest-side, rgba(255,255,255,.14), transparent); pointer-events:none;
    }
    .lg-top { display:flex; align-items:center; gap:12px; position:relative; z-index:2; }
    .lg-mark { width:40px; height:40px; border-radius:10px; background:#FFFFFF;
               display:flex; align-items:center; justify-content:center;
               box-shadow:0 1px 2px rgba(20,20,20,.10); }
    .lg-mark img { width:26px; height:auto; display:block; }
    .lg-fallback { font-weight:800; color:var(--blue-d); font-size:18px; }
    .lg-name { font-size:13px; font-weight:700; letter-spacing:.22em; color:#434343; }
    .lg-hero { position:relative; z-index:2; max-width:460px; }
    .lg-rule { width:40px; height:3px; border-radius:2px; background:#FFFFFF; opacity:.9; margin-bottom:22px; }
    .lg-title { font-size:40px; line-height:1.12; font-weight:800; letter-spacing:-.02em; color:#FFFFFF !important; margin:0 0 18px !important; padding:0 !important; text-shadow:0 1px 12px rgba(10,93,150,.25); }
    .lg-sub { font-size:15px; line-height:1.6; color:rgba(255,255,255,.86); margin:0; }
    .lg-foot { position:relative; z-index:2; font-size:12px; color:rgba(255,255,255,.75); }

    /* ── Panel kanan: putih polos ── */
    .lg-form-panel { position:fixed; inset:0 0 0 56%; background:#FFFFFF; z-index:1; }
    .lg-copy { position:fixed; left:calc(56% + 56px); bottom:24px; z-index:5; font-size:11.5px; color:#8A8A8A; }
    .lg-copy a { color:inherit; text-decoration:none; }
    .lg-copy a:hover { color:var(--blue-d); }

    [data-testid="stHorizontalBlock"] > div:last-child {
        position:fixed !important; top:50% !important; left:78% !important; transform:translate(-50%,-50%) !important;
        z-index:1000 !important; width:360px !important; max-width:360px !important;
        background:transparent !important; box-shadow:none !important; border:none !important; padding:0 !important;
    }
    .lg-wordmark { height:40px; width:auto; display:block; margin:0 0 36px; }
    .lg-h { font-size:26px; font-weight:700; color:var(--ink); letter-spacing:-.01em; margin:0 0 6px; }
    .lg-p { font-size:14px; color:var(--muted); margin:0 0 32px; }
    .lg-label { font-size:12px; font-weight:600; color:var(--ink); margin:0 0 6px; }
    .lg-label.pw { margin-top:18px; }

    .stTextInput label { display:none !important; }
    .stTextInput > div { border:none !important; background:transparent !important; }
    .stTextInput > div > div { border:1px solid var(--line) !important; border-radius:8px !important;
                               background:#FFFFFF !important; transition:border-color .15s, box-shadow .15s; }
    .stTextInput > div > div:focus-within { border-color:var(--blue) !important; box-shadow:0 0 0 3px rgba(27,160,226,.16) !important; }
    .stTextInput input { padding:12px 14px !important; font-size:14px !important; color:var(--ink) !important;
                         background:transparent !important; }
    .stTextInput input::placeholder { color:#9A9A9A !important; }
    .stTextInput button { background:transparent !important; border:none !important; color:#8A8A8A !important; }
    .stTextInput button:hover { color:var(--blue-d) !important; }

    div[data-testid="stFormSubmitButton"] > button, div[data-testid="stButton"] > button {
        width:100% !important; margin-top:26px !important; padding:12px 0 !important;
        background:var(--blue-d) !important; color:#FFFFFF !important; border:none !important; border-radius:8px !important;
        font-size:14px !important; font-weight:600 !important; letter-spacing:.02em !important;
        box-shadow:none !important; transition:background .15s !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover, div[data-testid="stButton"] > button:hover { background:var(--navy) !important; }
    div[data-testid="stFormSubmitButton"] > button p { font-weight:600 !important; }
    div[data-testid="stForm"] { border:none !important; padding:0 !important; }
    .lg-accent { height:3px; width:48px; border-radius:2px; margin:0 0 28px;
                 background:linear-gradient(90deg, var(--lime), var(--blue) 60%, var(--blue-d)); }
    .lg-note { font-size:12px; color:var(--muted); margin-top:18px; line-height:1.5; }
    .stAlert { border-radius:8px !important; font-size:13px !important; margin-top:10px !important; }

    @media (max-width: 900px) {
        .lg-brand-panel { display:none; }
        .lg-form-panel { inset:0; }
        .lg-copy { left:24px; }
        [data-testid="stHorizontalBlock"] > div:last-child { left:50% !important; width:88vw !important; max-width:360px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="lg-brand-panel">
        <div class="lg-top"></div>
        <div class="lg-hero">
            <div class="lg-rule"></div>
            <h1 class="lg-title">Analitik<br>Perjalanan Dinas</h1>
            <p class="lg-sub">Platform analitik perjalanan dinas —
               spend hotel, kepatuhan corporate rate, dan penghematan dalam satu tempat.</p>
        </div>
        <div class="lg-foot">Tanpa data identitas pekerja · setiap akses tercatat</div>
    </div>
    <div class="lg-form-panel"></div>
    <div class="lg-copy"><a href="https://www.linkedin.com/in/rifyalt/" target="_blank">© 2026 MTRAX · PT Mitra Tours And Travel</a> · v2026.11.25-PDP</div>
    """, unsafe_allow_html=True)

    _dummy, col_form = st.columns(2)
    with col_form:
        st.markdown(f"""
        {wordmark}
        <div class="lg-h">Selamat datang kembali</div>
        <div class="lg-p">Masuk ke akun MTRAX Anda.</div>
        <div class="lg-label">Username</div>
        """, unsafe_allow_html=True)

        # Dibungkus st.form supaya menekan Enter di kolom Username/Password
        # langsung men-submit login (form_submit_button otomatis ter-trigger oleh Enter),
        # tanpa harus mengklik tombol LOG IN secara manual.
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("_u", placeholder="Masukkan username",
                                     label_visibility="collapsed", key="login_username")
            st.markdown('<div class="lg-label pw">Password</div>', unsafe_allow_html=True)
            password = st.text_input("_p", type="password", placeholder="Masukkan password",
                                     label_visibility="collapsed", key="login_password")
            submitted = st.form_submit_button("Masuk", width="stretch")
        st.markdown('<div class="lg-note">Akun dengan 2FA akan diminta kode dari aplikasi authenticator.</div>',
                    unsafe_allow_html=True)

        if submitted:
            if not validate_username(username):
                st.error("Format username tidak valid.")
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
                st.error("Terlalu banyak percobaan gagal. Akun dikunci sementara.")
            else:
                remaining_attempts = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"Login gagal. Sisa {remaining_attempts} kali percobaan.")

# ======================================
# MAIN APP — tampilan mengikuti preview "Dashboard final"
# ======================================
import ui_kit as ui

PATRA_JASA_HOTELS = [
    "The Patra Bali Resort & Villas", "Patra Semarang Hotel & Convention", "Patra Cirebon Hotel & Convention",
    "Patra Malioboro Hotel", "Patra Dumai Hotel", "Patra Bandung Hotel", "Patra Jakarta Hotel",
    "Patra Anyer Hotel", "Patra Parapat Hotel",
]
DOMESTIC_ALIAS = ("INDONESIA", "ID", "IDN")
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ---------------------------------------------------------------- data
def prepare_data(df: pd.DataFrame):
    """Bersihkan data sekali per pemuatan: angka, kurs, tanggal, nama hotel, kota, peta properti.
    Return (df, peta properti, info kurs, peringatan)."""
    d = df.copy()
    fx_info, warn = {}, ""
    if "Invoice Amount" in d.columns:
        d["Invoice Amount"] = pd.to_numeric(
            d["Invoice Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce")
        d, fx_info = convert_currency_to_idr(d)
    if "Number of Rooms Night" in d.columns:
        d["Number of Rooms Night"] = pd.to_numeric(d["Number of Rooms Night"], errors="coerce")
    for c in ("Check in Date", "Check out Date", "Issue Time"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce", dayfirst=True)
    try:
        d = standardize_hotel_columns(d)
    except Exception as e:
        warn = f"Penyeragaman nama hotel dilewati: {e}"
    for c in ("City", "Country"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
            d.loc[d[c].isin(["Nan", "None", ""]), c] = np.nan
    pmap = get_property_merge_map(d, hotel_col="Hotel Name", city_col="City")
    if "Hotel Name" in d.columns:   # kolom gabungan properti dihitung sekali, bukan di setiap interaksi
        d = apply_property_merge(d, pmap, hotel_col="Hotel Name", city_col="City")
    return d, pmap, fx_info, warn


@st.cache_data(max_entries=6, show_spinner=False)
def _prepare_cached(sig: str, _raw: pd.DataFrame):
    """prepare_data dibagi antar sesi & pengguna selama data yang ditarik sama."""
    return prepare_data(_raw)


def _market(df, choice):
    if "Country" not in df.columns or choice in ("Semua", None):
        return df
    dom = df["Country"].astype(str).str.strip().str.upper().isin(DOMESTIC_ALIAS)
    return df[dom] if choice in ("Domestik", "Indonesia") else df[~dom]


def _date_col(df):
    return next((c for c in ("Issue Time", "Check in Date", "Check out Date") if c in df.columns), None)


def _period_label(df):
    dc = _date_col(df)
    if not dc or df.empty:
        return "Semua periode"
    s = pd.to_datetime(df[dc], errors="coerce").dropna()
    return f"{s.min():%d %b %Y} – {s.max():%d %b %Y}" if not s.empty else "Semua periode"


def _cols(df, cols):
    return df[[c for c in cols if c in df.columns]]


def _xlsx(df, sheet, **fmt):
    return excel_bytes_with_formats(df, sheet[:31], **fmt)


def _dl(label, df, sheet, fname, key, width="content", **fmt):
    if can_download():
        snap = df.copy()
        st.download_button(label, lambda: _xlsx(snap, sheet, **fmt).getvalue(),
                           file_name=f"{fname}_{datetime.now():%Y%m%d}.xlsx",
                           mime=XLSX_MIME, key=key, width=width)


def _fig(fig, key, height=None, **after):
    fig.update_layout(**ui.PLOTLY_LAYOUT)
    if height:
        fig.update_layout(height=height)
    if after:                                   # pengaturan yang harus menang atas gaya bawaan
        fig.update_layout(**after)
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def _hbar(labels, values, key, color=ui.BLUE, top_highlight=None, height=None, text=None):
    colors = [color if (top_highlight is None or i < top_highlight) else ui.BLUE_XL for i in range(len(values))]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           text=text if text is not None else [ui.fmt_int(v) for v in values],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                      showlegend=False, margin=dict(l=10, r=60, t=10, b=20))
    _fig(fig, key, height or max(260, 26 * len(values) + 40),
         xaxis=dict(showgrid=True, gridcolor="#EDEDED", range=[0, (max(values) if len(values) else 1) * 1.22]))


def _pareto_fig(names, values, top_n, key, value_title, max_show=100):
    total = float(np.nansum(values)) or 1.0
    cum = np.cumsum(values) / total * 100
    show = min(len(names), max_show)       # kumulatif % tetap dihitung dari SEMUA hotel/kota
    colors = [ui.BLUE if i < top_n else ui.BLUE_XL for i in range(show)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names[:show], y=values[:show], marker_color=colors, name=value_title,
                         hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=names[:show], y=cum[:show], yaxis="y2", mode="lines+markers", name="Kumulatif %",
                             line=dict(color=ui.NAVY, width=2.5), marker=dict(size=4),
                             hovertemplate="Kumulatif %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=80, yref="y2", line_dash="dash", line_color=ui.RED, opacity=0.7,
                  annotation_text="80%", annotation_position="right")
    fig.update_layout(yaxis=dict(title=value_title, gridcolor="#EDEDED"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False),
                      xaxis=dict(showticklabels=False), legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                      bargap=0.15)
    _fig(fig, key, 380)
    if len(names) > show:
        st.caption(f"Grafik menampilkan {show} teratas dari {len(names)}; persentase kumulatif dihitung dari semua.")


def _key_column(df, dim, merge, pmap):
    if dim == "Hotel":
        if merge:
            if "Hotel / Properti" in df.columns:
                return df, "Hotel / Properti"
            d = apply_property_merge(df, pmap, hotel_col="Hotel Name", city_col="City")
            return d, "Hotel / Properti"
        d = df.copy()
        d["Hotel / Properti"] = d["Hotel Name"].astype(str).str.strip()
        return d, "Hotel / Properti"
    return df, "City"


# ---------------------------------------------------------------- sidebar & header
def _client_ip() -> str:
    """IP publik pengguna dari header proxy (hanya ditampilkan ke pengguna itu sendiri, tidak dicatat)."""
    try:
        h = st.context.headers
        fwd = h.get("X-Forwarded-For") or h.get("x-forwarded-for") or ""
        if isinstance(fwd, str) and fwd:
            return fwd.split(",")[0].strip()
        ip = getattr(st.context, "ip_address", None)
        return ip if isinstance(ip, str) else ""
    except Exception:
        return ""


@st.cache_data(ttl=120, show_spinner=False)
def _drive_ping() -> dict:
    """Cek singkat koneksi server MTRAX -> Google Drive (di-cache 2 menit)."""
    if "gcp_service_account" not in st.secrets:
        return {}
    t0 = time.time()
    try:
        _drive_service().about().get(fields="user(emailAddress)").execute()
        return {"state": "ok", "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"state": "err", "msg": type(e).__name__}


def _net_chip():
    """Chip status koneksi di header (iframe kecil yang ditempatkan di atas placeholder chip)."""
    _html_frame(ui.net_chip_html(_client_ip(), _drive_ping()), height=34)


def render_sidebar():
    with st.sidebar:
        ui.brand()
        uname = st.session_state.get("username", "")
        role = st.session_state.get("role", "")
        if uname == GUEST_USERNAME:
            # Belum ada akun pengguna: kotak nama/peran tidak ditampilkan
            st.markdown(f'<div style="margin:2px 0 12px">{ui.status_line("Mode tanpa login", "warn")}</div>',
                        unsafe_allow_html=True)
            tfa = None
        elif _own_totp_secret(uname):
            tfa = ui.status_line("2FA pribadi aktif", "ok")
        elif role == "Demo":
            tfa = ui.status_line("Akun demo", "muted")
        else:
            tfa = ui.status_line("2FA belum diaktifkan", "warn")
        if tfa is not None:
            st.markdown(f'<div class="mx-userbox"><div class="n">{ui.esc(uname)}</div>'
                        f'<div class="r">{ui.esc(role)}</div>{tfa}</div>', unsafe_allow_html=True)
        if not login_required():
            st.caption(f"Login sementara dimatikan. Aktifkan lagi: [access] login_required = true.")

        all_folders = dict(st.secrets.get("drive_folders", {}))
        if is_demo_user():
            cfg = _demo_config() or {"periods": []}
            periods = [p for p in all_folders if not cfg["periods"] or p in cfg["periods"]]
        else:
            periods = list(all_folders.keys())
        if periods:
            sel = st.selectbox("Periode data", periods, key="sb_period")
            c1, c2 = st.columns([3, 1]) if is_admin() else (st.container(), None)
            with c1:
                go_clicked = st.button("Muat Data", type="primary", width="stretch", key="sb_get")
            if c2 is not None:
                with c2:
                    if st.button(":material/refresh:", width="stretch", key="sb_refresh",
                                 help="Bersihkan cache & tarik ulang (kode TR/Booking ID ikut berganti)"):
                        load_drive_data.clear()
                        st.toast("Cache dibersihkan. Klik Muat Data.")
            if go_clicked:
                with st.spinner(f"Memuat data {sel}…"):
                    try:
                        _t0 = time.time()
                        raw = load_drive_data(all_folders[sel], ["Unnamed: 0", "Travel Request Number.1"])
                        if is_demo_user():
                            raw = prepare_demo_dataset(raw)
                            st.session_state._demo_masked = True
                            audit("demo_muat_data", periode=sel)
                        st.session_state.df_all = raw
                        st.session_state.data_loaded = True
                        st.session_state.data_period = sel
                        st.session_state._data_ver = st.session_state.get("_data_ver", 0) + 1
                        st.session_state._load_secs = time.time() - _t0
                        st.rerun()
                    except Exception as e:
                        st.error("Gagal memuat data." if is_demo_user() else f"Gagal memuat data: {e}")
        else:
            st.info("Belum ada folder data ([drive_folders] di Secrets).")

        _df_now = st.session_state.get("df_all")
        if st.session_state.get("data_loaded") and _df_now is not None and not _df_now.empty:
            secs = st.session_state.get("_load_secs")
            st.markdown(ui.load_card(len(_df_now), str(st.session_state.get("data_period", "")), secs),
                        unsafe_allow_html=True)
            fx = st.session_state.get("_fx_info") or {}
            if fx.get("missing"):
                st.warning(f"{fx['missing']:,} booking tanpa kurs ({', '.join(fx.get('missing_currencies', []))}).")


        if is_admin():
            with st.expander("Unggah Excel (Admin)"):
                files = st.file_uploader("File transaksi", type=["xlsx", "xls"], accept_multiple_files=True,
                                         key="sb_upload", label_visibility="collapsed")
                if files and st.button("Proses file", key="sb_process"):
                    dfs = [trim_string_columns(select_columns(pd.read_excel(f))[0]) for f in files]
                    st.session_state.df_all = mask_dataframe(pd.concat(dfs, ignore_index=True))
                    st.session_state.data_loaded = True
                    st.session_state.data_period = "Upload manual"
                    st.session_state._data_ver = st.session_state.get("_data_ver", 0) + 1
                    audit("upload_manual", files=len(files))
                    st.rerun()

        st.markdown('<div class="mx-src"><b>Sumber data</b><br>13 kolom transaksi hotel, tanpa identitas pekerja. '
                    'No. TR &amp; Booking ID berupa kode acak yang berganti setiap data ditarik. '
                    'Invoice non-Rupiah dikonversi dengan kurs.</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if is_admin():
            if st.session_state.get("view") == "akses":
                if st.button("← Kembali ke dashboard", width="stretch", key="sb_back"):
                    st.session_state.view = "main"
                    st.rerun()
            elif st.button("Akun & akses", width="stretch", key="sb_akses"):
                st.session_state.view = "akses"
                st.rerun()
        if uname != GUEST_USERNAME and st.button("Keluar", width="stretch", key="sb_logout"):
            _revoke_session_token(st.session_state.get("_session_token"))
            _clear_session_cookie()
            time.sleep(0.5)
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


_ICO = {  # ikon garis sederhana (inline SVG, tanpa emoji)
    "cal": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/></svg>',
    "db": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    "user": '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/></svg>',
    "clock": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
}


def render_header():
    greet, now = get_greeting()
    uname = st.session_state.get("username", "")
    role = st.session_state.get("role", "")
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][now.weekday()]
    guest = uname == GUEST_USERNAME
    chips = [("cal", f"{hari}, {now:%d %b %Y}")] + ([] if guest else [("user", role or "-")])
    raw = st.session_state.get("df_all")
    if st.session_state.get("data_loaded") and raw is not None and not raw.empty:
        chips.insert(0, ("clock", f"Periode {st.session_state.get('data_period', '-')}"))
    chip_html = "".join(f'<span class="mx-chip">{_ICO[i]}{ui.esc(t)}</span>' for i, t in chips)
    st.markdown(
        f'<div class="mx-hero"><div class="mx-hero-l">'
        f'<div class="mx-eyebrow">Corporate Travel Analytics</div>'
        f'<div class="mx-hero-t">{greet if guest else f"{greet}, {ui.esc(uname.title())}"}</div>'
        f'<div class="mx-hero-s">Analitik transaksi hotel perjalanan dinas untuk negosiasi tarif '
        f'dan pengendalian biaya.</div></div>'
        f'<div class="mx-hero-r">{chip_html}{ui.NET_CHIP_PLACEHOLDER}</div></div>', unsafe_allow_html=True)
    _net_chip()


PAGE_ICONS = {
    "Overview": ":material/space_dashboard:", "Price Intelligence": ":material/sell:",
    "Cost Avoidance": ":material/savings:",
    "Sankey Flow": ":material/account_tree:", "Top Hotel/City": ":material/leaderboard:",
    "Cluster Analysis": ":material/hub:", "Patra Jasa": ":material/apartment:", "Reports": ":material/description:",
}


# ---------------------------------------------------------------- tab: Dashboard
_HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN_ID = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

def _last_complete_month(d):
    """Bulan lengkap terakhir: sebelum bulan data terakhir (data bulan berjalan biasanya belum lengkap)."""
    ref = d["Issue Time"].max() if "Issue Time" in d and d["Issue Time"].notna().any() else d["Check in Date"].max()
    return (pd.Timestamp(ref).to_period("M") - 1) if pd.notna(ref) else None


def _forecast_card(d):
    """Prakiraan room night 6 bulan ke depan (per bulan check-in): tren x pola musiman."""
    if not {"Check in Date", "Number of Rooms Night"}.issubset(d.columns):
        return
    with st.container(border=True, key="db_fc"):
        h1, h2 = st.columns([2, 1.3], vertical_alignment="top")
        cities = d["City"].dropna().astype(str) if "City" in d else pd.Series(dtype=str)
        opts = ["Semua kota"] + list(cities.value_counts().head(30).index)
        pick = h2.selectbox("Kota", opts, key="db_fc_city", label_visibility="collapsed")
        x = d if pick == "Semua kota" else d[d["City"].astype(str) == pick]
        last = _last_complete_month(d)
        m = (x.dropna(subset=["Check in Date"])
             .assign(P=lambda z: z["Check in Date"].dt.to_period("M"))
             .groupby("P")["Number of Rooms Night"].sum())
        if last is not None:
            m = m[m.index <= last]
        if len(m):
            m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"), fill_value=0)
        n = len(m)
        seasonal = n >= 24
        with h1:
            ui.section("Prakiraan room night 6 bulan",
                       ("Tren + pola musiman" if seasonal else "Tren saja (musiman butuh ≥ 24 bulan)")
                       + f" · {n} bulan data")
        if n < 6 or m.sum() <= 0:
            st.caption("Data belum cukup: prakiraan butuh minimal 6 bulan data check-in yang lengkap. "
                       "Muat periode data yang lebih panjang (mis. 2 tahun) untuk hasil yang lebih baik.")
            return
        t = np.arange(n)
        lg = np.log(np.clip(m.to_numpy(float), 1, None))
        mon = np.array([p.month - 1 for p in m.index])
        base = np.polyval(np.polyfit(t, lg, 1), t)
        si = np.zeros(12)
        if seasonal:
            for k in range(12):
                if (mon == k).any():
                    si[k] = np.mean((lg - base)[mon == k])
            si -= si.mean()
        coef = np.polyfit(t, lg - si[mon], 1)
        resid = lg - (np.polyval(coef, t) + si[mon])
        sd = float(np.std(resid, ddof=1)) if n > 2 else 0.1
        h = 6
        fidx = pd.period_range(m.index[-1] + 1, periods=h, freq="M")
        ft = np.arange(n, n + h)
        fm = np.polyval(coef, ft) + si[np.array([p.month - 1 for p in fidx])]
        widen = 1.28 * sd * np.sqrt(1 + np.arange(1, h + 1) / max(n, 1))
        fy, lo, hi = np.exp(fm), np.exp(fm - widen), np.exp(fm + widen)
        lab = lambda p: f"{_BULAN_ID[p.month - 1]} {p.year}"
        xs = [p.to_timestamp() for p in m.index]
        fx = [p.to_timestamp() for p in fidx]
        green = "#5FA80C"
        fmt = ui.fmt_int
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[xs[-1]] + fx + fx[::-1] + [xs[-1]],
                                 y=[m.iloc[-1]] + list(hi) + list(lo[::-1]) + [m.iloc[-1]], fill="toself", mode="lines",
                                 fillcolor="rgba(95,168,12,0.13)", line=dict(width=0), hoverinfo="skip",
                                 name="Rentang perkiraan (80%)"))
        fig.add_trace(go.Scatter(x=xs, y=m.to_numpy(), mode="lines+markers", name="Aktual",
                                 line=dict(color=ui.P1, width=2.4), marker=dict(size=5),
                                 customdata=[lab(p) for p in m.index],
                                 hovertemplate="%{customdata}<br>%{y:,.0f} room night<extra></extra>"))
        fig.add_trace(go.Scatter(x=[xs[-1]] + fx, y=[m.iloc[-1]] + list(fy), mode="lines+markers", name="Prakiraan",
                                 line=dict(color=green, width=2.6, dash="dash"),
                                 marker=dict(size=7, symbol="circle-open", line=dict(width=2)),
                                 customdata=[lab(m.index[-1])] + [f"{lab(p)}<br>rentang {fmt(a)}–{fmt(c)}"
                                                                  for p, a, c in zip(fidx, lo, hi)],
                                 hovertemplate="%{customdata}<br><b>%{y:,.0f}</b> room night<extra></extra>"))
        fig.add_vrect(x0=xs[-1], x1=fx[-1], fillcolor="rgba(95,168,12,0.05)", line_width=0)
        prev = [m.get(p - 12) for p in fidx]
        yoy = (fy.sum() / sum(prev) - 1) if all(v is not None and v > 0 for v in prev) else None
        top_txt = (f"<b>{lab(fidx[0])} – {lab(fidx[-1])}: {fmt(fy.sum())} room night</b><br>"
                   f"<span style='font-size:11px;color:#6B6B6B'>rentang {fmt(lo.sum())} – {fmt(hi.sum())}"
                   + (f" · {'+' if yoy >= 0 else ''}{yoy * 100:.0f}% vs tahun lalu" if yoy is not None else "")
                   + "</span>")
        fig.add_annotation(x=0.01, xref="paper", y=1.0, yref="paper", xanchor="left", yanchor="top", align="left",
                           text=top_txt, showarrow=False, font=dict(size=12, color=ui.INK),
                           bgcolor="rgba(255,255,255,0.9)")
        allx, alll = xs + fx, [lab(p) for p in m.index] + [lab(p) for p in fidx]
        step = max(1, -(-len(allx) // 4))
        fig.update_layout(**ui.PLOTLY_LAYOUT)
        fig.update_layout(height=320, separators=",.", hovermode="x unified", showlegend=False,
                          margin=dict(l=10, r=10, t=20, b=10),
                          xaxis=dict(tickvals=allx[::step], ticktext=alll[::step], tickangle=0, gridcolor="#F3F3F3"),
                          yaxis=dict(title="Room night per bulan", tickformat=",d", gridcolor="#EFEFEF",
                                     range=[0, float(max(hi.max(), m.max())) * 1.35]),
                          legend=dict(orientation="h", y=-0.12, x=0))
        st.plotly_chart(fig, width="stretch", key="db_fc_fig",
                        config={"displaylogo": False, "displayModeBar": False})
        tb = pd.DataFrame({"Bulan": [lab(p) for p in fidx], "Prakiraan room night": np.round(fy).astype(int),
                           "Batas bawah": np.round(lo).astype(int), "Batas atas": np.round(hi).astype(int)})
        with st.expander("Tabel & cara membaca", icon=":material/table_chart:"):
            st.dataframe(tb.style.format({c: lambda v: ui.fmt_int(v) for c in tb.columns[1:]}),
                         hide_index=True, width="stretch")
            _dl("Unduh Excel", tb, "Prakiraan room night", "prakiraan_room_night", "db_fc_dl",
                int_cols=("Prakiraan room night", "Batas bawah", "Batas atas"))
            st.caption("Garis biru = aktual · garis hijau putus-putus = prakiraan · pita hijau = rentang perkiraan 80% "
                       "(8 dari 10 kali angka sebenarnya ada di dalam rentang). Dihitung per bulan check-in; bulan "
                       "berjalan yang datanya belum lengkap tidak dipakai. Pola musiman dipakai bila data ≥ 24 bulan.")


def _checkin_heatmap(d):
    """Heatmap porsi booking menurut hari check-in x bulan check-in (gaya sama dengan 'Waktu pengajuan')."""
    if "Check in Date" not in d:
        return
    with st.container(border=True, key="db_ci"):
        ui.section("Hari mulai check-in", "Jumlah booking per hari & bulan check-in (Check in Date)")
        x = d.dropna(subset=["Check in Date"])
        if x.empty:
            st.caption("Tidak ada data check-in.")
            return
        dmap = dict(enumerate(_HARI_ID))
        x = x.assign(Hari=x["Check in Date"].dt.day_of_week.map(dmap), P=x["Check in Date"].dt.to_period("M"))
        pv = (x.groupby(["Hari", "P"]).size().unstack(fill_value=0)
              .reindex(_HARI_ID).fillna(0))
        pv = pv[list(pv.columns)[-12:]]                     # 12 bulan terakhir agar tetap terbaca
        multi_year = len({p_.year for p_ in pv.columns}) > 1
        lab = [f"{_BULAN_ID[p_.month - 1]}" + (f" {str(p_.year)[2:]}" if multi_year else "") for p_ in pv.columns]
        pct = pv / max(pv.values.sum(), 1) * 100
        fig = go.Figure(go.Heatmap(z=pv.values, x=lab, y=list(pv.index), colorscale=ui.SEQUENTIAL,
                                   text=[[ui.fmt_int(v) for v in r] for r in pv.values], texttemplate="%{text}",
                                   customdata=pct.values, showscale=False, textfont=dict(size=9.5), xgap=1.5, ygap=1.5,
                                   hovertemplate="%{y} · %{x}<br>%{z:,d} booking · %{customdata:.1f}% dari total"
                                                 "<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), separators=",.")
        _fig(fig, "db_ci_fig", 340)

def tab_dashboard(df):
    market = ui.choice("Wilayah", ["Semua", "Domestik", "Internasional"], key="db_market")
    st.markdown(f"<div style='margin-top:-6px;font-size:13px;color:{ui.MUTED}'>Periode: {_period_label(df)} · "
                f"nilai dalam Rupiah</div>", unsafe_allow_html=True)
    d = _market(df, market)
    if d.empty:
        st.info("Tidak ada data untuk pilihan ini.")
        return
    rn = d["Number of Rooms Night"].sum() if "Number of Rooms Night" in d else 0
    inv = d["Invoice Amount"].sum() if "Invoice Amount" in d else 0
    tr = d["Travel Request Number"].nunique() if "Travel Request Number" in d else len(d)
    lead_txt, lead_sub = "-", ""
    if {"Issue Time", "Check in Date"}.issubset(d.columns):
        lt = (d["Check in Date"].dt.normalize() - d["Issue Time"].dt.normalize()).dt.days
        lt = lt[lt >= 0]
        if len(lt):
            lead_txt, lead_sub = f"{lt.mean():.0f} hari", f"{(lt <= 2).mean() * 100:.0f}% last-minute"
    n_hotel = d["Hotel Name"].nunique() if "Hotel Name" in d else 0
    n_city = d["City"].nunique() if "City" in d else 0
    dec = lambda v: f"{v:.1f}".replace(".", ",")
    rate_txt, rate_unit = ("-", "")
    if rn:
        rate_txt, rate_unit = f"Rp{inv / rn / 1000:,.0f}".replace(",", "."), "rb"
    lead_unit = ""
    if lead_txt.endswith(" hari"):
        lead_txt, lead_unit = lead_txt[:-5], "hari"
    lead_sub_h = (f"<b>{ui.esc(lead_sub.split(' ')[0])}</b> last-minute" if lead_sub else "")
    ui.kpi_row([ui.kpi("Booking", ui.fmt_int(len(d)), icon="booking",
                       sub=f"<b>{dec(len(d) / max(tr, 1))}</b> booking / request", sub_html=True),
                ui.kpi("Travel request", ui.fmt_int(tr), icon="request",
                       sub=f"<b>{dec(rn / max(tr, 1))}</b> malam / request", sub_html=True),
                ui.kpi("Hotel", ui.fmt_int(n_hotel), icon="hotel",
                       sub=f"tersebar di <b>{ui.fmt_int(n_city)}</b> kota", sub_html=True),
                ui.kpi("Room night", ui.fmt_int(rn), icon="night",
                       sub=f"<b>{dec(rn / max(len(d), 1))}</b> malam / booking", sub_html=True),
                ui.kpi("Rate / malam", rate_txt, unit=rate_unit, icon="rate",
                       sub=f"total <b>{ui.esc(ui.fmt_big(inv))}</b>", sub_html=True),
                ui.kpi("Lead time", lead_txt, unit=lead_unit, icon="clock",
                       sub=lead_sub_h, sub_html=True)], 138)
    multi, multi_p = "-", None
    if "Travel Request Number" in d:
        per_tr = d.groupby("Travel Request Number").size()
        multi_p = (per_tr > 1).mean() * 100
        multi = ui.fmt_pct(multi_p, digits=0)
    bf, bf_p = "-", None
    if "Meal Plan" in d or "Room Category" in d:
        import cost_avoidance as _ca
        meal = _ca.umap(d["Meal Plan"], _ca.norm_meal) if "Meal Plan" in d else pd.Series("Tidak diketahui", index=d.index)
        if "Room Category" in d:
            meal = meal.where(meal != "Tidak diketahui", _ca.umap(d["Room Category"], _ca.norm_meal))
        known = meal != "Tidak diketahui"
        if known.any():
            bf_p = (meal[known] == "Breakfast Included").mean() * 100
            bf = ui.fmt_pct(bf_p, digits=0)
    ui.mini_row([("Kota tujuan", ui.fmt_int(n_city)),
                 ("Negara", ui.fmt_int(d["Country"].nunique() if "Country" in d else 0)),
                 ("Multi-booking", multi, multi_p), ("Breakfast included", bf, bf_p)])

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            ui.section("Kota tujuan teratas (booking)")
            if "City" in d:
                top = d["City"].value_counts().head(10)
                _hbar(list(top.index), list(top.values), "db_cities")
    with b:
        with st.container(border=True):
            ui.section("Domestik vs internasional")
            if "Country" in d:
                dom = d["Country"].astype(str).str.upper().isin(DOMESTIC_ALIAS)
                fig = go.Figure(go.Pie(labels=["Domestik", "Internasional"], values=[int(dom.sum()), int((~dom).sum())],
                                       hole=0.6, marker=dict(colors=[ui.BLUE, ui.LIME]), texttemplate="%{percent:.1%}",
                                       sort=False, domain=dict(x=[0.1, 0.9], y=[0.14, 1])))
                fig.update_layout(annotations=[dict(text=f"<b>{ui.fmt_int(len(d))}</b><br>booking", x=0.5, y=0.57,
                                                    showarrow=False, font=dict(size=14))],
                                  legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top",
                                              itemwidth=40, bgcolor="rgba(0,0,0,0)"))
                _fig(fig, "db_market_pie", 300)
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            ui.section("Tren travel request & room night bulanan", "Berdasarkan bulan Issue Time")
            if "Issue Time" in d and "Travel Request Number" in d:
                m = d.dropna(subset=["Issue Time"]).assign(Bulan=lambda x: x["Issue Time"].dt.to_period("M"))
                t = m.groupby("Bulan").agg(TR=("Travel Request Number", "nunique"),
                                           RN=("Number of Rooms Night", "sum")).sort_index()
                xs = [p_.to_timestamp() for p_ in t.index]
                lbl = [f"{_BULAN_ID[p_.month - 1]} {p_.year}" for p_ in t.index]
                green = "#5FA80C"
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=t["TR"], name="Travel request", mode="lines+markers",
                                         line=dict(color=ui.P1, width=2.5, shape="spline", smoothing=0.5),
                                         marker=dict(size=6), customdata=lbl,
                                         hovertemplate="%{y:,d} travel request<extra></extra>"))
                fig.add_trace(go.Scatter(x=xs, y=t["RN"], name="Room night", mode="lines+markers", yaxis="y2",
                                         line=dict(color=green, width=2.5, shape="spline", smoothing=0.5),
                                         marker=dict(size=6), customdata=lbl,
                                         hovertemplate="%{y:,d} room night<extra></extra>"))
                def _nice(v, n=4):          # batas atas & langkah "bulat" agar garis bantu kedua sumbu sejajar
                    import math
                    raw = max(float(v), 1.0) * 1.08 / n
                    mag = 10 ** math.floor(math.log10(raw))
                    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
                    return step * n, step
                ltop, lstep = _nice(t["TR"].max())
                rtop, rstep = _nice(t["RN"].max())
                fig.update_layout(**ui.PLOTLY_LAYOUT)
                fig.update_layout(
                    height=320, showlegend=True, hovermode="x unified", separators=",.",
                    legend=dict(orientation="h", y=1.12, x=0, xanchor="left"),
                    xaxis=dict(tickvals=xs[::-(-len(xs) // 4) or 1], ticktext=lbl[::-(-len(xs) // 4) or 1], gridcolor="#F3F3F3", tickangle=0),
                    yaxis=dict(title=dict(text="Travel request", font=dict(color=ui.P1)), gridcolor="#EDEDED",
                               tickfont=dict(color=ui.P1), range=[0, ltop], dtick=lstep, tickformat=",d"),
                    yaxis2=dict(title=dict(text="Room night", font=dict(color=green)), overlaying="y", side="right",
                                tickfont=dict(color=green), range=[0, rtop], dtick=rstep, tickformat=",d",
                                showgrid=False, tickmode="linear"),
                    margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, width="stretch", key="db_trend", config={"displaylogo": False, "displayModeBar": False})
                with st.expander("Cara membaca", icon=":material/menu_book:"):
                    pk_tr, pk_rn = t["TR"].idxmax(), t["RN"].idxmax()
                    lo_rn = t["RN"].idxmin()
                    ratio = t["RN"].sum() / max(t["TR"].sum(), 1)
                    lbl_of = lambda p_: f"{_BULAN_ID[p_.month - 1]} {p_.year}"
                    st.markdown(
                        "- **Garis biru** = jumlah travel request (TR) unik yang diajukan per bulan, dibaca di "
                        "**sumbu kiri**.\n"
                        "- **Garis hijau** = total room night dari TR yang diajukan di bulan itu, dibaca di "
                        "**sumbu kanan**. Skala kedua sumbu berbeda, jadi bandingkan **naik-turunnya**, bukan "
                        "tinggi garisnya.\n"
                        "- Bila kedua garis naik-turun bersamaan, kebutuhan menginap mengikuti jumlah perjalanan. "
                        "Bila garis hijau naik sendiri, perjalanan yang ada makin lama menginapnya.\n"
                        "- Bulan dihitung dari **tanggal pengajuan (Issue Time)**, bukan tanggal menginap; bulan "
                        "berjalan biasanya masih rendah karena datanya belum lengkap.\n"
                        "- Arahkan kursor ke satu bulan untuk melihat kedua angkanya sekaligus.")
                    ui.note(f"<b>Ringkasnya:</b> puncak travel request di <b>{lbl_of(pk_tr)}</b> "
                            f"({ui.fmt_int(t.loc[pk_tr, 'TR'])} TR), puncak room night di <b>{lbl_of(pk_rn)}</b> "
                            f"({ui.fmt_int(t.loc[pk_rn, 'RN'])} RN), terendah di {lbl_of(lo_rn)}. Rata-rata "
                            f"<b>{ratio:.1f}".replace(".", ",") + " room night per travel request</b>.")
    with b:
        _forecast_card(d)
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            ui.section("Waktu pengajuan travel request", "Porsi travel request per hari & jam (Issue Time)")
            if "Issue Time" in d:
                h = d.dropna(subset=["Issue Time"]).copy()
                if "Travel Request Number" in h:
                    h = h.drop_duplicates("Travel Request Number")
                dmap = dict(enumerate(_HARI_ID))
                h["Hari"] = h["Issue Time"].dt.day_of_week.map(dmap)
                h["Jam"] = pd.cut(h["Issue Time"].dt.hour, [-1, 5, 8, 11, 14, 17, 23],
                                  labels=["00–06", "06–09", "09–12", "12–15", "15–18", "18–24"])
                pv = (h.groupby(["Hari", "Jam"], observed=False).size().unstack(fill_value=0)
                      .reindex(_HARI_ID).fillna(0))
                pct = pv / max(pv.values.sum(), 1) * 100
                fig = go.Figure(go.Heatmap(z=pct.values, x=list(pct.columns), y=list(pct.index),
                                           colorscale=ui.SEQUENTIAL,
                                           text=[[f"{v:.0f}%" for v in r] for r in pct.values], texttemplate="%{text}",
                                           showscale=False, hovertemplate="%{y} · %{x}<br>%{z:.1f}%<extra></extra>"))
                fig.update_layout(yaxis=dict(autorange="reversed"))
                _fig(fig, "db_heat", 340)
    with b:
        _checkin_heatmap(d)
    st.caption("Data transaksi hotel tanpa identitas pekerja, dipakai untuk negosiasi tarif dan pengendalian biaya "
               "perjalanan dinas. Invoice non-Rupiah dikonversi dengan kurs.")


# ---------------------------------------------------------------- tab: Price Intelligence
def tab_price(df, pmap):
    ui.page_title("Price Intelligence", "Konsentrasi Pareto 80/20 · benchmark rate hotel · simulator negosiasi")
    with st.container(border=True):
        c1, c2, c3, c5, c4 = st.columns([1.3, 1.8, 1, 1.1, 1.4])
        with c1:
            wil = ui.choice("Wilayah", ["Indonesia", "Internasional"], key="pi_wil")
        d = _market(df, wil)
        with c2:
            hotels = st.multiselect("Filter hotel (opsional)", sorted(d["Hotel Name"].dropna().unique()),
                                    placeholder="Semua hotel", key="pi_hotels")
        with c3:
            dim = st.selectbox("Dimensi Pareto", ["Hotel", "Kota"], key="pi_dim")
        with c5:
            basis = st.selectbox("Basis Pareto", ["Room night", "Spend"], key="pi_basis",
                                 help="Peringkat dan kurva 80/20 dihitung dari total room night atau total spend")
        with c4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            merge = st.checkbox("Gabungkan properti yang sama", value=True, key="pi_merge", disabled=dim != "Hotel")
    if hotels:
        d = d[d["Hotel Name"].isin(hotels)]
    d = d[d["Invoice Amount"].notna()] if "Invoice Amount" in d else d.iloc[0:0]
    if d.empty:
        st.info("Tidak ada data untuk filter ini.")
        return
    d, key = _key_column(d, dim, merge, pmap)
    g = d.groupby(key).agg(Spend=("Invoice Amount", "sum"), RN=("Number of Rooms Night", "sum"),
                           Kota=("City", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                           Booking=("Travel Request Number", "nunique") if "Travel Request Number" in d else
                           ("Invoice Amount", "size")).reset_index()
    by_rn = basis == "Room night"
    mcol = "RN" if by_rn else "Spend"
    mname = "room night" if by_rn else "spend"
    fmt_m = ui.fmt_int if by_rn else ui.fmt_big
    share = "RN %" if by_rn else "Spend %"
    g = g.sort_values([mcol, "Spend" if by_rn else "RN"], ascending=False).reset_index(drop=True)
    total = g[mcol].sum()
    g["Rata-rata rate / malam"] = g["Spend"] / g["RN"].replace(0, np.nan)
    g[share] = g[mcol] / total * 100 if total else 0.0
    g["Kumulatif %"] = g[share].cumsum().clip(upper=100)
    n20 = max(1, int(len(g) * 0.2))
    top_m = g[mcol].head(n20).sum()
    ent = "properti" if dim == "Hotel" else "kota"
    ui.kpi_row([ui.kpi(f"Total {mname} · {wil}", fmt_m(total), _period_label(d), dark=True),
                ui.kpi("20% teratas", ui.fmt_int(n20), f"dari {ui.fmt_int(len(g))} {ent}"),
                ui.kpi("Kontribusi 20% teratas", ui.fmt_pct(top_m / total * 100 if total else 0), f"dari total {mname}"),
                ui.kpi("Sisa 80%", fmt_m(total - top_m), f"{ui.fmt_int(len(g) - n20)} {ent}")])
    with st.container(border=True):
        ui.section(f"Konsentrasi {mname} — Pareto 80/20", "Batang biru tua = 20% teratas · garis = persentase kumulatif")
        _pareto_fig(list(g[key]), g[mcol].to_numpy(), n20, "pi_pareto", "Room night" if by_rn else "Spend (Rp)")
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        with h1:
            ui.section("Kontributor teratas", f"20% teratas: {ent} dengan {mname} terbesar")
        top = g.head(n20)
        if dim != "Hotel":
            top = top.drop(columns=["Kota"])   # kolom 'Kota' bantu hanya untuk dimensi Hotel (hindari kolom kembar)
        top = top.rename(columns={key: "Hotel / properti" if dim == "Hotel" else "Kota"})
        order_cols = ["RN", "Spend"] if by_rn else ["Spend", "RN"]
        cols = [top.columns[0]] + (["Kota"] if dim == "Hotel" else []) + order_cols + ["Booking", "Rata-rata rate / malam",
                                                                                     share, "Kumulatif %"]
        top = top[cols].rename(columns={"RN": "Room night", "Spend": "Spend (Rp)"})
        with h2:
            _dl("Unduh Excel", top, "Kontributor teratas", "pareto_room_night" if by_rn else "pareto_spend", "pi_dl",
                money_cols=("Spend (Rp)", "Rata-rata rate / malam"), int_cols=("Room night", "Booking"),
                pct_cols=(share, "Kumulatif %"))
        show = top.copy()
        show["Spend (Rp)"] = show["Spend (Rp)"].map(ui.fmt_big)
        show["Room night"] = show["Room night"].map(ui.fmt_int)
        show["Rata-rata rate / malam"] = show["Rata-rata rate / malam"].map(ui.fmt_rp)
        show[share] = show[share].map(ui.fmt_pct)
        show["Kumulatif %"] = show["Kumulatif %"].map(ui.fmt_pct)
        show.index = show.index + 1
        st.dataframe(show, width="stretch", height=min(420, 38 * len(show) + 40))
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        with h1:
            ui.section("Analisis harga hotel & simulator negosiasi")
        hd = d.copy()
        hd["_rate"] = hd["Invoice Amount"] / hd["Number of Rooms Night"].replace(0, np.nan)
        hd = hd[hd["_rate"].notna()]
        mval = "Number of Rooms Night" if by_rn else "Invoice Amount"
        order = list(g[key]) if dim == "Hotel" else list(hd.groupby("Hotel / Properti" if "Hotel / Properti" in hd
                                                                     else "Hotel Name")[mval].sum()
                                                         .sort_values(ascending=False).index)
        hcol = key if dim == "Hotel" else ("Hotel / Properti" if "Hotel / Properti" in hd else "Hotel Name")
        with h2:
            pick = st.selectbox("Pilih hotel", order, key="pi_pick")
        x = hd[hd[hcol] == pick]
        if x.empty:
            st.info("Data rate hotel ini belum cukup.")
            return
        avg = x["Invoice Amount"].sum() / x["Number of Rooms Night"].sum()
        ui.kpi_row([ui.kpi("Rata-rata rate", ui.fmt_rp(avg)), ui.kpi("Rate median", ui.fmt_rp(x["_rate"].median())),
                    ui.kpi("Rate tertinggi", ui.fmt_rp(x["_rate"].max())), ui.kpi("Rate terendah", ui.fmt_rp(x["_rate"].min())),
                    ui.kpi("Room night", ui.fmt_int(x["Number of Rooms Night"].sum()))], 150)
        s1, s2, s3 = st.columns([2, 1, 1])
        with s1:
            disc = st.slider("Target diskon (%)", 0, 30, 10, key="pi_disc")
        nego = avg * (1 - disc / 100)
        with s2:
            st.metric("Rate hasil negosiasi", ui.fmt_rp(nego))
        with s3:
            st.metric("Estimasi penghematan", ui.fmt_big((avg - nego) * x["Number of Rooms Night"].sum()))


# ---------------------------------------------------------------- tab: Room Intelligence
def tab_room(df, pmap):
    ui.page_title("Room Intelligence", "Konsentrasi room night · simulator volume leverage · detail per properti")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.3, 2, 1.2, 1.5])
        with c1:
            wil = ui.choice("Wilayah", ["Indonesia", "Internasional"], key="ri_wil")
        d = _market(df, wil)
        with c2:
            hotels = st.multiselect("Filter hotel (opsional)", sorted(d["Hotel Name"].dropna().unique()),
                                    placeholder="Semua hotel", key="ri_hotels")
        with c3:
            dim = st.selectbox("Dimensi Pareto", ["Hotel", "Kota"], key="ri_dim")
        with c4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            merge = st.checkbox("Gabungkan properti yang sama", value=True, key="ri_merge", disabled=dim != "Hotel")
    if hotels:
        d = d[d["Hotel Name"].isin(hotels)]
    d = d[d["Number of Rooms Night"] > 0]
    if d.empty:
        st.info("Tidak ada data room night untuk filter ini.")
        return
    d, key = _key_column(d, dim, True if dim != "Hotel" else merge, pmap)
    if "Hotel / Properti" not in d.columns:
        d = apply_property_merge(d, pmap, hotel_col="Hotel Name", city_col="City")
    g = d.groupby(key).agg(RN=("Number of Rooms Night", "sum"), Spend=("Invoice Amount", "sum"),
                           Booking=("Travel Request Number", "nunique") if "Travel Request Number" in d else
                           ("Number of Rooms Night", "size")).reset_index().sort_values("RN", ascending=False)
    g = g.reset_index(drop=True)
    tot = g["RN"].sum()
    n20 = max(1, int(len(g) * 0.2))
    top_rn = g["RN"].head(n20).sum()
    ent = "properti" if dim == "Hotel" else "kota"
    ui.kpi_row([ui.kpi(f"Total room night · {wil}", ui.fmt_int(tot), _period_label(d), dark=True),
                ui.kpi("20% teratas", ui.fmt_int(n20), f"dari {ui.fmt_int(len(g))} {ent}"),
                ui.kpi("Kontribusi 20% teratas", ui.fmt_pct(top_rn / tot * 100), "dari total room night"),
                ui.kpi("Rata-rata lama menginap", f"{tot / max(g['Booking'].sum(), 1):.1f} malam".replace(".", ","),
                       "per booking")])
    with st.container(border=True):
        ui.section("Konsentrasi room night — Pareto 80/20",
                   f"100 teratas: {ent} · batang biru tua = 20% teratas · garis = persentase kumulatif")
        _pareto_fig(list(g[key]), g["RN"].to_numpy(), n20, "ri_pareto", "Room night")
        with st.expander("Detail data 20% teratas"):
            t = g.head(n20).copy()
            t["Rata-rata rate / malam"] = t["Spend"] / t["RN"].replace(0, np.nan)
            t["RN %"] = t["RN"] / tot * 100
            t = t.rename(columns={key: "Hotel / properti" if dim == "Hotel" else "Kota", "RN": "Room night",
                                  "Spend": "Spend (Rp)"})
            _dl("Unduh Excel", t, "Room night teratas", "pareto_room_nights", "ri_dl",
                money_cols=("Spend (Rp)", "Rata-rata rate / malam"), int_cols=("Room night", "Booking"), pct_cols=("RN %",))
            st.dataframe(t, width="stretch", hide_index=True,
                         column_config={"Spend (Rp)": st.column_config.NumberColumn(format="%,.0f"),
                                        "Rata-rata rate / malam": st.column_config.NumberColumn(format="%,.0f"),
                                        "RN %": st.column_config.NumberColumn(format="%.2f%%")})
    with st.container(border=True):
        ui.section("Simulator volume leverage", "Konsolidasi room night ke properti 20% teratas untuk diskon tier volume")
        s1, s2 = st.columns(2)
        with s1:
            shift = st.slider("Room night dialihkan dari 80% bawah (%)", 0, 50, 10, key="ri_shift")
        with s2:
            disc = st.slider("Diskon tier volume 20% teratas (%)", 0, 20, 5, key="ri_vdisc")
        top_inv = g["Spend"].head(n20).sum()
        rate = top_inv / top_rn if top_rn else 0
        proj = top_rn + (tot - top_rn) * shift / 100
        ui.kpi_row([ui.kpi("Room night 20% teratas saat ini", ui.fmt_int(top_rn)),
                    ui.kpi("Proyeksi setelah konsolidasi", ui.fmt_int(proj)),
                    ui.kpi("Potensi saving (simulasi)", ui.fmt_big(proj * rate * disc / 100),
                           f"rata-rata rate 20% teratas {ui.fmt_rp(rate)}", dark=True)], 220)
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        with h1:
            ui.section("Detail properti")
        pr = d.groupby("Hotel / Properti")["Number of Rooms Night"].sum().sort_values(ascending=False)
        with h2:
            pick = st.selectbox("Pilih properti", list(pr.index), key="ri_pick",
                                format_func=lambda x: f"{x} · {ui.fmt_int(pr[x])} RN")
        x = d[d["Hotel / Properti"] == pick]
        rn = x["Number of Rooms Night"].sum()
        bk = x["Travel Request Number"].nunique() if "Travel Request Number" in x else len(x)
        ui.kpi_row([ui.kpi("Room night", ui.fmt_int(rn)), ui.kpi("Booking", ui.fmt_int(bk)),
                    ui.kpi("Lama menginap", f"{rn / max(bk, 1):.1f} malam".replace(".", ",")),
                    ui.kpi("Porsi room night", ui.fmt_pct(rn / d["Number of Rooms Night"].sum() * 100, digits=2)),
                    ui.kpi("Rata-rata rate / malam", ui.fmt_rp(x["Invoice Amount"].sum() / rn if rn else np.nan))], 150)
        if "Check in Date" in x:
            m = x.dropna(subset=["Check in Date"]).groupby(x["Check in Date"].dt.to_period("M").astype(str))[
                "Number of Rooms Night"].sum()
            fig = go.Figure(go.Bar(x=list(m.index), y=list(m.values), marker_color=ui.BLUE,
                                   hovertemplate="%{x}<br>%{y:,.0f} RN<extra></extra>"))
            fig.update_layout(yaxis=dict(gridcolor="#EDEDED"))
            st.markdown("<div style='font-size:13px;font-weight:600'>Tren room night bulanan (check-in)</div>",
                        unsafe_allow_html=True)
            _fig(fig, "ri_month", 260)


# ---------------------------------------------------------------- tab: Sankey
def tab_sankey(df):
    ui.page_title("Sankey Flow", "Alur spend berbasis Invoice Amount: Negara → Kota → Hotel")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1.2])
        with c1:
            wil = ui.choice("Wilayah", ["Semua", "Domestik", "Internasional"], key="sk_wil")
        with c2:
            n_city = st.selectbox("Kota teratas", [5, 8, 10, 15, 20], index=2, key="sk_city")
        with c3:
            n_hotel = st.selectbox("Hotel teratas", [10, 15, 20, 30, 50], index=2, key="sk_hotel")
    base = _market(df, wil).dropna(subset=["Country", "City", "Hotel Name", "Invoice Amount"])
    if base.empty:
        st.info("Tidak ada data.")
        return
    tc = base.groupby("City")["Invoice Amount"].sum().nlargest(n_city).index
    th = base.groupby("Hotel Name")["Invoice Amount"].sum().nlargest(n_hotel).index
    d = base[base["City"].isin(tc) & base["Hotel Name"].isin(th)]
    flow = d["Invoice Amount"].sum()
    exp = d.groupby(["Country", "City", "Hotel Name"])["Invoice Amount"].sum().reset_index().sort_values(
        "Invoice Amount", ascending=False).rename(columns={"Invoice Amount": "Spend (Rp)"})
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _dl("Unduh Excel", exp, "Sankey Flow", "sankey_flow", "sk_dl", money_cols=("Spend (Rp)",))
    ui.kpi_row([ui.kpi("Negara", ui.fmt_int(d["Country"].nunique())), ui.kpi("Kota tujuan", ui.fmt_int(d["City"].nunique())),
                ui.kpi("Hotel", ui.fmt_int(d["Hotel Name"].nunique())),
                ui.kpi("Total spend dalam alur", ui.fmt_big(flow),
                       f"{ui.fmt_pct(flow / base['Invoice Amount'].sum() * 100, digits=0)} dari total", dark=True)])
    if d.empty:
        st.info("Tidak ada alur untuk kombinasi top kota & hotel ini.")
        return
    a = d.groupby(["Country", "City"])["Invoice Amount"].sum().reset_index()
    b = d.groupby(["City", "Hotel Name"])["Invoice Amount"].sum().reset_index()
    countries, cities, hotels = list(a["Country"].unique()), list(a["City"].unique()), list(b["Hotel Name"].unique())
    labels = countries + [f"{c} " for c in cities] + hotels
    idx = {n: i for i, n in enumerate(labels)}
    src = [idx[r.Country] for r in a.itertuples()] + [idx[f"{r.City} "] for r in b.itertuples()]
    tgt = [idx[f"{r.City} "] for r in a.itertuples()] + [idx[r._2] for r in b.itertuples()]
    val = list(a["Invoice Amount"]) + list(b["Invoice Amount"])
    fig = go.Figure(go.Sankey(arrangement="snap",
                              node=dict(label=labels, pad=18, thickness=14, color=ui.NAVY,
                                        line=dict(color="white", width=0.5)),
                              link=dict(source=src, target=tgt, value=val, color="rgba(27,160,226,0.22)")))
    with st.container(border=True):
        ui.section("Alur spend: Negara → Kota → Hotel", "Ketebalan alur = Invoice Amount (Rupiah)")
        _fig(fig, "sk_fig", max(460, 26 * len(hotels) + 80))
    c1, c2 = st.columns(2)
    for col, title, grp in ((c1, "Top kota", "City"), (c2, "Top hotel", "Hotel Name")):
        with col:
            with st.container(border=True):
                ui.section(title)
                t = d.groupby(grp)["Invoice Amount"].sum().sort_values(ascending=False).head(10).reset_index()
                t["Spend"] = t["Invoice Amount"].map(ui.fmt_big)
                st.dataframe(t[[grp, "Spend"]].rename(columns={grp: "Kota" if grp == "City" else "Hotel"}),
                             hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab: Top Hotel/City
def tab_top(df):
    ui.page_title("Top Hotel / City", "Peringkat hotel dan kota berdasarkan total room night")
    wil = ui.choice("Wilayah", ["Domestik", "Internasional"], key="th_wil")
    d = _market(df, wil)
    if d.empty:
        st.info("Tidak ada data.")
        return
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            h = d.groupby("Hotel Name")["Number of Rooms Night"].sum().sort_values(ascending=False).head(100)
            city = d.dropna(subset=["City"]).groupby("Hotel Name")["City"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
            rate = d.groupby("Hotel Name").apply(lambda x: x["Invoice Amount"].sum() /
                                                 max(x["Number of Rooms Night"].sum(), 1), include_groups=False)
            t = pd.DataFrame({"Hotel Name": h.index, "City": h.index.map(city), "Number of Rooms Night": h.values,
                              "Average Room Rate": h.index.map(rate).round(0)})
            a, b = st.columns([3, 2])
            with a:
                ui.section("100 hotel teratas", "Biru tua = 20 teratas · room night")
            with b:
                _dl("Unduh Excel", t, "Top 100 Hotels", "top_100_hotels", "th_dl_h",
                    money_cols=("Average Room Rate",), int_cols=("Number of Rooms Night",))
            with st.container(height=640, border=False):
                _hbar(list(h.index), list(h.values), "th_h", top_highlight=20, height=26 * len(h) + 40)
    with c2:
        with st.container(border=True):
            c = d.groupby("City")["Number of Rooms Night"].sum().sort_values(ascending=False).head(100)
            a, b = st.columns([3, 2])
            with a:
                ui.section("100 kota teratas", "Biru tua = 20 teratas · room night")
            with b:
                _dl("Unduh Excel", c.reset_index(), "Top 100 Cities", "top_100_cities", "th_dl_c",
                    int_cols=("Number of Rooms Night",))
            with st.container(height=640, border=False):
                _hbar(list(c.index), list(c.values), "th_c", top_highlight=20, height=26 * len(c) + 40)
    st.caption("File unduhan hotel berisi kota dan rata-rata rate kamar (total Invoice Amount ÷ total room night).")


# ---------------------------------------------------------------- tab: Dendrogram
CLUSTER_COLORS = [ui.P3, "#5FA80C", ui.ORANGE_A, ui.CRIMSON, "#7A4FD6", "#00A3A3"]
_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def _rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _per_label(p):
    """'2025-04' -> 'Apr 2025'."""
    try:
        y, m = str(p).split("-")[:2]
        return f"{_BULAN[int(m) - 1]} {y}"
    except Exception:
        return str(p)


def _pattern_label(prof, cols):
    """Label otomatis pola musiman klaster dari profil porsi bulanan rata-rata."""
    prof = np.asarray(prof, float)
    if len(prof) < 3 or prof.mean() == 0 or prof.std() / prof.mean() < 0.12:
        return "Stabil sepanjang periode"
    i = int(np.argmax(prof))
    return f"Puncak {_per_label(cols[i])}"


def _circle_xy(x, y, r, n=48):
    t = np.linspace(0, 2 * np.pi, n)
    return x + r * np.cos(t), y + r * np.sin(t)


@st.cache_data(max_entries=16, show_spinner=False)
def _pack(groups: tuple):
    """groups = ((klaster, ((nama, nilai), ...)), ...) -> {('K',k): (x,y,r), nama: (x,y,r)}"""
    import circlify
    data = [{"id": f"K{k}", "datum": float(sum(v for _, v in mem)) or 1.0,
             "children": [{"id": n, "datum": float(v) or 1e-6} for n, v in mem]} for k, mem in groups]
    out = {}
    for c in circlify.circlify(data, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1)):
        key = ("K", int(c.ex["id"][1:])) if c.level == 1 else c.ex["id"]
        out[key] = (c.x, c.y, c.r)
    return out


def tab_dendro(df):
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import normalize
    ui.page_title("Cluster Analysis",
                  "Mengelompokkan kota atau hotel yang pola volume bulanannya mirip")
    with st.container(border=True):
        c = st.columns(5)
        ent = c[0].selectbox("Entitas", ["Kota", "Hotel"], key="dd_ent")
        metric = c[1].selectbox("Metric", ["Invoice Amount", "Number of Rooms Night"], key="dd_metric",
                                format_func=lambda m: "Spend (Rupiah)" if m == "Invoice Amount" else "Room night")
        method = c[2].selectbox("Metode linkage", ["ward", "complete", "average", "single"], key="dd_link")
        top_n = c[3].selectbox("Jumlah teratas", [12, 20, 30, 50], key="dd_n")
        k = c[4].selectbox("Jumlah klaster", [4, 3, 5, 6], key="dd_k")
    col = "City" if ent == "Kota" else "Hotel Name"
    unit = "kota" if ent == "Kota" else "hotel"
    is_rp = metric == "Invoice Amount"
    fmt_v = (lambda v: ui.fmt_big(v)) if is_rp else (lambda v: f"{ui.fmt_int(v)} RN")
    d = df.dropna(subset=[col, metric, "Issue Time"]).copy()
    d["Bulan"] = d["Issue Time"].dt.to_period("M").astype(str)
    top = d.groupby(col)[metric].sum().nlargest(top_n).index
    pv = d[d[col].isin(top)].groupby([col, "Bulan"])[metric].sum().unstack(fill_value=0)
    if len(pv) < 3:
        st.info("Data belum cukup (minimal 3 entitas).")
        return
    X = normalize(pv.values, norm="l2")
    Z = linkage(X, method=method)
    k = min(k, len(pv) - 1)
    cl = pd.Series(fcluster(Z, t=k, criterion="maxclust"), index=pv.index)
    ks = sorted(cl.unique())
    vals = pv.sum(axis=1).clip(lower=0)
    tot = float(vals.sum()) or 1.0
    share = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    months = list(pv.columns)
    colr = {kk: CLUSTER_COLORS[(i) % len(CLUSTER_COLORS)] for i, kk in enumerate(ks)}
    prof = {kk: share[cl == kk].mean().values for kk in ks}
    plab = {kk: _pattern_label(prof[kk], months) for kk in ks}
    peak_ent = share.idxmax(axis=1).map(_per_label)

    # ---------- pilihan sorotan (klik lingkaran juga mengubah ini)
    opts = ["Semua"] + [f"Klaster {kk}" for kk in ks]
    pend = st.session_state.pop("_dd_click", None)
    if pend in opts:
        st.session_state["dd_focus"] = pend
    if st.session_state.get("dd_focus") not in opts:
        st.session_state["dd_focus"] = "Semua"
    focus_s = st.segmented_control("Sorot klaster", opts, key="dd_focus", label_visibility="collapsed")
    focus = int(focus_s.split()[-1]) if focus_s and focus_s != "Semua" else None

    groups = tuple((int(kk), tuple((str(n), float(vals[n])) for n in vals[cl == kk].sort_values(ascending=False).index))
                   for kk in ks)
    pos = _pack(groups)
    go_ = go
    fig = go_.Figure()
    nk = len(ks)
    # lingkaran klaster (latar) + nomor
    for kk in ks:
        x, y, r = pos[("K", int(kk))]
        dim = focus is not None and focus != kk
        cx, cy = _circle_xy(x, y, r, 72)
        fig.add_trace(go_.Scatter(x=cx, y=cy, mode="lines", fill="toself",
                                  fillcolor=_rgba(colr[kk], 0.03 if dim else 0.07),
                                  line=dict(color=_rgba(colr[kk], 0.2 if dim else 0.5), width=1.4, dash="dot"),
                                  hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=x - r * 0.72, y=y + r * 0.72, text=f"<b>{kk}</b>", showarrow=False,
                           font=dict(size=13, color="white"), bgcolor=_rgba(colr[kk], 0.35 if dim else 1),
                           borderpad=5, bordercolor="white", borderwidth=2)
    # lingkaran entitas: poligon terisi -> hover di seluruh area lingkaran
    for kk in ks:
        dim = focus is not None and focus != kk
        for n in cl[cl == kk].index:
            x, y, r = pos[str(n)]
            cx, cy = _circle_xy(x, y, r)
            fig.add_trace(go_.Scatter(x=cx, y=cy, mode="lines", fill="toself", hoveron="fills",
                                      fillcolor=_rgba(colr[kk], 0.18 if dim else 0.88),
                                      line=dict(color="white", width=2), hoverinfo="skip",
                                      showlegend=False))
    # penanda tak terlihat seukuran lingkaran: sumber tooltip & klik
    fig_h = max(520, 120 * nk + 60)
    ppu = (fig_h - 20) / 2.1          # piksel per satuan (tinggi yang membatasi skala)
    cxs, cys, cds, szs, tips = [], [], [], [], []
    for n in cl.index:
        x, y, r = pos[str(n)]
        kk = int(cl[n]); v = float(vals[n])
        cxs.append(x); cys.append(y); cds.append(kk); szs.append(max(8, 2 * r * ppu))
        tips.append(f"<b>{ui.esc(n)}</b><br>Klaster {kk} · {plab[kk]}<br>{fmt_v(v)} · "
                    f"{ui.fmt_pct(v / tot * 100)} dari total<br>Bulan tersibuk: {peak_ent[n]}")
    fig.add_trace(go_.Scatter(x=cxs, y=cys, mode="markers", customdata=cds, text=tips,
                              hovertemplate="%{text}<extra></extra>", showlegend=False, name="pilih",
                              marker=dict(size=szs, color="rgba(0,0,0,0)", line_width=0),
                              selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
                              hoverlabel=dict(bgcolor="white", bordercolor=[colr[c_] for c_ in cds],
                                              font=dict(color=ui.INK, size=12))))
    # label di dalam lingkaran (hanya yang muat)
    for n in cl.index:
        x, y, r = pos[str(n)]
        if r < 0.09 or len(str(n)) * 0.021 > 1.8 * r:
            continue
        dim = focus is not None and focus != cl[n]
        big = r > 0.15
        txt = f"<b>{ui.esc(n)}</b><br>{fmt_v(vals[n])}" if big else f"<b>{ui.esc(n)}</b>"
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=12 if big else 10, color=_rgba("#FFFFFF", 0.5 if dim else 1)))
    # kartu klaster kanan + sparkline
    for j, kk in enumerate(ks):
        y0 = nk - 1 - j
        dim = focus is not None and focus != kk
        a = 0.35 if dim else 1
        fig.add_shape(type="rect", xref="x2", yref="y2", x0=0, x1=1, y0=y0 + 0.06, y1=y0 + 0.94,
                      fillcolor=_rgba(colr[kk], 0.10 if focus == kk else 0.05),
                      line=dict(color=_rgba(colr[kk], 0.9 if focus == kk else 0.35), width=2 if focus == kk else 1))
        fig.add_shape(type="rect", xref="x2", yref="y2", x0=0, x1=0.018, y0=y0 + 0.06, y1=y0 + 0.94,
                      fillcolor=_rgba(colr[kk], a), line_width=0)
        v = float(vals[cl == kk].sum())
        fig.add_annotation(xref="x2", yref="y2", x=0.06, y=y0 + 0.84, xanchor="left", yanchor="top", showarrow=False,
                           align="left", font=dict(size=12, color=_rgba(ui.INK, a)),
                           text=f"<span style='color:{_rgba(colr[kk], a)}'><b>KLASTER {kk}</b></span>  <b>{plab[kk]}</b><br>"
                                f"<span style='color:{_rgba('#6B6B6B', a)}'>{(cl == kk).sum()} {unit} · {fmt_v(v)} · "
                                f"{ui.fmt_pct(v / tot * 100, digits=0)}</span>")
        p = prof[kk]
        amp = 0.26 if plab[kk].startswith("Puncak") else 0.08
        sx = np.linspace(0.06, 0.94, len(p))
        sy = y0 + 0.13 + (p - p.min()) / (p.max() - p.min() + 1e-9) * amp
        fig.add_trace(go_.Scatter(x=sx, y=sy, xaxis="x2", yaxis="y2", mode="lines",
                                  line=dict(color=_rgba(colr[kk], a), width=2.2, shape="spline", smoothing=0.6),
                                  customdata=np.c_[[_per_label(m) for m in months], p],
                                  hovertemplate="%{customdata[0]}: %{customdata[1]:.1f}% volume klaster<extra></extra>",
                                  showlegend=False))
    fig.update_layout(**{kk_: vv for kk_, vv in ui.PLOTLY_LAYOUT.items() if kk_ not in ("xaxis", "yaxis", "margin")})
    fig.update_layout(height=fig_h, margin=dict(l=10, r=10, t=10, b=10), dragmode=False,
                      hovermode="closest", clickmode="event+select",
                      xaxis=dict(visible=False, domain=[0, 0.62], range=[-1.05, 1.05], fixedrange=True),
                      yaxis=dict(visible=False, range=[-1.05, 1.05], scaleanchor="x", fixedrange=True),
                      xaxis2=dict(visible=False, domain=[0.66, 1], range=[0, 1], anchor="y2", fixedrange=True),
                      yaxis2=dict(visible=False, range=[nk - max(nk, 4), nk], anchor="x2", fixedrange=True))
    with st.container(border=True):
        ui.section(f"Klaster {unit} berdasarkan pola bulanan",
                   f"Lingkaran besar = klaster · lingkaran kecil = {unit}, ukuran = "
                   f"{'spend' if is_rp else 'room night'} · kartu kanan = pola musiman · "
                   f"arahkan kursor untuk detail, klik lingkaran untuk menyorot klasternya")
        ev = st.plotly_chart(fig, width="stretch", key=f"dd_pack_{focus}", on_select="rerun",
                             selection_mode="points", config={"displaylogo": False, "displayModeBar": False})
        try:
            pts = ev.selection.points if ev and ev.selection else []
        except Exception:
            pts = []
        if pts:
            cd = pts[0].get("customdata")
            kk = int(cd[0] if isinstance(cd, (list, tuple)) else cd) if cd is not None else None
            if kk in ks:
                want = "Semua" if focus == kk else f"Klaster {kk}"
                st.session_state["_dd_click"] = want
                st.rerun()

    # ---------- detail klaster yang disorot
    if focus is not None:
        kk = focus
        mem = cl[cl == kk].index
        with st.container(border=True):
            ui.section(f"Klaster {kk} · {plab[kk]}", f"{len(mem)} {unit} dengan pola bulanan serupa")
            a, b = st.columns([1.35, 1])
            with a:
                f2 = go_.Figure()
                xl = [_per_label(m) for m in months]
                for n in mem:
                    f2.add_trace(go_.Scatter(x=xl, y=share.loc[n].values, mode="lines", name=str(n),
                                             line=dict(color="#CFCFCF", width=1),
                                             hovertemplate=f"{ui.esc(n)}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
                f2.add_trace(go_.Scatter(x=xl, y=prof[kk], mode="lines+markers", name="Rata-rata klaster",
                                         line=dict(color=colr[kk], width=3.5, shape="spline", smoothing=0.6),
                                         marker=dict(size=5),
                                         hovertemplate="Rata-rata klaster<br>%{x}: %{y:.1f}%<extra></extra>"))
                f2.add_hline(y=100 / len(months), line_dash="dot", line_color="#9A9A9A",
                             annotation_text="rata-rata merata", annotation_font_size=10)
                f2.update_layout(showlegend=False, yaxis=dict(title="Porsi volume per bulan", ticksuffix="%"),
                                 xaxis=dict(title=None))
                _fig(f2, "dd_prof", 330)
            with b:
                t = pd.DataFrame({ent: mem, "Nilai": vals[mem].values,
                                  "Porsi": vals[mem].values / tot * 100,
                                  "Bulan tersibuk": peak_ent[mem].values}).sort_values("Nilai", ascending=False)
                t["Nilai"] = t["Nilai"].map(fmt_v)
                t["Porsi"] = t["Porsi"].map(lambda v: ui.fmt_pct(v))
                st.dataframe(t.rename(columns={"Nilai": "Spend" if is_rp else "Room night"}), hide_index=True,
                             width="stretch", height=min(330, 36 * len(t) + 40))
            ui.note(f"Garis tebal = pola rata-rata Klaster {kk}; garis abu = tiap {unit}. Gunakan bulan puncak untuk "
                    f"menjadwalkan negosiasi corp rate <b>sebelum</b> masa ramai, dan satukan {unit} dalam klaster "
                    f"ini dalam satu paket negosiasi.")

    # ---------- cara membaca visual
    big_k = max(ks, key=lambda q: float(vals[cl == q].sum()))
    big_v = float(vals[cl == big_k].sum())
    top_ent = vals.idxmax()
    peaks = [f"Klaster {q} ({plab[q].replace('Puncak', 'puncak')})" for q in ks if plab[q].startswith("Puncak")]
    with st.expander("Cara membaca grafik", icon=":material/menu_book:"):
        st.markdown(f"""
**1. Lingkaran besar = klaster.** Berisi {unit} yang pola volume bulanannya mirip, yaitu ramai dan sepinya di
bulan yang sama, walaupun besar transaksinya berbeda. Nomor di pojok lingkaran sama dengan nomor kartu di kanan.

**2. Lingkaran kecil = {unit}.** Makin besar lingkarannya, makin besar {'spend' if is_rp else 'room night'}-nya.
Nama dan nilai tertulis di dalam lingkaran yang cukup besar; yang kecil bisa dilihat dengan mengarahkan kursor.

**3. Kartu di kanan = pola musiman klaster.** Garis menunjukkan porsi volume klaster per bulan, puncak garis
adalah bulan tersibuk. Label *Puncak …* artinya ada bulan yang jelas lebih ramai; *Stabil sepanjang periode*
artinya volumenya merata.

**4. Klik lingkaran** (atau tombol klaster di atas grafik) untuk menyorot satu klaster. Di bawahnya muncul
grafik pola tiap anggota dan tabel {unit} dalam klaster itu. Klik lagi atau pilih *Semua* untuk kembali.

**5. Pengaturan.** *Jumlah klaster* menentukan berapa kelompok yang dibentuk; *Linkage* adalah cara
pengelompokan (biarkan *ward* bila ragu); *Top N* adalah jumlah {unit} dengan nilai terbesar yang dianalisis.
""")
    ui.note(f"<b>Ringkasnya:</b> Klaster {big_k} adalah kelompok terbesar ({fmt_v(big_v)}, "
            f"{ui.fmt_pct(big_v / tot * 100, digits=0)} dari total), dan {ui.esc(top_ent)} adalah {unit} "
            f"dengan nilai terbesar."
            + (f" Klaster dengan musim puncak: {', '.join(peaks)}; jadwalkan negosiasi corp rate sebelum "
               f"bulan puncak tersebut." if peaks else " Semua klaster relatif stabil sepanjang periode."))
    ui.note(f"{unit.capitalize()} dalam satu klaster punya pola perjalanan serupa, sehingga bisa dinegosiasikan "
            f"dengan strategi yang sama, misalnya satu paket corp rate untuk {unit}-{unit} dalam satu klaster.")


# ---------------------------------------------------------------- tab: Patra Jasa
def tab_patra(df):
    ui.page_title("Patra Jasa Group vs Non-Patra Jasa", "Perbandingan hotel domestik (Country = Indonesia)")
    d = _market(df, "Domestik").copy()
    if d.empty or "Hotel Name" not in d:
        st.info("Tidak ada data domestik.")
        return
    norm = {h.lower().strip(): h for h in PATRA_JASA_HOTELS}
    norm.update({str(standardize_name(h)).lower().strip(): h for h in PATRA_JASA_HOTELS})
    d["_h"] = d["Hotel Name"].astype(str).str.lower().str.strip()
    d["Grup"] = np.where(d["_h"].isin(norm), "Patra Jasa Group", "Non-Patra Jasa")
    d["_dt"] = d["Issue Time"].fillna(d.get("Check in Date")) if "Issue Time" in d else d.get("Check in Date")
    years = sorted(d["_dt"].dropna().dt.year.unique().astype(int), reverse=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        year = c1.selectbox("Tahun (tabel bulanan)", years, key="pj_year") if years else None
        met = c2.selectbox("Metrik tabel", ["Invoice (TR unik)", "Room night"], key="pj_met")
    tr = "Travel Request Number" in d
    cnt = lambda x: x["Travel Request Number"].nunique() if tr else len(x)  # noqa: E731
    pj, npj = d[d["Grup"] == "Patra Jasa Group"], d[d["Grup"] == "Non-Patra Jasa"]
    tot = cnt(pj) + cnt(npj)  # satu travel request bisa menginap di kedua grup
    ui.kpi_row([ui.kpi("Total invoice domestik", ui.fmt_int(tot), "travel request unik per grup"),
                ui.kpi("Patra Jasa Group", ui.fmt_int(cnt(pj)), f"{ui.fmt_pct(cnt(pj) / max(tot, 1) * 100)} dari total",
                       dark=True),
                ui.kpi("Non-Patra Jasa", ui.fmt_int(cnt(npj)), f"{ui.fmt_pct(cnt(npj) / max(tot, 1) * 100)} dari total"),
                ui.kpi("Total room night domestik", ui.fmt_int(d["Number of Rooms Night"].sum()))], 200)
    a, b = st.columns(2)
    for col, title, v1, v2, center, key in (
            (a, "Invoice unik — Patra vs Non-Patra", cnt(pj), cnt(npj), "invoice", "pj_d1"),
            (b, "Room night — Patra vs Non-Patra", pj["Number of Rooms Night"].sum(),
             npj["Number of Rooms Night"].sum(), "room night", "pj_d2")):
        with col:
            with st.container(border=True):
                ui.section(title)
                fig = go.Figure(go.Pie(labels=["Patra Jasa Group", "Non-Patra Jasa"], values=[v1, v2], hole=0.6,
                                       marker=dict(colors=[ui.BLUE, ui.ORANGE]), sort=False, textinfo="percent"))
                fig.update_layout(annotations=[dict(text=f"<b>{ui.fmt_int(v1 + v2)}</b><br>{center}", x=0.5, y=0.5,
                                                    showarrow=False, font=dict(size=14))],
                                  legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top",
                                              itemwidth=40, bgcolor="rgba(0,0,0,0)"))
                _fig(fig, key, 300)
    if year is None:
        return
    y = d[d["_dt"].dt.year == year].copy()
    y["_m"] = y["_dt"].dt.month
    if met.startswith("Invoice") and tr:
        pv = y.groupby(["Grup", "_m"])["Travel Request Number"].nunique().unstack(fill_value=0)
    else:
        pv = y.groupby(["Grup", "_m"])["Number of Rooms Night"].sum().unstack(fill_value=0)
    pv = pv.reindex(index=["Patra Jasa Group", "Non-Patra Jasa"], columns=range(1, 13), fill_value=0)
    names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agt", "Sep", "Okt", "Nov", "Des"]
    rows = ""
    for grp, color, bg in (("Patra Jasa Group", ui.P3, "#F3F9FD"), ("Non-Patra Jasa", ui.POS_TEXT, "#F7FDEB")):
        vals = pv.loc[grp]
        rows += f'<tr style="background:{bg}"><td style="font-weight:500">{grp}</td>' + "".join(
            f'<td style="color:{color}">{ui.fmt_int(v) if v else "—"}</td>' for v in vals) + \
            f'<td style="font-weight:600;background:#EAF5FC">{ui.fmt_int(vals.sum())}</td></tr>'
    tot_m = pv.sum()
    rows += '<tr style="font-style:italic"><td style="color:#6B6B6B">% Patra Jasa</td>' + "".join(
        f'<td style="color:#6B6B6B">{(pv.loc["Patra Jasa Group", m] / t * 100):.0f}%</td>' if t else
        '<td style="color:#9A9A9A">—</td>' for m, t in tot_m.items()) + \
        f'<td style="color:#6B6B6B;background:#EAF5FC">{pv.loc["Patra Jasa Group"].sum() / max(tot_m.sum(), 1) * 100:.0f}%</td></tr>'
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        with h1:
            ui.section(f"Perbandingan bulanan — {met.lower()} · {year}")
        with h2:
            out = pv.copy()
            out.columns = names
            out["Total"] = out.sum(axis=1)
            _dl("Unduh Excel", out.reset_index().rename(columns={"Grup": "Grup hotel"}), "Patra vs Non-Patra",
                "patra_jasa_comparison", "pj_dl")
        st.markdown('<div style="overflow-x:auto"><table class="mx-table"><thead><tr><th>Grup hotel</th>' +
                    "".join(f"<th>{n}</th>" for n in names) + "<th>Total</th></tr></thead><tbody>" + rows +
                    "</tbody></table></div>", unsafe_allow_html=True)
    with st.expander(f"Detail per hotel Patra Jasa Group · {year}"):
        p = y[y["Grup"] == "Patra Jasa Group"].copy()
        p["Hotel"] = p["_h"].map(norm).fillna(p["Hotel Name"])
        t = p.groupby("Hotel").agg(**{"Travel request": ("Travel Request Number", "nunique") if tr else
                                      ("Hotel Name", "size"), "Room night": ("Number of Rooms Night", "sum")})
        st.dataframe(t.sort_values("Travel request", ascending=False), width="stretch")


# ---------------------------------------------------------------- tab: Laporan
def tab_laporan(df, data_sig=None):
    import cost_avoidance as ca
    ui.page_title("Reports", "Semua unduhan berupa laporan agregat dalam format Excel dan tercatat atas nama akun Anda")
    r = df.copy()
    has_tr = "Travel Request Number" in r

    def agg(x, keys):
        g = x.groupby(keys, dropna=False).agg(**{"Room night": ("Number of Rooms Night", "sum"),
                                                 "Spend (Rp)": ("Invoice Amount", "sum")})
        if has_tr:
            g["Travel request"] = x.groupby(keys, dropna=False)["Travel Request Number"].nunique()
        g["Rata-rata rate / malam (Rp)"] = (g["Spend (Rp)"] / g["Room night"].replace(0, np.nan)).round(0)
        return g.reset_index().sort_values("Room night", ascending=False)

    reps = [("Ringkasan per hotel", "Hotel, kota, room night, spend dan rata-rata rate per malam.",
             agg(r, ["Hotel Name", "City"]), "ringkasan_per_hotel"),
            ("Ringkasan per kota", "Negara, kota, room night, spend dan rata-rata rate per malam.",
             agg(r, ["Country", "City"]), "ringkasan_per_kota")]
    if "Issue Time" in r:
        m = r.dropna(subset=["Issue Time"]).copy()
        m["Bulan"] = m["Issue Time"].dt.to_period("M").astype(str)
        m["Market"] = np.where(m["Country"].astype(str).str.upper().isin(DOMESTIC_ALIAS), "Domestik", "Internasional")
        reps.append(("Tren bulanan", "Travel request, room night dan spend per bulan, domestik vs internasional.",
                     agg(m, ["Bulan", "Market"]).sort_values(["Bulan", "Market"]), "tren_bulanan"))
    src = corp_source()
    if src is None or src.get("df") is None:
        src = ca.local_corp_source()
    if src is not None:
        try:
            res = ca.run_analysis(r, data_sig or ca.frame_sig(r), src,
                                  "Belum" if str(st.session_state.get("ca_basis", "")).startswith("Belum") else "Nett",
                                  st.session_state.get("ca_carry", True), standardize_name, _normalize_city_key,
                                  None if str(st.session_state.get("ca_year", "")).isdigit() is False
                                  else int(st.session_state["ca_year"]))
            if not res.get("ready"):
                raise ValueError("belum ada booking yang cocok dengan corp rate")
            mm, corp, pools, s = res["m"], res["corp"], res["pools"], res["summary"]
            if not s.empty:
                reps.append(("Cost avoidance",
                             "Per hotel & tipe kamar: corp rate, rate dibayar, selisih, potensi dan status daya saing.",
                             _cols(s, ["Hotel", "Kota", "Room type", "Meal plan", "Corp", "Rata-rata dibayar", "Selisih %",
                                       "RN", "Potensi", "Room Night < Corp %", "Posisi kota", "Daya saing"])
                             .rename(columns={"Corp": "Corp rate", "RN": "Room night"}), "cost_avoidance"))
            lk = res["leak100"]
            if not lk.empty:
                reps.append(("Kebocoran non-kontrak", "100 hotel non-kontrak teratas dan potensi pengalihan ke hotel ber-corp rate.",
                             _cols(lk, ["Hotel", "Kota", "Room type", "Meal plan", "RN", "Rata-rata dibayar", "Setara",
                                        "Selisih %", "Potensi"]).rename(columns={"RN": "Room night",
                                                                                 "Setara": "Corp rate setara"}),
                             "kebocoran_non_kontrak"))
            md = ca.load_manual()
            rg = ca.renegotiation(s, pools, 40, dict(zip(md["key"].astype(str), pd.to_numeric(md["persen"], errors="coerce"))))
            if not rg.empty:
                reps.append(("Target renegosiasi", "Target rate P40 kota atau target manual, beserta hard savings.",
                             _cols(rg, ["Hotel", "Kota", "Room type", "Meal plan", "Daya saing", "Corp",
                                        "Target dipakai", "Sumber target", "Turun %", "RN", "Hard savings"])
                             .rename(columns={"Corp": "Corp rate", "RN": "Room night"}), "target_renegosiasi"))
        except Exception as e:
            st.caption(f"Laporan Cost Avoidance dilewati: {e}")
    icons = {"ringkasan_per_hotel": "hotel", "ringkasan_per_kota": "pin", "tren_bulanan": "trend",
             "cost_avoidance": "coin", "kebocoran_non_kontrak": "leak", "target_renegosiasi": "target"}
    groups = [("Ringkasan transaksi", "Agregat dari seluruh data transaksi pada periode terpilih",
               ("ringkasan_per_hotel", "ringkasan_per_kota", "tren_bulanan")),
              ("Cost avoidance", "Booking termasuk sarapan yang dibandingkan dengan corp rate",
               ("cost_avoidance", "kebocoran_non_kontrak", "target_renegosiasi"))]
    by = {fname: (title, desc, data) for title, desc, data, fname in reps}
    allow = can_download()
    for gtitle, gsub, names in groups:
        items = [n for n in names if n in by]
        if not items:
            continue
        st.markdown(f'<div class="mx-rep-g"><b>{ui.esc(gtitle)}</b><span>{ui.esc(gsub)}</span></div>',
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, fname in enumerate(items):
            title, desc, data = by[fname]
            with cols[i % 3]:
                with st.container(border=True, key=f"rep_{fname}"):
                    st.markdown(ui.report_card(title, desc, len(data), icons.get(fname, "file")),
                                unsafe_allow_html=True)
                    if allow:
                        _dl("Unduh Excel", data, title, fname, f"lap_{fname}", width="stretch",
                            money_cols=tuple(c for c in data.columns if "(Rp)" in c or c in
                                             ("Corp rate", "Rata-rata dibayar", "Potensi", "Target dipakai",
                                              "Hard savings", "Corp rate setara")),
                            int_cols=tuple(c for c in data.columns if c in ("Room night", "Travel request", "RN")))
                    else:
                        st.caption("Unduhan tidak tersedia untuk akun ini.")
    ui.note("<b>Data per transaksi tidak bisa diunduh dari dashboard.</b> Kebutuhan data per booking diajukan lewat "
            "permintaan data dan disetujui atasan serta Pejabat Pelindungan Data Pribadi.")
    if is_super_admin():
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            ui.section("Log audit", "Login dan unduhan · 200 catatan terakhir · hanya Super Admin")
            log = read_audit_tail(200)
            if log.empty:
                st.caption("Belum ada catatan.")
            else:
                st.dataframe(log, hide_index=True, width="stretch", height=min(360, 36 * len(log) + 40))


# ---------------------------------------------------------------- halaman Akun & akses
def page_akses():
    ui.page_title("Akun, role & akses", "Lima akun tetap. Password & 2FA diatur di Secrets (bukan di kode).")
    yes, no = "Ya", "Tidak"
    matrix = [
        ("Data yang dimuat", "13 kolom transaksi hotel", "13 kolom", "13 kolom", "13 kolom", "13 kolom, nominal disamarkan"),
        ("Semua tab analisis", yes, yes, yes, yes, yes),
        ("Data identitas pekerja", "Tidak ada di sumber", "Tidak ada di sumber", "Tidak ada di sumber",
         "Tidak ada di sumber", "Tidak ada di sumber"),
        ("Unduh laporan agregat", "Ya, tercatat", "Ya, tercatat", "Ya, tercatat", "Ya, tercatat", no),
        ("Unggah Excel, corp rate & target manual", yes, yes, no, no, no),
        ("Lihat log audit", yes, no, no, no, no),
    ]
    head = "".join(f"<th>{h}</th>" for h in ["Kemampuan", "superadmin", "admin", "ssc", "dtm", "demo"])
    body = "".join("<tr>" + "".join(f"<td style='text-align:left'>{ui.esc(c)}</td>" for c in r) + "</tr>" for r in matrix)
    with st.container(border=True):
        ui.section("Hak akses per akun")
        st.markdown(f"<div style='overflow-x:auto'><table class='mx-table' style='font-size:13px'><thead><tr>{head}"
                    f"</tr></thead><tbody>{body}</tbody></table></div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            ui.section("Akun")
            rows = []
            for name, role in USER_ROLES.items():
                aktif = name in USERS
                rows.append({"Akun": name, "Role": role, "Status": "Aktif" if aktif else "Belum ada password",
                             "2FA": "Aktif" if _own_totp_secret(name) else ("Tidak dipakai" if role == "Demo" else "Belum")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b:
        with st.container(border=True):
            ui.section("Keamanan login")
            argon = all(str(u["password"]).startswith("$argon2") for u in USERS.values()) if USERS else False
            ui.mini_row([("Login", "Wajib" if login_required() else "Dimatikan sementara")])
            ui.mini_row([("Hash password", "Argon2id" if argon else "Sebagian masih SHA-256")])
            ui.mini_row([("Kunci akun setelah gagal", f"{MAX_LOGIN_ATTEMPTS}× · {LOCKOUT_SECONDS // 60} menit")])
            ui.mini_row([("Batas waktu sesi", f"{SESSION_TIMEOUT // 60} menit")])


# ---------------------------------------------------------------- navigasi halaman
PAGES = ["Overview", "Price Intelligence", "Cost Avoidance", "Sankey Flow",
         "Top Hotel/City", "Cluster Analysis", "Patra Jasa", "Reports"]


def ca_frame_sig(df):
    import cost_avoidance as _ca
    return _ca.frame_sig(df)


@st.fragment
def _render_page(page, df, pmap, data_sig):
    if page == "Overview":
        tab_dashboard(df)
    elif page == "Price Intelligence":
        tab_price(df, pmap)
    elif page == "Cost Avoidance":
        render_cost_avoidance(df, can_download=can_download(), can_edit=is_admin(),
                              username=st.session_state.get("username", ""), excel_fn=excel_bytes_with_formats,
                              standardize_name=standardize_name, normalize_city=_normalize_city_key,
                              corp_source=corp_source, data_sig=data_sig)
    elif page == "Sankey Flow":
        tab_sankey(df)
    elif page == "Top Hotel/City":
        tab_top(df)
    elif page == "Cluster Analysis":
        tab_dendro(df)
    elif page == "Patra Jasa":
        tab_patra(df)
    elif page == "Reports":
        tab_laporan(df, data_sig)


# ---------------------------------------------------------------- main
def _ensure_state():
    """Pastikan kunci sesi dasar selalu ada (mis. setelah app di-reboot/di-update saat halaman masih terbuka)."""
    for k, v in (("df_all", pd.DataFrame()), ("data_loaded", False), ("data_period", ""), ("username", "")):
        if k not in st.session_state:
            st.session_state[k] = v


def main_app():
    _ensure_state()
    st.set_page_config(page_title="MTRAX | Travel Analytics", layout="wide", initial_sidebar_state="expanded")
    if st.session_state.get("_session_token"):
        _set_session_cookie(st.session_state._session_token)
    ui.inject_css()
    render_sidebar()
    render_header()

    if st.session_state.get("view") == "akses" and is_admin():
        page_akses()
        return

    if is_demo_user() and not st.session_state.get("_demo_masked"):
        st.session_state.df_all = pd.DataFrame()
        st.session_state.data_loaded = False
    raw = st.session_state.get("df_all")
    if raw is None or raw.empty:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        ui.note("Pilih <b>periode data</b> di sidebar lalu klik <b>Muat Data</b>.")
        return

    ver = st.session_state.get("_data_ver", 0)
    if st.session_state.get("_sig_ver") != ver:          # sidik jari data dihitung sekali per penarikan
        st.session_state._raw_sig = ca_frame_sig(raw)
        st.session_state._sig_ver = ver
    sig = st.session_state._raw_sig
    with st.spinner("Menyiapkan data…"):
        df, pmap, fx_info, prep_warn = _prepare_cached(sig, raw)
    st.session_state["_fx_info"] = fx_info
    if prep_warn:
        st.warning(prep_warn)

    dc = _date_col(df)
    rng_key = "all"
    if dc:
        dates = df[dc].dropna()
        lo, hi = dates.min().date(), dates.max().date()
        with st.sidebar:
            rng = st.date_input("Rentang tanggal (semua tab)", (lo, hi), min_value=lo, max_value=hi,
                                format="DD/MM/YYYY", key=f"gf_range_{ver}")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            d_only = df[dc].dt.normalize()
            df = df[(d_only >= pd.Timestamp(rng[0])) & (d_only <= pd.Timestamp(rng[1]))]
            rng_key = f"{rng[0]}_{rng[1]}"
    data_sig = f"{sig}|{rng_key}"

    # Hanya halaman yang dipilih yang dihitung & digambar (st.tabs menghitung SEMUA tab di setiap interaksi).
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.container(key="mx_nav"):
        page = st.radio("Halaman", PAGES, horizontal=True, key="nav_page", label_visibility="collapsed",
                        format_func=lambda p: f"{PAGE_ICONS.get(p, '')} {p}")
    _render_page(page, df, pmap, data_sig)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    ui.note("<b>Catatan penggunaan &amp; kepatuhan</b><br>Aplikasi ini untuk analisis internal. Output tidak bersifat "
            "final dan harus divalidasi sesuai kebijakan perusahaan.")
    st.markdown("<div class='mx-footer'>© 2026 MTRAX Travel Analytics · PT Mitra Tours And Travel · Dikembangkan oleh "
                "<a href='https://www.linkedin.com/in/rifyalt/' target='_blank'>Rifyal Tumber</a>"
                f" · UI {getattr(ui, 'UI_VERSION', 'lama')}</div>",
                unsafe_allow_html=True)



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
