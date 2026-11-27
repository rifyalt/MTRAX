"""
cost_avoidance.py — Tab "Cost Avoidance" MTRAX.

Membandingkan rate per malam yang dibayar (Invoice Amount ÷ Number of Rooms Night,
sudah dalam Rupiah) dengan daftar corporate rate per hotel × tipe kamar × meal plan.

Sub-tab:
  - Ringkasan        : gross potential cost avoidance, kepatuhan rate, daya saing corp rate
  - Analisis lanjutan: kebocoran ke hotel non-kontrak, target renegosiasi (persentil +
                       target turun manual %), kepatuhan per kota, kualitas pencocokan

Fungsi perhitungan (prepare_*, match_*, summarize_*, ...) tidak bergantung pada Streamlit
sehingga bisa diuji terpisah.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

import ui_kit as ui

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DEFAULT_TAX = 0.21                 # pajak + service bila corp rate belum termasuk
STATUS_RULES = {                   # ambang status daya saing (bisa diubah)
    "below_tidak": 25,             # % room night lebih murah dari corp rate  > 25  -> Tidak kompetitif
    "pos_tidak": 75,               # persentil corp rate di kota          > P75 -> Tidak kompetitif
    "below_tinjau": 10,            # >= 10%                                   -> Perlu ditinjau
    "pos_tinjau": 60,              # > P60                                    -> Perlu ditinjau
}
COMPETITIVE_GREEN = ui.POS_TEXT     # warna teks corp rate berstatus Kompetitif
NOT_COMPETITIVE_RED = ui.NEG_TEXT   # warna teks corp rate berstatus Tidak kompetitif
MIN_MARKET_SAMPLE = 5              # minimal booking pembanding untuk menghitung persentil kota

CORP_COLUMNS = {                   # nama kolom standar -> alias yang diterima di file corp rate
    "Hotel Name": ["nama hotel", "hotel name", "hotel"],
    "City": ["kota/kabupaten", "kota / kabupaten", "kota kabupaten", "kabupaten/kota", "city", "kota"],
    "Province": ["provinsi", "province"],
    "Room Type": ["type room", "room type", "tipe kamar", "room category", "jenis kamar"],
    "Meal Plan": ["meal plan", "meal", "board"],
    "Corp Rate": ["corp rate", "corporate rate", "rate", "harga"],
    "Tax Basis": ["tax basis", "basis pajak", "basis"],
    "Valid From": ["valid from", "berlaku dari", "start date"],
    "Valid To": ["valid to", "berlaku sampai", "end date"],
}

_GENERIC = {"hotel", "hotels", "the", "and", "by", "at", "of", "resort", "resorts",
            "residence", "residences", "inn", "suites", "convention", "center", "centre", "pt", "tbk"}


# ---------------------------------------------------------------------------
# NORMALISASI
# ---------------------------------------------------------------------------
def umap(series: pd.Series, func) -> pd.Series:
    """series.map(func) tetapi func hanya dihitung sekali per nilai unik (puluhan ribu baris -> ratusan nilai)."""
    if series is None or len(series) == 0:
        return series
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    lut = np.empty(len(uniques) + 1, dtype=object)
    for i, u in enumerate(uniques):
        lut[i] = func(u)
    lut[-1] = func(np.nan)                      # kode -1 = NaN
    return pd.Series(lut[codes], index=series.index, name=series.name).infer_objects()


def _top(s, default=""):
    """Nilai paling sering di sebuah grup; aman bila grup kosong / semuanya NaN."""
    vc = s.value_counts()
    return vc.index[0] if len(vc) else default


def _fold(s) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip().lower()


_ROOM_FIX = [  # ejaan yang sering berbeda -> bentuk baku
    (r"\bde\s*-?\s*lux(e)?\b", "deluxe"), (r"\bprivila(ge|dge)\b", "privilege"),
    (r"\bapartement\b", "apartment"), (r"\bguestroom\b", "guest"), (r"\bcondo\s*loft\b", "loft"), (r"\bcondoloft\b", "loft"),
]


def _room_canon(s: str) -> str:
    for pat, rep in _ROOM_FIX:
        s = re.sub(pat, rep, s)
    return s


# Tipe non-standar yang variannya digabung (urutan penting: yang pertama cocok dipakai)
_OTHER_GROUPS = [
    (r"\bstudio\b", "Studio"),
    (r"\btriple\b", "Triple"),
    (r"\bfamily\b", "Family"),
    (r"\b(run of house|assigned on arrival|roh)\b", "Run of House"),
    (r"\bloft\b", "Loft"),
    (r"\b(two|2) beds?\b", "Two Beds"),
    (r"\b(apartment|\d\s*bedroom)\b", "Apartment"),
]


def norm_room_type(value) -> str:
    """Kelompokkan Room Category/Room Type ke tipe standar. Ganti dengan
    room_category_standardizer bila ingin aturan yang sama dengan modul lain."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Lainnya"
    s = _fold(value)
    if not s or s in ("nan", "none", "-"):
        return "Lainnya"
    s = _room_canon(s)
    if "suite" in s or "presidential" in s:
        return "Suite"
    if any(k in s for k in ("executive", "club", "premier", "premium", "business", "privilege")) \
            or re.search(r"\bthe level\b", s):
        return "Executive"
    if "deluxe" in s or "dlx" in s:
        return "Deluxe"
    if re.search(r"\bharris\b", s):      # kamar standar jaringan HARRIS punya kategori sendiri
        return "Harris Room"
    if any(k in s for k in ("superior", "standard", "standar", "moderate", "classic", "std", "spr")):
        return "Superior"
    if re.search(r"\b(smart|pop|single|twin|double|queen|king)\b", s):
        return "Superior"
    return "Lainnya"


_ROOM_NOISE = re.compile(r"\b(rooms?|kamar|include[sd]?|incl|with|without|w o|exclude|excluding|excl|non|no|tanpa|"
                         r"sarapan|bb|breakfast|room only|only|ro|rbf|abf|\d?bf|"
                         r"single|double|twin|king|queen|city view|ocean view|pool view|garden view|view)\b")


def room_key(value) -> str:
    """Kunci tipe kamar untuk pencocokan. Tipe standar (Superior/Deluxe/Executive/Suite/Harris Room) dipakai apa adanya;
    tipe lain memakai namanya sendiri, mis. 'Signature Rooms' -> 'Lainnya · Signature', supaya
    'Raffles Room' tidak dibandingkan dengan 'Signature Room'."""
    rt = norm_room_type(value)
    if rt != "Lainnya":
        return rt
    s = re.sub(r"[^a-z0-9 ]", " ", _room_canon(_fold(value)))
    for pat, name in _OTHER_GROUPS:
        if re.search(pat, s):
            return f"Lainnya · {name}"
    s = re.sub(r"\s+", " ", _ROOM_NOISE.sub(" ", s)).strip()
    return f"Lainnya · {s.title()}" if s and s not in ("nan", "none") else "Lainnya"


def is_other_room(rt) -> bool:
    return str(rt).startswith("Lainnya")


MEAL_ANY = "Semua"   # corp rate tanpa meal plan: berlaku untuk Room Only maupun Breakfast Included


def meal_from_room_text(value) -> str:
    """Meal plan yang tertulis di nama tipe kamar, mis. 'Superior Include Breakfast', 'Deluxe (1BF)', 'Room Only'."""
    m = norm_meal(value)
    return MEAL_ANY if m == "Tidak diketahui" else m


_RO_PAT = re.compile(r"room only|\(ro\)|\bro\b|without breakfast|w/o breakfast|wo breakfast|no breakfast|"
                     r"non[- ]?breakfast|excl(?:ude|uding)?\.? breakfast|tanpa sarapan|no meal|room only")
_BF_PAT = re.compile(r"breakfast|sarapan|\b\d ?bf\b|\(\d?bf\)|\brbf\b|\babf\b|\bbf\b|\bbb\b|b&b|"
                     r"half board|full board|all inclusive")


def norm_meal(value) -> str:
    """Meal plan booking: 'Room Only', 'Breakfast Included', atau 'Tidak diketahui'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Tidak diketahui"
    s = _fold(value)
    if not s or s in ("nan", "none", "-"):
        return "Tidak diketahui"
    if _RO_PAT.search(s):
        return "Room Only"
    if _BF_PAT.search(s):
        return "Breakfast Included"
    return "Tidak diketahui"


def safe_standardize(name, standardize_name=None) -> str:
    """Nama hotel sesudah hotel_name_standardizer.standardize_name (kalau gagal, nama asli dirapikan)."""
    base = re.sub(r"\s+", " ", str(name)).strip()
    if standardize_name is None:
        return base
    try:
        out = standardize_name(base)
        return re.sub(r"\s+", " ", str(out)).strip() if out is not None and str(out).strip() else base
    except Exception:
        return base


def hotel_key(name, standardize_name=None) -> str:
    if standardize_name is not None:
        try:
            name = standardize_name(name)
        except Exception:
            pass
    s = re.sub(r"[^a-z0-9 ]", " ", _fold(name))
    toks = [t for t in s.split() if t not in _GENERIC]
    return " ".join(toks) if toks else s.strip()


def _to_number(v) -> float:
    """Angka dari sel Excel (850000 / 850000.0) atau teks ('Rp 850.000', '850,000', '850.000,50')."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = re.sub(r"[^0-9.,]", "", str(v))
    if not s:
        return np.nan
    if "," in s and "." in s:                      # 850.000,50 atau 850,000.50
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif s.count(".") > 1 or s.count(",") > 1:     # 1.250.000 / 1,250,000
        s = s.replace(".", "").replace(",", "")
    elif re.fullmatch(r"\d{1,3}[.,]\d{3}", s):     # 850.000 / 850,000
        s = s.replace(".", "").replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_dt(series) -> pd.Series:
    """Tanggal ISO (2026-01-31) maupun gaya Indonesia (31/01/2026)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = pd.to_datetime(series, errors="coerce", format="ISO8601")
        dmy = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return iso.fillna(dmy)


def _pick_columns(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    lookup = {_fold(c): c for c in df.columns}
    ren = {}
    for target, aliases in spec.items():
        for a in [target.lower()] + aliases:
            if a in lookup:
                ren[lookup[a]] = target
                break
    return df.rename(columns=ren)


# ---------------------------------------------------------------------------
# PERSIAPAN DATA
# ---------------------------------------------------------------------------
def prepare_bookings(df: pd.DataFrame, standardize_name=None, normalize_city=None) -> pd.DataFrame:
    """Satu baris per booking dengan rate per malam (Rupiah) dan kunci pencocokan."""
    need = ["Hotel Name", "City", "Number of Rooms Night", "Invoice Amount"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return pd.DataFrame()
    b = df.copy()
    b["Number of Rooms Night"] = pd.to_numeric(b["Number of Rooms Night"], errors="coerce")
    b["Invoice Amount"] = pd.to_numeric(b["Invoice Amount"], errors="coerce")
    b = b[(b["Number of Rooms Night"] > 0) & b["Invoice Amount"].notna() & (b["Invoice Amount"] > 0)]
    b = b[b["Hotel Name"].notna() & (b["Hotel Name"].astype(str).str.strip() != "")]
    if b.empty:
        return pd.DataFrame()
    ncity = normalize_city or (lambda c: _fold(c).upper())
    b["Hotel Name"] = umap(b["Hotel Name"], lambda v: safe_standardize(v, standardize_name))
    b["_hotel_key"] = umap(b["Hotel Name"], hotel_key)
    b["Hotel Name"] = b.groupby("_hotel_key")["Hotel Name"].transform(lambda x: _top(x, x.iloc[0]))
    b["_city_key"] = umap(b["City"], ncity)
    b["_rt"] = umap(b["Room Category"], room_key) if "Room Category" in b.columns else "Lainnya"
    # meal plan: kolom Meal Plan; bila kosong/tidak terbaca, dari keterangan di Room Category
    meal = umap(b["Meal Plan"], norm_meal) if "Meal Plan" in b.columns else pd.Series("Tidak diketahui", index=b.index)
    if "Room Category" in b.columns:
        from_room = umap(b["Room Category"], norm_meal)
        meal = meal.where(meal != "Tidak diketahui", from_room)
    b["_meal"] = meal
    b["_rate"] = b["Invoice Amount"] / b["Number of Rooms Night"]
    b["_checkin"] = _to_dt(b["Check in Date"]) if "Check in Date" in b.columns else pd.NaT
    b["_bid"] = np.arange(len(b))
    return b.reset_index(drop=True)


YEAR_COL = re.compile(r"(?:corp(?:orate)?\s*rate|harga|rate)\D*((?:19|20)\d{2})(?:\.\d+)?$")


def _year_columns(df: pd.DataFrame) -> dict:
    """{tahun: [kolom, ...]} untuk kolom 'Corporate Rate 2024', 'Corporate Rate 2025', ...
    Kolom kembar (mis. dua 'Corporate Rate 2026') dikelompokkan ke tahun yang sama."""
    out = {}
    for c in df.columns:
        m = YEAR_COL.search(_fold(c))
        if m:
            out.setdefault(int(m.group(1)), []).append(c)
    return dict(sorted(out.items()))


def _cell_rate(v):
    """(angka, belum_pajak?) dari sel corp rate. '-', 'upon request', kosong -> NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, False
    if isinstance(v, str):
        t = _fold(v)
        if not re.search(r"\d", t):
            return np.nan, False
        return _to_number(t), "++" in t
    return _to_number(v), False


def corp_to_long(corp: pd.DataFrame, carry_forward: bool = True) -> tuple[pd.DataFrame, list]:
    """Format MasterData (satu kolom 'Corporate Rate <tahun>' per tahun) -> satu baris per tahun
    dengan Valid From/Valid To = 1 Jan - 31 Des. Bila carry_forward, tahun yang kosong memakai rate
    tahun sebelumnya (sampai tahun terakhir di file). Versi vektor (tanpa loop per baris)."""
    years = _year_columns(corp)
    issues = []
    if not years:
        return corp, issues
    dup = [y for y, cols in years.items() if len(cols) > 1]
    if dup:
        issues.append("Kolom tahun kembar digabung: " + ", ".join(f"Corporate Rate {y}" for y in dup)
                      + " (dipakai nilai pertama yang terisi).")
    base_cols = [c for c in corp.columns if not any(c in cols for cols in years.values())]
    base = corp[base_cols].reset_index(drop=True)
    V, P = {}, {}
    for y, cols in years.items():
        parsed = [umap(corp[c].reset_index(drop=True).astype(object), _cell_rate) for c in cols]
        V[y] = pd.concat([p.str[0] for p in parsed], axis=1).bfill(axis=1).iloc[:, 0].astype(float)
        P[y] = pd.concat([p.str[1] for p in parsed], axis=1).fillna(False).any(axis=1)
    V, P = pd.DataFrame(V), pd.DataFrame(P)
    src = pd.DataFrame({y: np.where(V[y].notna(), y, np.nan) for y in V.columns})
    if carry_forward:
        Vf = V.ffill(axis=1)
        src = src.ffill(axis=1)
        Pf = P.where(V.notna()).ffill(axis=1).fillna(False).astype(bool)
        carried = int((Vf.notna() & V.isna()).sum().sum())
    else:
        Vf, Pf, carried = V, P, 0
    parts = []
    for y in V.columns:
        ok = Vf[y].notna()
        if not ok.any():
            continue
        part = base[ok].copy()
        part["_row"] = np.flatnonzero(ok.to_numpy())
        part["Corp Rate"] = Vf.loc[ok, y].values
        part["Valid From"] = pd.Timestamp(int(y), 1, 1)
        part["Valid To"] = pd.Timestamp(int(y), 12, 31)
        part["Tahun rate"] = src.loc[ok, y].astype(int).values
        part["_plus"] = Pf.loc[ok, y].values
        parts.append(part)
    out = (pd.concat(parts, ignore_index=True).sort_values(["_row", "Valid From"], kind="stable")
           .drop(columns="_row").reset_index(drop=True)) if parts else base.iloc[0:0]   # urutan sama: baris, lalu tahun
    yrs = list(V.columns)
    issues.append(f"Format per tahun terbaca: {yrs[0]}–{yrs[-1]} · {len(base):,} baris tipe kamar → "
                  f"{len(out):,} corp rate per tahun.".replace(",", "."))
    if carried:
        issues.append(f"{carried:,} sel tahun kosong memakai rate tahun sebelumnya.".replace(",", "."))
    return out, issues


def prepare_corp(corp: pd.DataFrame, default_basis: str = "Nett",
                 standardize_name=None, normalize_city=None, carry_forward: bool = True) -> tuple[pd.DataFrame, list]:
    """Rapikan file corp rate (format MasterData per tahun, atau format lama satu kolom Corp Rate).
    Return (tabel, daftar catatan)."""
    issues = []
    if corp is None or corp.empty:
        return pd.DataFrame(), ["File corp rate kosong."]
    c = _pick_columns(corp, CORP_COLUMNS)
    c, notes = corp_to_long(c, carry_forward)
    issues += notes
    for col in ("Hotel Name", "Corp Rate"):
        if col not in c.columns:
            issues.append(f"Kolom wajib '{col}' tidak ditemukan." if col == "Hotel Name" else
                          "Kolom harga tidak ditemukan (contoh: 'Corporate Rate 2026').")
    if any(i.startswith("Kolom wajib") or i.startswith("Kolom harga") for i in issues):
        return pd.DataFrame(), issues
    if "City" not in c.columns:
        c["City"] = ""
    if "Room Type" not in c.columns:
        c["Room Type"] = "Superior"
        issues.append("Kolom Type Room tidak ada: semua corp rate dianggap Superior.")
    c = c[c["Hotel Name"].notna() & (c["Hotel Name"].astype(str).str.strip() != "")].copy()
    if "_plus" not in c.columns:
        parsed = c["Corp Rate"].map(_cell_rate)
        c["Corp Rate"], c["_plus"] = parsed.map(lambda t: t[0]), parsed.map(lambda t: t[1])
    c = c[c["Corp Rate"] > 0].copy()
    # meal plan: kolom Meal Plan bila terisi, lalu teks di Type Room, selain itu berlaku untuk semua
    meal_col = c["Meal Plan"] if "Meal Plan" in c.columns else pd.Series(np.nan, index=c.index)
    from_col = umap(meal_col, lambda v: norm_meal(v) if pd.notna(v) and str(v).strip() not in ("", "-") else MEAL_ANY)
    from_col = from_col.replace("Tidak diketahui", MEAL_ANY)
    from_room = umap(c["Room Type"], meal_from_room_text)
    c["_meal"] = np.where(from_col != MEAL_ANY, from_col, from_room)
    # pajak: '++' di sel, atau kolom Tax Basis, atau pilihan basis di aplikasi
    basis = c["Tax Basis"].astype(str) if "Tax Basis" in c.columns else pd.Series(default_basis, index=c.index)
    excl = basis.map(_fold).str.contains("excl|belum|before|\\+\\+", regex=True) | c["_plus"].astype(bool)
    c["_corp"] = np.where(excl, c["Corp Rate"] * (1 + DEFAULT_TAX), c["Corp Rate"])
    ncity = normalize_city or (lambda v: _fold(v).upper())
    c["Nama hotel di master"] = c["Hotel Name"].astype(str).str.strip()
    c["Hotel Name"] = umap(c["Nama hotel di master"], lambda v: safe_standardize(v, standardize_name))
    c["_hotel_key"] = umap(c["Hotel Name"], hotel_key)
    c["_city_key"] = umap(c["City"], lambda v: ncity(v) if pd.notna(v) else "")
    c["_rt"] = umap(c["Room Type"], room_key)
    c["_from"] = _to_dt(c["Valid From"]) if "Valid From" in c.columns else pd.NaT
    c["_to"] = _to_dt(c["Valid To"]) if "Valid To" in c.columns else pd.NaT
    n_pl = int(c["_plus"].astype(bool).sum())
    if n_pl:
        issues.append(f"{n_pl} corp rate bertanda '++' dianggap belum pajak (+{DEFAULT_TAX * 100:.0f}%).")
    unk = int(c["_rt"].map(is_other_room).sum())
    if unk:
        issues.append(f"{unk:,}".replace(",", ".") + " corp rate dengan tipe kamar di luar Superior/Deluxe/"
                      "Executive/Suite (mis. Family, Studio). Tipe ini hanya dicocokkan dengan booking "
                      "yang nama tipe kamarnya sama.")
    return c.reset_index(drop=True), issues


NAME_MATCH_MIN = 0.88   # kemiripan minimal untuk menyamakan nama hotel master corp rate dengan nama di invoice


def _tokens(key: str) -> set:
    return {t for t in str(key).split() if len(t) > 1}


def harmonize_corp_names(b: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Samakan nama hotel di master corp rate dengan nama hotel di invoice (Google Drive) yang sudah
    diseragamkan hotel_name_standardizer. Pertama lewat kunci nama yang sama persis; sisanya lewat
    kemiripan nama (>= NAME_MATCH_MIN) dengan syarat kotanya sama/berbagi kata dan semua kata pembeda
    di nama yang lebih pendek ada di nama yang lebih panjang.
    Return (corp yang sudah diseragamkan, tabel perubahan nama)."""
    if b is None or b.empty or c.empty:
        return c, pd.DataFrame()
    from difflib import SequenceMatcher
    c = c.copy()
    names = (b.groupby("_hotel_key")["Hotel Name"].agg(lambda s: _top(s, s.iloc[0])))
    bcity = b.groupby("_hotel_key")["_city_key"].agg(lambda s: set(s.dropna()))
    known = set(names.index)
    city_words = {k: set(" ".join(v).lower().split()) for k, v in bcity.items()}
    by_word = {}
    for k in known:
        for w in city_words.get(k, set()):
            by_word.setdefault(w, set()).add(k)
    changes = []
    for ck in sorted(set(c["_hotel_key"]) - known):
        rows = c[c["_hotel_key"] == ck]
        cw = set(" ".join(rows["_city_key"].astype(str)).lower().split())
        cand = set().union(*[by_word.get(w, set()) for w in cw]) if cw else set()
        best, score = None, 0.0
        strip = lambda k, ws: " ".join(t for t in k.split() if t not in ws) or k  # noqa: E731
        for bk in cand:
            ws = cw | city_words.get(bk, set())          # nama kota di nama hotel tidak ikut dibandingkan
            a, z = strip(ck, ws), strip(bk, ws)
            sm = SequenceMatcher(None, a, z)
            floor = max(score, NAME_MATCH_MIN - 1e-9)
            if sm.real_quick_ratio() <= floor or sm.quick_ratio() <= floor:
                continue                                  # batas atas kemiripan tidak cukup -> lewati (hasil sama)
            r = sm.ratio()
            short, long_ = sorted((_tokens(a), _tokens(z)), key=len)
            if r > score and short and short <= long_ | {t for t in short if len(t) <= 2}:
                best, score = bk, r
        if best is not None and score >= NAME_MATCH_MIN:
            c.loc[rows.index, "_hotel_key"] = best
            changes.append((rows["Nama hotel di master"].iat[0], names[best], "mirip", round(score, 2)))
    c["Hotel Name"] = c["_hotel_key"].map(names).fillna(c["Hotel Name"])
    diff = c["Hotel Name"].map(lambda v: _fold(v)) != c["Nama hotel di master"].map(lambda v: _fold(v))
    std = c[diff].drop_duplicates("Nama hotel di master")  # beda huruf besar/kecil saja tidak dicatat
    done = {x[0] for x in changes}
    for _, r in std.iterrows():
        if r["Nama hotel di master"] not in done:
            changes.append((r["Nama hotel di master"], r["Hotel Name"], "standardizer / sama dengan invoice", 1.0))
    ch = pd.DataFrame(changes, columns=["Nama di master corp rate", "Nama seragam (dipakai)", "Cara", "Kemiripan"])
    return c, ch.sort_values(["Cara", "Nama di master corp rate"]).reset_index(drop=True)


ANALYSIS_MEAL = "Breakfast Included"   # analisis Cost Avoidance hanya untuk booking termasuk sarapan


def breakfast_only(b: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Booking termasuk sarapan saja, dan corp rate yang berlaku untuk sarapan
    (Meal Plan 'Breakfast' atau kosong). Corp rate khusus Room Only tidak dipakai."""
    bb = b[b["_meal"] == ANALYSIS_MEAL].reset_index(drop=True) if not b.empty else b
    cc = c[c["_meal"].isin([ANALYSIS_MEAL, MEAL_ANY])].reset_index(drop=True) if not c.empty else c
    return bb, cc


def align_corp_cities(b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Kota di daftar corp rate sering ditulis beda dengan invoice (Jakarta Selatan vs Jakarta, Badung vs Bali).
    Untuk hotel yang ada di invoice, kota corp rate disamakan dengan kota di invoice bila:
    hotel itu hanya punya satu kota di daftar corp rate, atau nama kotanya berbagi kata (Jakarta Selatan ~ Jakarta)."""
    if b is None or b.empty or c.empty:
        return c
    c = c.copy()
    bc = (b.groupby(["_hotel_key", "_city_key"])["Number of Rooms Night"].sum().reset_index()
          .sort_values("Number of Rooms Night", ascending=False))
    b_cities = bc.groupby("_hotel_key", sort=False)["_city_key"].apply(list).to_dict()
    n_city = c.groupby("_hotel_key")["_city_key"].nunique().to_dict()

    def pick(h, k):
        cand = b_cities.get(h)
        if not cand or k in cand:
            return k
        kw = set(str(k).lower().split())
        for bk in cand:
            if kw & set(str(bk).lower().split()):
                return bk
        return cand[0] if n_city.get(h, 0) <= 1 else k

    c["_city_key"] = [pick(h, k) for h, k in zip(c["_hotel_key"], c["_city_key"])]
    return c


def match_bookings(b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan corp rate yang cocok (hotel + kota + tipe kamar + meal plan + masa berlaku)."""
    if b.empty:
        return b
    out = b.copy()
    out["_corp"] = np.nan
    out["_match"] = "Hotel non-kontrak"
    if c.empty:
        return out
    keys = ["_hotel_key", "_city_key", "_rt"]
    m = out[["_bid", "_checkin", "_meal"] + keys].merge(
        c[keys + ["_meal", "_corp", "_from", "_to"]].rename(columns={"_meal": "_cmeal"}), on=keys, how="inner")
    ok = (m["_from"].isna() | (m["_checkin"] >= m["_from"])) & (m["_to"].isna() | (m["_checkin"] <= m["_to"]))
    ok &= (m["_cmeal"] == m["_meal"]) | (m["_cmeal"] == MEAL_ANY)
    m = m[ok].assign(_exact=lambda x: (x["_cmeal"] == x["_meal"]).astype(int))
    m = m.sort_values(["_exact", "_corp"], ascending=[False, True]).drop_duplicates("_bid")
    out.loc[out["_bid"].isin(m["_bid"]), "_corp"] = out["_bid"].map(m.set_index("_bid")["_corp"])
    hotels = set(zip(c["_hotel_key"], c["_city_key"]))
    hotel_rt = set(zip(c["_hotel_key"], c["_city_key"], c["_rt"]))
    is_hotel = [(h, k) in hotels for h, k in zip(out["_hotel_key"], out["_city_key"])]
    is_rt = [(h, k, r) in hotel_rt for h, k, r in zip(out["_hotel_key"], out["_city_key"], out["_rt"])]
    out["_contract_hotel"] = is_hotel
    out.loc[out["_corp"].notna(), "_match"] = "Cocok"
    rest = out["_corp"].isna() & out["_contract_hotel"]
    other = out["_rt"].map(is_other_room)
    out.loc[rest & other & ~pd.Series(is_rt, index=out.index), "_match"] = "Tipe kamar tidak dikenali"
    out.loc[rest & ~other & ~pd.Series(is_rt, index=out.index), "_match"] = "Tipe kamar tidak ada di kontrak"
    out.loc[rest & pd.Series(is_rt, index=out.index) & out["_match"].eq("Hotel non-kontrak"), "_match"] = \
        "Meal plan / masa berlaku tidak cocok"
    return out


# ---------------------------------------------------------------------------
# PERHITUNGAN
# ---------------------------------------------------------------------------
def _market_percentile(market: pd.DataFrame, city, rt, meal, value) -> float:
    pool = market.get((city, rt, meal))
    if pool is None or len(pool) < MIN_MARKET_SAMPLE:
        return np.nan
    return float((pool < value).mean() * 100)


def market_pools(b: pd.DataFrame) -> dict:
    """Distribusi rate dibayar per (kota, tipe kamar, meal plan) — basis persentil."""
    return {k: g["_rate"].to_numpy() for k, g in b.groupby(["_city_key", "_rt", "_meal"])}


def status_of(below_pct, pos) -> str:
    r = STATUS_RULES
    pos = -1 if pd.isna(pos) else pos
    if below_pct > r["below_tidak"] or pos > r["pos_tidak"]:
        return "Tidak kompetitif"
    if below_pct >= r["below_tinjau"] or pos > r["pos_tinjau"]:
        return "Perlu ditinjau"
    return "Kompetitif"


def summarize_matched(m: pd.DataFrame, pools: dict) -> pd.DataFrame:
    """Satu baris per hotel × tipe kamar × meal plan yang punya corp rate."""
    x = m[m["_corp"].notna()].copy()
    if x.empty:
        return pd.DataFrame()
    x["_rn"] = x["Number of Rooms Night"]
    x["_above_val"] = np.clip(x["_rate"] - x["_corp"], 0, None) * x["_rn"]
    x["_rn_below"] = np.where(x["_rate"] < x["_corp"], x["_rn"], 0)
    x["_rn_above"] = np.where(x["_rate"] > x["_corp"], x["_rn"], 0)
    g = x.groupby(["_hotel_key", "_city_key", "_rt", "_meal"]).agg(
        Hotel=("Hotel Name", _top),
        Kota=("City", _top),
        Corp=("_corp", "median"), Inv=("Invoice Amount", "sum"), RN=("_rn", "sum"),
        Potensi=("_above_val", "sum"), RN_below=("_rn_below", "sum"), RN_above=("_rn_above", "sum"),
        Booking=("_bid", "nunique")).reset_index()
    g["Rata-rata dibayar"] = g["Inv"] / g["RN"]
    g["Selisih %"] = (g["Rata-rata dibayar"] - g["Corp"]) / g["Corp"] * 100
    g["Room Night < Corp %"] = g["RN_below"] / g["RN"] * 100
    g["Posisi kota"] = [_market_percentile(pools, k, r, ml, v)
                        for k, r, ml, v in zip(g["_city_key"], g["_rt"], g["_meal"], g["Corp"])]
    g["Daya saing"] = [status_of(bl, p) for bl, p in zip(g["Room Night < Corp %"], g["Posisi kota"])]
    hotel_rn = g.groupby(["_hotel_key", "_city_key"])["RN"].transform("sum")
    g["_hotel_rn"] = hotel_rn
    g = g.rename(columns={"_rt": "Room type", "_meal": "Meal plan"})
    return g


def top_hotels(summary: pd.DataFrame, n: int) -> pd.DataFrame:
    if summary.empty:
        return summary
    ranks = (summary.groupby(["_hotel_key", "_city_key"])["_hotel_rn"].first()
             .sort_values(ascending=False).head(n).index)
    keep = summary.set_index(["_hotel_key", "_city_key"]).index.isin(ranks)
    return summary[keep].copy()


def leakage(m: pd.DataFrame, c: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Booking di hotel tanpa corp rate, di kota yang punya hotel ber-corp rate."""
    if m.empty or c.empty:
        return pd.DataFrame()
    cities = set(c["_city_key"])
    x = m[(~m["_contract_hotel"]) & m["_city_key"].isin(cities)].copy()
    if x.empty:
        return pd.DataFrame()
    eq3 = c.groupby(["_city_key", "_rt", "_meal"])["_corp"].median()
    eq2 = c.groupby(["_city_key", "_rt"])["_corp"].median()
    eq1 = c.groupby("_city_key")["_corp"].median()
    x["_eq"] = [eq3.get((k, r, ml), eq2.get((k, r), eq1.get(k, np.nan)))
                for k, r, ml in zip(x["_city_key"], x["_rt"], x["_meal"])]
    x["_pot"] = np.clip(x["_rate"] - x["_eq"], 0, None) * x["Number of Rooms Night"]
    g = x.groupby(["_hotel_key", "_city_key", "_rt", "_meal"]).agg(
        Hotel=("Hotel Name", _top),
        Kota=("City", _top),
        RN=("Number of Rooms Night", "sum"), Inv=("Invoice Amount", "sum"),
        Setara=("_eq", "median"), Potensi=("_pot", "sum")).reset_index()
    g["Rata-rata dibayar"] = g["Inv"] / g["RN"]
    g["Selisih %"] = (g["Rata-rata dibayar"] - g["Setara"]) / g["Setara"] * 100
    g = g.rename(columns={"_rt": "Room type", "_meal": "Meal plan"})
    g = g.sort_values("RN", ascending=False)
    return (g if top_n is None else g.head(top_n)).reset_index(drop=True)


def renegotiation(summary: pd.DataFrame, pools: dict, pct: int, manual: dict | None = None,
                  top_n: int = 100) -> pd.DataFrame:
    """Target renegosiasi untuk corp rate yang tidak/kurang kompetitif.
    manual: {kunci_baris: persen_turun} menggantikan target persentil."""
    if summary.empty:
        return pd.DataFrame()
    s = summary[summary["Daya saing"] != "Kompetitif"].copy()
    if s.empty:
        return s
    s = s.sort_values("RN", ascending=False).head(top_n)
    s["_key"] = s["_hotel_key"] + "|" + s["_city_key"] + "|" + s["Room type"] + "|" + s["Meal plan"]

    def _pt(row):
        pool = pools.get((row["_city_key"], row["Room type"], row["Meal plan"]))
        if pool is None or len(pool) < MIN_MARKET_SAMPLE:
            return np.nan
        return float(np.percentile(pool, pct))

    s["Target persentil"] = s.apply(_pt, axis=1)
    manual = manual or {}
    s["Target turun manual (%)"] = s["_key"].map(lambda k: manual.get(k, np.nan))
    use_manual = s["Target turun manual (%)"].between(0.01, 99.99)
    s["Target dipakai"] = np.where(use_manual, s["Corp"] * (1 - s["Target turun manual (%)"] / 100),
                                   s["Target persentil"])
    s["Sumber target"] = np.where(use_manual, "manual", f"P{pct} kota")
    s["Turun %"] = (s["Target dipakai"] - s["Corp"]) / s["Corp"] * 100
    s["Hard savings"] = np.clip(s["Corp"] - s["Target dipakai"], 0, None) * s["RN"]
    s.loc[s["Target dipakai"].isna(), "Sumber target"] = "data pembanding kurang"
    return s.reset_index(drop=True)


def city_compliance(m: pd.DataFrame, c: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if m.empty:
        return pd.DataFrame()
    cities = set(c["_city_key"]) if not c.empty else set()
    x = m[m["_city_key"].isin(cities)].copy()
    if x.empty:
        return pd.DataFrame()
    rn = x["Number of Rooms Night"]
    x["_ok"] = np.where(x["_corp"].notna() & (x["_rate"] <= x["_corp"]), rn, 0)
    x["_matched"] = np.where(x["_corp"].notna(), rn, 0)
    x["_noncon"] = np.where(~x["_contract_hotel"], rn, 0)
    g = x.groupby("_city_key").agg(Kota=("City", _top),
                                   RN=("Number of Rooms Night", "sum"), OK=("_ok", "sum"),
                                   Matched=("_matched", "sum"), Non=("_noncon", "sum")).reset_index()
    g["Kepatuhan %"] = np.where(g["Matched"] > 0, g["OK"] / g["Matched"] * 100, np.nan)
    g["Non-kontrak %"] = g["Non"] / g["RN"] * 100
    return g.sort_values("RN", ascending=False).head(top_n).reset_index(drop=True)


def uncovered_room_types(m: pd.DataFrame, c: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Booking di hotel ber-corp rate yang TIDAK punya corp rate untuk tipe kamarnya
    (tipe kamar tidak ada di kontrak / tidak dikenali / meal plan atau masa berlaku tidak cocok)."""
    if m.empty or c.empty:
        return pd.DataFrame()
    x = m[m["_contract_hotel"] & m["_corp"].isna()]
    if x.empty:
        return pd.DataFrame()
    hk = ["_hotel_key", "_city_key"]
    hotel_rn = m[m["_contract_hotel"]].groupby(hk)["Number of Rooms Night"].sum().rename("_hrn")
    g = x.groupby(hk + ["_rt"]).agg(Hotel=("Hotel Name", _top), Kota=("City", _top), Alasan=("_match", _top),
                                     RN=("Number of Rooms Night", "sum"), Inv=("Invoice Amount", "sum"),
                                     Booking=("_bid", "nunique")).reset_index()
    g = g.join(hotel_rn, on=hk)
    g["Rata-rata dibayar"] = g["Inv"] / g["RN"]
    g["Porsi di hotel %"] = g["RN"] / g["_hrn"] * 100
    cc = c.groupby(hk).agg(Corp_min=("_corp", "min"),
                           Tipe_kontrak=("_rt", lambda s_: ", ".join(sorted(set(map(str, s_))))))
    g = g.join(cc, on=hk)
    g["Vs corp termurah %"] = (g["Rata-rata dibayar"] - g["Corp_min"]) / g["Corp_min"] * 100
    g = g.rename(columns={"_rt": "Tipe kamar", "Corp_min": "Corp termurah hotel", "Tipe_kontrak": "Tipe di kontrak"})
    return g.sort_values("RN", ascending=False).head(top_n).reset_index(drop=True)


def match_quality(m: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ringkasan status pencocokan + daftar nama hotel non-kontrak yang mirip hotel kontrak."""
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = (m.groupby("_match")["Number of Rooms Night"].sum() / m["Number of Rooms Night"].sum() * 100)
    q = q.rename("Porsi room night %").reset_index().rename(columns={"_match": "Status"})
    suspects = pd.DataFrame()
    if not c.empty:
        from difflib import SequenceMatcher
        nc = m[~m["_contract_hotel"]].groupby(["_hotel_key", "_city_key"]).agg(
            Hotel=("Hotel Name", "first"), Kota=("City", "first"),
            RN=("Number of Rooms Night", "sum")).reset_index()
        ref_by_city = {k: g.drop_duplicates("_hotel_key") for k, g in c.groupby("_city_key")}
        rows = []
        for _, r in nc.iterrows():
            ref = ref_by_city.get(r["_city_key"])
            if ref is None:
                continue
            scores = [SequenceMatcher(None, r["_hotel_key"], h).ratio() for h in ref["_hotel_key"]]
            j = int(np.argmax(scores))
            if scores[j] >= 0.8:
                rows.append({"Nama di invoice": r["Hotel"], "Kota": r["Kota"],
                             "Mirip hotel kontrak": ref.iloc[j]["Hotel Name"],
                             "Kemiripan": round(float(scores[j]), 2), "Room night": int(r["RN"])})
        if rows:
            suspects = pd.DataFrame(rows).sort_values("Room night", ascending=False).reset_index(drop=True)
    return q, suspects


# ---------------------------------------------------------------------------
# PENYIMPANAN (file corp rate & target manual) — di folder data server, bukan di repo
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    base = os.environ.get("MTRAX_DATA_DIR", "")
    try:
        import streamlit as st
        base = str(dict(st.secrets.get("data", {})).get("dir", "") or base)
    except Exception:
        pass
    p = Path(base or os.path.expanduser("~/.local/share/mtrax"))
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    return p


def corp_path() -> Path:
    return data_dir() / "corp_rate.xlsx"


def manual_path() -> Path:
    return data_dir() / "reneg_manual_targets.csv"


def load_manual() -> pd.DataFrame:
    p = manual_path()
    if p.exists():
        try:
            return pd.read_csv(p, dtype={"key": str})
        except Exception:
            pass
    return pd.DataFrame(columns=["key", "persen", "alasan", "oleh", "tanggal"])


def save_manual(df: pd.DataFrame):
    p = manual_path()
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def read_corp_file(path) -> pd.DataFrame:
    """Baca sheet 'Corp Rate' bila ada, selain itu sheet pertama."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheet = next((n for n in xl.sheet_names if _fold(n) in ("corp rate", "corporate rate")), xl.sheet_names[0])
    return xl.parse(sheet)


# ---------------------------------------------------------------- sumber corp rate & pipeline ber-cache
def frame_sig(df: pd.DataFrame) -> str:
    """Sidik jari cepat sebuah DataFrame (untuk kunci cache)."""
    if df is None or df.empty:
        return "empty"
    try:
        h = pd.util.hash_pandas_object(df, index=False).to_numpy()
        return f"{len(df)}:{int(h.sum()) & 0xFFFFFFFFFFFF:x}:{int(h[::97].sum()) & 0xFFFFFF:x}"
    except Exception:
        return f"{len(df)}:{id(df)}"


def _read_corp_cached(path: str, mtime: float) -> pd.DataFrame:
    import streamlit as st

    @st.cache_data(max_entries=4, show_spinner=False)
    def _r(p, m):
        return read_corp_file(p)
    return _r(path, mtime)


def local_corp_source():
    """Sumber cadangan: file corp_rate.xlsx yang diunggah ke server lewat aplikasi."""
    cp = corp_path()
    if not cp.exists():
        return None
    mt = cp.stat().st_mtime
    return {"df": _read_corp_cached(str(cp), mt), "name": "corp_rate.xlsx", "origin": "server",
            "updated": datetime.fromtimestamp(mt), "sig": f"file:{mt}", "error": None}


def corp_template() -> BytesIO:
    """Template mengikuti format MasterData: satu baris per hotel × tipe kamar, satu kolom per tahun."""
    t = pd.DataFrame([
        ["Raffles Jakarta", "Jakarta Selatan", "DKI Jakarta", "Raffles Room", "Breakfast", 2200000, 2400000, 2500000],
        ["Raffles Jakarta", "Jakarta Selatan", "DKI Jakarta", "Signature Rooms", None, 3050000, 3250000, 3250000],
        ["Vega Hotel Gading Serpong", "Tangerang", "Banten", "Superior Single / Twin", None, 588000, 600000, "-"],
        ["Vega Hotel Gading Serpong", "Tangerang", "Banten", "Deluxe King", None, 750000, 762000, "-"],
        ["Grand Jatra Hotel", "Pekanbaru", "Riau", "Superior Room", None, 998000, "-", 1250000],
        ["Grand Jatra Hotel", "Pekanbaru", "Riau", "Deluxe Room", None, 1128000, "-", 1420000],
    ], columns=["Nama Hotel", "Kota/Kabupaten", "Provinsi", "Type Room", "Meal Plan",
                "Corporate Rate 2024", "Corporate Rate 2025", "Corporate Rate 2026"])
    notes = pd.DataFrame({"Petunjuk": [
        "Satu baris per hotel × tipe kamar. Tambah kolom 'Corporate Rate <tahun>' untuk tahun berikutnya.",
        "Rate tahun X berlaku untuk check-in 1 Januari – 31 Desember tahun X.",
        "Sel kosong atau '-' = tidak ada rate tahun itu; aplikasi memakai rate tahun sebelumnya.",
        "Teks seperti 'upon request' atau 'ask to hotel' diabaikan.",
        "Tulis '++' di belakang angka (mis. 1.100.000 ++) bila rate belum termasuk pajak & service.",
        "Meal Plan boleh kosong: rate berlaku untuk Room Only maupun Breakfast. "
        "Isi 'Breakfast' atau 'Room Only' bila rate khusus salah satu. Teks di Type Room "
        "(mis. 'Include Breakfast', '(1BF)', 'Room Only') juga dibaca.",
        "Type Room dikelompokkan ke Superior / Deluxe / Executive / Suite agar bisa dicocokkan dengan Room Category di invoice.",
        "Kota boleh ditulis sampai kecamatan (mis. Jakarta Selatan); hotel dicocokkan dengan kota di invoice.",
    ]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        t.to_excel(w, index=False, sheet_name="Corp Rate")
        notes.to_excel(w, index=False, sheet_name="Petunjuk")
        w.sheets["Corp Rate"].set_column(0, 0, 30)
        w.sheets["Corp Rate"].set_column(1, 4, 20)
        w.sheets["Corp Rate"].set_column(5, 7, 20, w.book.add_format({"num_format": "#,##0"}))
        w.sheets["Petunjuk"].set_column(0, 0, 120)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# TAMPILAN STREAMLIT
# ---------------------------------------------------------------------------
def _rp(v) -> str:
    return "-" if pd.isna(v) else "Rp" + f"{v:,.0f}".replace(",", ".")


def _jt(v) -> str:
    if pd.isna(v):
        return "-"
    if abs(v) >= 1e9:
        return "Rp" + f"{v / 1e9:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " M"
    return "Rp" + f"{v / 1e6:,.0f}".replace(",", ".") + " jt"


def _pct(v, signed=False) -> str:
    if pd.isna(v):
        return "-"
    s = f"{abs(v):.1f}".replace(".", ",") + "%"
    if signed:
        return ("+" if v > 0 else "−" if v < 0 else "") + s
    return s


def _excel_now(df: pd.DataFrame, sheet: str, excel_fn=None, **fmt) -> BytesIO:
    if excel_fn is not None:
        try:
            return excel_fn(df, sheet, **fmt)
        except Exception:
            pass
    buf = BytesIO()
    df.to_excel(buf, index=False, sheet_name=sheet[:31])
    buf.seek(0)
    return buf


def _excel(df: pd.DataFrame, sheet: str, excel_fn=None, **fmt):
    """Data untuk st.download_button: fungsi yang membuat Excel SAAT tombol diklik (bukan setiap rerun).
    Bila versi Streamlit belum mendukung data berupa fungsi, app.py memanggilnya langsung (lihat download guard)."""
    snapshot = df.copy()
    return lambda: _excel_now(snapshot, sheet, excel_fn, **fmt).getvalue()


def _corp_uploader(cp, where: str):
    """Unggah & simpan file corp rate (dipakai di halaman kosong dan di menu Corp rate)."""
    import streamlit as st
    up = st.file_uploader("Unggah / ganti daftar corp rate (.xlsx)", type=["xlsx"], key=f"ca_corp_up_{where}")
    if up is not None and st.button("Simpan daftar corp rate", key=f"ca_corp_save_{where}", type="primary"):
        tmp = cp.with_suffix(".tmp")
        tmp.write_bytes(up.getvalue())
        os.chmod(tmp, 0o600)
        os.replace(tmp, cp)
        st.cache_data.clear()
        st.success("Daftar corp rate disimpan.")
        st.rerun()


def corp_years(src) -> list:
    """Tahun yang tersedia di daftar corp rate (kolom 'Corporate Rate <tahun>')."""
    try:
        return sorted(_year_columns(src["df"]).keys())
    except Exception:
        return []


def run_analysis(df_all, data_sig, src, basis_flag, carry, standardize_name=None, normalize_city=None,
                 corp_year=None) -> dict:
    """Seluruh perhitungan Cost Avoidance, di-cache per (data, file corp rate, basis, carry).
    Dipakai tab Cost Avoidance dan tab Laporan, dan dibagi antar pengguna."""
    import streamlit as st

    @st.cache_data(max_entries=8, show_spinner=False)
    def _run(_df, dsig, _corp_raw, csig, basis, carry_, _std, _nc, year):
        corp, issues = prepare_corp(_corp_raw, basis, _std, _nc, carry_forward=carry_)
        if year and not corp.empty and "_from" in corp.columns:
            # bandingkan SEMUA booking dengan corp rate satu tahun pilihan (masa berlaku diabaikan)
            corp = corp[corp["_from"].dt.year == int(year)].copy()
            corp["_from"], corp["_to"] = pd.NaT, pd.NaT
        res = {"issues": issues, "ready": False}
        b = prepare_bookings(_df, _std, _nc)
        if b.empty or corp.empty:
            return res
        corp, name_changes = harmonize_corp_names(b, corp)
        corp = align_corp_cities(b, corp)
        n_all, rn_all = len(b), float(b["Number of Rooms Night"].sum())
        n_hotel = int(corp["_hotel_key"].nunique())
        n_in = int(corp.drop_duplicates("_hotel_key")["_hotel_key"].isin(set(b["_hotel_key"])).sum())
        b, corp = breakfast_only(b, corp)
        res.update(name_changes=name_changes, n_all=n_all, rn_all=rn_all, n_hotel=n_hotel, n_in=n_in,
                   n_bf=len(b), rn_bf=float(b["Number of Rooms Night"].sum()) if not b.empty else 0.0)
        if b.empty or corp.empty:
            return res
        m = match_bookings(b, corp)
        pools = market_pools(b)
        res.update(ready=True, corp=corp, m=m, pools=pools, summary=summarize_matched(m, pools),
                   leak100=leakage(m, corp, 100), quality=match_quality(m, corp),
                   city10=city_compliance(m, corp, 10), uncovered=uncovered_room_types(m, corp, None))
        return res

    return _run(df_all, data_sig, src["df"], src["sig"], basis_flag, carry, standardize_name, normalize_city,
                corp_year)


CUM_TARGET = 80   # % potensi yang dijadikan batas fokus


def hotel_cumulative(s: pd.DataFrame) -> pd.DataFrame:
    """Potensi per hotel (gabungan semua tipe kamar), urut terbesar, dengan kontribusi & kumulatif %."""
    x = s[s["Potensi"] > 0]
    if x.empty:
        return pd.DataFrame()
    g = (x.groupby(["Hotel", "Kota"], dropna=False)
         .agg(Potensi=("Potensi", "sum"), RN=("RN", "sum"), Tipe=("Room type", "nunique"),
              Selisih=("Selisih %", "max"))
         .reset_index().sort_values("Potensi", ascending=False).reset_index(drop=True))
    tot = g["Potensi"].sum()
    g["Kontribusi %"] = g["Potensi"] / tot * 100
    g["Kumulatif %"] = g["Kontribusi %"].cumsum().clip(upper=100)
    g.index = g.index + 1
    return g


def _cumulative_view(s: pd.DataFrame, topn: int, can_download: bool, excel_fn):
    """Opsi 7: kurva kumulatif potensi cost avoidance per hotel + daftar hotel fokus."""
    import plotly.graph_objects as go
    import streamlit as st
    g = hotel_cumulative(s)
    if g.empty:
        st.info("Tidak ada hotel yang dibayar di atas corp rate.")
        return
    n80 = int((g["Kumulatif %"] < CUM_TARGET).sum()) + 1
    n80 = min(n80, len(g))
    tot = g["Potensi"].sum()
    rn_tot, rn_above = s["RN"].sum(), s["RN_above"].sum()
    vc = s["Daya saing"].value_counts()
    ui.kpi_row([
        ui.kpi(f"Hotel penyumbang {CUM_TARGET}% potensi", f"{n80} dari {len(g)} hotel",
               "fokus penegakan kontrak", dark=True),
        ui.kpi("Potensi cost avoidance (gross)", _jt(tot), f"{topn} hotel teratas"),
        ui.kpi("Kepatuhan rate", _pct((1 - rn_above / rn_tot) * 100) if rn_tot else "-",
               f"{rn_above:,.0f} dari {rn_tot:,.0f} room night di atas corp rate".replace(",", ".")),
        ui.kpi("Daya saing (hotel × tipe)", f"{vc.get('Kompetitif', 0)} kompetitif",
               f"{vc.get('Perlu ditinjau', 0)} ditinjau · {vc.get('Tidak kompetitif', 0)} tidak"),
    ], 200)

    with st.container(border=True):
        ui.section(f"Kumulatif potensi cost avoidance per hotel · {topn} hotel teratas",
                   "Hotel diurutkan dari potensi terbesar · batang = kontribusi tiap hotel · garis = kumulatif")
        idx = list(range(1, len(g) + 1))
        focus = [i <= n80 for i in idx]
        names = [f"{h} · {k}" if isinstance(k, str) and k else str(h) for h, k in zip(g["Hotel"], g["Kota"])]
        custom = np.stack([names, [_jt(v) for v in g["Potensi"]], g["Tipe"].astype(int),
                           g["Kumulatif %"].round(1)], axis=-1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=idx, y=g["Kontribusi %"], name="Kontribusi hotel",
                             marker_color=[ui.P1 if f else ui.TINT for f in focus], customdata=custom,
                             hovertemplate="#%{x} %{customdata[0]}<br>Potensi %{customdata[1]} · "
                                           "%{y:.1f}% dari total<br>%{customdata[2]} tipe kamar di atas corp rate"
                                           "<br>Kumulatif %{customdata[3]}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=idx, y=g["Kumulatif %"], name="Kumulatif", mode="lines",
                                 line=dict(color=ui.P3, width=2.5, shape="spline", smoothing=0.6),
                                 fill="tozeroy", fillcolor="rgba(13,127,204,0.07)", customdata=custom,
                                 hovertemplate="#%{x} %{customdata[0]}<br>Kumulatif %{y:.1f}%<extra></extra>"))
        y80 = float(g["Kumulatif %"].iloc[n80 - 1])
        fig.add_shape(type="line", x0=0.5, x1=n80, y0=y80, y1=y80, line=dict(color="#9A9A9A", dash="dash", width=1))
        fig.add_shape(type="line", x0=n80, x1=n80, y0=0, y1=y80, line=dict(color="#9A9A9A", dash="dash", width=1))
        fig.add_trace(go.Scatter(x=[n80], y=[y80], mode="markers", marker=dict(color=ui.RED_FILL, size=11),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=n80, y=y80, text=f"<b>{n80} hotel pertama = {y80:.0f}% potensi</b> → fokus di sini",
                           showarrow=False, xanchor="left", yanchor="bottom", xshift=12, yshift=8,
                           font=dict(size=13, color=ui.INK), bgcolor="rgba(255,255,255,0.85)")
        span = max(len(g), 12)          # hotel sedikit -> batang tetap ramping
        fig.update_layout(height=400, bargap=0.25, showlegend=False,
                          **{k: v for k, v in ui.PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
                          margin=dict(l=64, r=24, t=34, b=56),
                          xaxis=dict(title="Jumlah hotel (urut dari potensi terbesar)", gridcolor="#EDEDED",
                                     range=[0.4, span + 0.6], tick0=1, dtick=max(1, round(span / 10)),
                                     tickformat="d", automargin=True),
                          yaxis=dict(title="% dari total potensi", range=[0, 105], ticksuffix="%",
                                     dtick=20, gridcolor="#EDEDED", automargin=True))
        st.plotly_chart(fig, width="stretch", key="ca_cum", config={"displaylogo": False})

        st.markdown(f"**{n80} hotel fokus** · penyumbang {y80:.0f}% potensi")
        top = g.head(n80).rename(columns={"Potensi": "Potensi (Rp)", "RN": "Room night",
                                          "Tipe": "Tipe kamar di atas corp", "Selisih": "Selisih terbesar %"})
        show = top.copy()
        show["Potensi (Rp)"] = show["Potensi (Rp)"].map(_jt)
        show["Room night"] = show["Room night"].map(lambda v: f"{v:,.0f}".replace(",", "."))
        for c in ("Selisih terbesar %", "Kontribusi %", "Kumulatif %"):
            show[c] = show[c].map(lambda v: _pct(v, c.startswith("Selisih")))
        st.dataframe(show, width="stretch", height=min(420, 36 * len(show) + 40))
        if can_download:
            st.download_button("Unduh Excel", _excel(g.rename(columns={"Potensi": "Potensi (Rp)", "RN": "Room night",
                                                                       "Tipe": "Tipe kamar di atas corp",
                                                                       "Selisih": "Selisih terbesar %"}).reset_index(names="Peringkat"),
                                                     "Kumulatif potensi", excel_fn,
                                                     money_cols=("Potensi (Rp)",), int_cols=("Room night",),
                                                     dec_cols=("Selisih terbesar %", "Kontribusi %", "Kumulatif %")),
                               file_name=f"kumulatif_potensi_hotel_{datetime.now():%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ca_dl_cum")
    _chart_guide(g, n80, y80)


def _chart_guide(g: pd.DataFrame, n80: int, y80: float):
    """Narasi cara membaca grafik kumulatif + ringkasan otomatis."""
    import streamlit as st
    with st.expander("Cara membaca grafik", icon=":material/help:"):
        st.markdown('<div class="mx-guide">'
                    "• Setiap <b>batang</b> = satu hotel (semua tipe kamarnya digabung), diurutkan dari potensi cost "
                    "avoidance terbesar ke terkecil. Tinggi batang = porsi hotel itu dari total potensi. Arahkan kursor "
                    "untuk melihat nama hotel, nilai rupiah, dan jumlah tipe kamar yang dibayar di atas corp rate.<br>"
                    "• <b>Garis biru</b> = potensi yang terkumpul bila hotel ditangani berurutan dari atas. Garis naik "
                    "tajam di awal lalu mendatar: sebagian kecil hotel menyumbang sebagian besar potensi.<br>"
                    f"• <b>Titik merah</b> = batas {CUM_TARGET}%. Hotel di sebelah kiri titik (batang biru tua) adalah "
                    "hotel fokus: menegakkan corp rate di hotel-hotel ini menutup sebagian besar potensi. Hotel di "
                    "kanan (batang biru muda) nilainya kecil, cukup dipantau.<br>"
                    "• <b>Potensi</b> = selisih rate dibayar di atas corp rate × room night, hanya untuk booking termasuk "
                    "sarapan. Rincian per tipe kamar ada di tabel <b>Detail per hotel &amp; tipe kamar</b> di bawah."
                    "</div>", unsafe_allow_html=True)
    top3 = g.head(3)
    ui.note(f"<b>Ringkasnya:</b> {n80} dari {len(g)} hotel menyumbang {y80:.0f}% dari total potensi "
            f"<b>{_jt(g['Potensi'].sum())}</b>. Tiga terbesar: " +
            ", ".join(f"{ui.esc(r['Hotel'])} ({_jt(r['Potensi'])}, {r['Kontribusi %']:.0f}%)".replace(".", ",")
                      for _, r in top3.iterrows()) +
            f". Mulai penegakan kontrak dari {n80} hotel ini.")


def _position_guide() -> str:
    """Penjelasan Posisi kota (P0–P100) dalam bahasa sehari-hari, mengikuti ambang STATUS_RULES."""
    r = STATUS_RULES
    return (
        "<b>Posisi kota (P0–P100)</b> = peringkat corp rate dibanding harga yang benar-benar dibayar untuk tipe "
        "kamar & meal plan yang sama <b>di kota yang sama</b>. Bayangkan rate per malam semua booking di kota itu "
        "diurutkan dari termurah ke termahal: <b>P30</b> artinya corp rate lebih mahal dari 30% booking dan lebih murah "
        "dari 70% sisanya. <b>Makin kecil angkanya, makin murah (makin bagus) corp rate-nya.</b><br>"
        f"&nbsp;&nbsp;– <b>P0–P25</b>: termasuk yang termurah di kota itu.<br>"
        f"&nbsp;&nbsp;– <b>P26–P{r['pos_tinjau']}</b>: setara harga pasar.<br>"
        f"&nbsp;&nbsp;– <b>P{r['pos_tinjau'] + 1}–P{r['pos_tidak']}</b>: mulai mahal (Perlu ditinjau).<br>"
        f"&nbsp;&nbsp;– <b>P{r['pos_tidak'] + 1}–P100</b>: lebih mahal dari kebanyakan transaksi di kota itu "
        "(Tidak kompetitif).<br>"
        f"&nbsp;&nbsp;– <b>\"-\"</b>: transaksi pembanding di kota itu kurang dari {MIN_MARKET_SAMPLE}, jadi belum bisa "
        "dinilai.")


NO_CORP = "Belum ada corp rate"
_NA_SENT = -9999.0   # penanda "tidak ada nilai" agar kolom tetap numerik (bisa diurutkan)
STATUS_ICON = {"Kompetitif": "● Kompetitif", "Perlu ditinjau": "● Perlu ditinjau",
               "Tidak kompetitif": "● Tidak kompetitif", NO_CORP: "○ Belum ada corp rate"}


def _detail_table(out):
    """Tabel detail per hotel & tipe kamar: angka tetap numerik (bisa diurutkan), bar untuk persentase,
    warna status konsisten (hijau / abu / merah)."""
    import streamlit as st
    show = pd.DataFrame({
        "Hotel": out["Hotel"],
        "Kota": out["Kota"],
        "Room type": out["Room type"].astype(str).str.replace(r"^Lainnya · ", "Lainnya: ", regex=True),
        "Corp rate": out["Corp"].round(0).fillna(0),          # 0 = tanpa corp rate (ditampilkan "–")
        "Rata-rata dibayar": out["Rata-rata dibayar"].round(0),
        "Selisih %": out["Selisih %"].round(1).fillna(_NA_SENT),
        "Room night": out["RN"],
        "Potensi (Rp jt)": (out["Potensi"] / 1e6).round(1),
        "Room Night < Corp %": out["Room Night < Corp %"].round(0),
        "Status": out["Daya saing"].map(STATUS_ICON).fillna(out["Daya saing"]),
    })
    st_col = {"Kompetitif": COMPETITIVE_GREEN, "Tidak kompetitif": NOT_COMPETITIVE_RED,
              "Perlu ditinjau": ui.MUTED}
    st_bg = {"Kompetitif": "rgba(59,122,11,0.08)", "Tidak kompetitif": "rgba(214,27,84,0.07)",
             "Perlu ditinjau": "rgba(107,107,107,0.07)"}
    status = out["Daya saing"]

    nocorp = status.eq(NO_CORP).to_numpy()

    def _style(col):
        if col.name == "Status" and nocorp.any():
            base = [f"color:{st_col.get(v, ui.INK)};background-color:{st_bg.get(v, 'transparent')};font-weight:600"
                    for v in status]
            return [f"color:{ui.MUTED};font-style:italic" if n else b_ for n, b_ in zip(nocorp, base)]
        if nocorp.any() and col.name not in ("Selisih %",):
            inner = _style_inner(col)
            return [f"color:{ui.MUTED};font-style:italic" if n else c_ for n, c_ in zip(nocorp, inner)]
        return _style_inner(col)

    def _style_inner(col):
        if col.name == "Status":
            return [f"color:{st_col.get(v, ui.INK)};background-color:{st_bg.get(v, 'transparent')};font-weight:600"
                    for v in status]
        if col.name == "Corp rate":
            return [f"color:{st_col[v]};font-weight:600" if v in ("Kompetitif", "Tidak kompetitif") else ""
                    for v in status]
        if col.name == "Selisih %":   # dibayar di atas corp = merah, di bawah = hijau
            return [f"color:{NOT_COMPETITIVE_RED}" if v > 0 else
                    (f"color:{COMPETITIVE_GREEN}" if _NA_SENT < v < 0 else f"color:{ui.MUTED}" if v == _NA_SENT else "")
                    for v in col]
        return [""] * len(col)

    rp = lambda v: "–" if pd.isna(v) or v == 0 else "Rp" + f"{v:,.0f}".replace(",", ".")
    sty = (show.style.apply(_style, axis=0)
           .format({"Corp rate": rp, "Rata-rata dibayar": rp,
                    "Selisih %": lambda v: "–" if pd.isna(v) or v == _NA_SENT else f"{v:+.1f}%".replace(".", ","),
                    "Room night": lambda v: f"{v:,.0f}".replace(",", ".")}, na_rep="-"))
    pmax = float(show["Potensi (Rp jt)"].max() or 1)
    st.dataframe(
        sty, hide_index=True, width="stretch", height=min(560, 35 * len(show) + 38),
        column_config={
            "Hotel": st.column_config.TextColumn("Hotel", width="medium"),
            "Kota": st.column_config.TextColumn("Kota"),
            "Room type": st.column_config.TextColumn("Tipe kamar"),
            "Corp rate": st.column_config.Column("Corp rate"),
            "Rata-rata dibayar": st.column_config.Column("Rata-rata dibayar"),
            "Selisih %": st.column_config.Column("Selisih", width="small",
                                                 help="Rata-rata dibayar dibanding corp rate. + = dibayar di atas corp rate."),
            "Room night": st.column_config.Column("RN", width="small", help="Room night"),
            "Potensi (Rp jt)": st.column_config.ProgressColumn(
                "Potensi", format="Rp%.0f jt", min_value=0, max_value=pmax, width="small",
                help="Potensi avoidance: penghematan bila semua room night dibayar sesuai corp rate."),
            "Room Night < Corp %": st.column_config.ProgressColumn(
                "RN < Corp", format="%.0f%%", min_value=0, max_value=100, width="small",
                help="Room Night < Corp %: porsi room night yang dibayar lebih murah dari corp rate. Makin kecil makin bagus."),
            "Status": st.column_config.TextColumn("Daya saing"),
        })
    st.caption("Semua baris adalah booking Breakfast Included. Klik judul kolom untuk mengurutkan."
               + (" Baris abu-abu (○ Belum ada corp rate) = tipe kamar yang dibeli tetapi tidak ada di kontrak: "
                  "hanya ditampilkan, tidak ikut dihitung di potensi, kepatuhan, maupun status." if nocorp.any() else ""))




def render_cost_avoidance(df_all: pd.DataFrame, *, can_download: bool, can_edit: bool,
                          username: str = "", excel_fn=None, standardize_name=None,
                          normalize_city=None, corp_source=None, data_sig=None):
    """corp_source: fungsi tanpa argumen -> dict sumber corp rate (mis. Google Drive) atau None.
    data_sig: sidik jari data (kunci cache); dihitung sendiri bila kosong."""
    import streamlit as st

    # ---------- sumber corp rate ----------
    cp = corp_path()
    src = corp_source() if corp_source else None
    drive_mode = bool(src and src.get("origin") == "drive")
    if src is None or src.get("df") is None:
        fallback = local_corp_source()
        if fallback is not None:
            fallback["error"] = (src or {}).get("error")
            src = fallback
    has_corp = bool(src and src.get("df") is not None)

    # ---------- judul + toolbar (pengaturan & catatan di popover agar halaman ringkas) ----------
    ui.page_title("Cost Avoidance",
                  "Rate dibayar (Invoice Amount ÷ room night) dibandingkan corporate rate per hotel, "
                  "tipe kamar dan meal plan")
    t1, t2, t3, _sp = st.columns([1.5, 1.1, 1.1, 3])   # toolbar di baris sendiri: tidak terjepit di layar sempit
    # basis pajak & carry-forward memakai nilai baku (tanda '++' di file tetap dibaca sebagai belum pajak)
    basis = st.session_state.get("ca_basis", "Nett (sudah pajak & service)")
    carry = st.session_state.get("ca_carry", True)
    years = corp_years(src) if has_corp else []
    YEAR_AUTO = "Sesuai tahun check-in"
    with t1:
        year_pick = st.selectbox("Tahun corp rate", [YEAR_AUTO] + [str(y) for y in years], key="ca_year",
                                 label_visibility="collapsed",
                                 format_func=lambda v: "Per tahun check-in" if v == YEAR_AUTO else f"Corp rate {v}",
                                 help="Pilih tahun corp rate pembanding. 'Sesuai tahun check-in' = booking dibandingkan "
                                      "dengan corp rate tahun check-in-nya.", disabled=not years)
    corp_year = None if year_pick == YEAR_AUTO else int(year_pick)

    if src and src.get("error"):
        ui.note("<b>Corp rate dari Google Drive belum bisa dibaca.</b> " + ui.esc(src["error"]) +
                (" Sementara memakai file unggahan di server." if has_corp else ""), warn=True)
    if not has_corp:
        if can_edit and not drive_mode:
            with st.container(border=True):
                ui.section("Unggah daftar corporate rate",
                           "Belum ada daftar corp rate. Atur folder Google Drive di Secrets ([corp_rate] folder_id) "
                           "agar tidak perlu unggah manual, atau unggah file di sini (hilang saat aplikasi di-reboot).")
                _corp_uploader(cp, "main")
                st.download_button("Unduh template", lambda: corp_template().getvalue(), file_name="template_corp_rate.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ca_tpl_main")
        elif not src or not src.get("error"):
            ui.note("<b>Belum ada daftar corporate rate.</b> Minta Super Admin atau Admin menyiapkannya.")
        return

    res = run_analysis(df_all, data_sig or frame_sig(df_all), src, "Belum" if basis.startswith("Belum") else "Nett",
                       carry, standardize_name, normalize_city, corp_year)
    issues = res["issues"]
    errors = [i for i in issues if i.startswith("Kolom wajib") or i.startswith("Kolom harga") or "kosong." in i]
    for msg in errors:
        st.error(msg)
    notes = [i for i in issues if i not in errors]
    with t2:
        with st.popover("Catatan data", icon=":material/info:", width="stretch"):
            where = "Google Drive" if src.get("origin") == "drive" else "server aplikasi (unggahan)"
            st.markdown("**Sumber daftar corp rate**")
            st.caption(f"{where} · {src.get('name', '')}" +
                       (f" · diperbarui {src['updated']:%d %b %Y %H:%M}" if src.get("updated") else ""))
            if src.get("refresh"):
                if st.button("Muat ulang dari Google Drive", icon=":material/sync:", key="ca_corp_refresh"):
                    src["refresh"]()
                    st.rerun()
                st.caption("Ganti file di folder Google Drive, lalu klik Muat ulang. Otomatis dicek ulang tiap 10 menit.")
            elif can_edit and cp.exists():
                _corp_uploader(cp, "pop")
            if notes:
                st.markdown("**Catatan pembacaan**")
                st.markdown("\n".join(f"- {n}" for n in notes))
            st.download_button("Template corp rate", lambda: corp_template().getvalue(),
                               file_name="template_corp_rate.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ca_tpl")
    if "name_changes" not in res:
        st.info("Data booking atau corp rate belum cukup untuk dianalisis.")
        return
    name_changes = res["name_changes"]
    with t3:
        with st.popover("Nama hotel", icon=":material/sell:", width="stretch", disabled=name_changes.empty):
            st.markdown(f"**Penyeragaman nama hotel** · {len(name_changes):,} nama master disesuaikan".replace(",", "."))
            st.caption("Nama di master corp rate diseragamkan dengan hotel_name_standardizer dan nama di invoice. "
                       f"'mirip' = disamakan otomatis (kemiripan ≥ {NAME_MATCH_MIN:.2f}, kota sama). "
                       "Perbaiki kamus standardizer bila ada yang keliru.")
            st.dataframe(name_changes, hide_index=True, width="stretch", height=min(360, 36 * len(name_changes) + 40))
            if can_download and not name_changes.empty:
                st.download_button("Unduh daftar penyeragaman", _excel(name_changes, "Penyeragaman nama", excel_fn),
                                   file_name=f"penyeragaman_nama_hotel_{datetime.now():%Y%m%d}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ca_dl_names")

    # ---------- satu kartu cakupan data ----------
    pct = lambda x, t: f"{x / t * 100:.0f}%" if t else "-"  # noqa: E731
    f = lambda v: f"{v:,.0f}".replace(",", ".")  # noqa: E731
    origin = "Google Drive" if src.get("origin") == "drive" else "unggahan server"
    ui.scope_bar([
        ("Corp rate pembanding", f"Tahun {corp_year}" if corp_year else "Per tahun check-in",
         f"{origin} · {f(res['n_hotel'])} hotel kontrak"),
        ("Hotel kontrak ada di invoice", f"{f(res['n_in'])} / {f(res['n_hotel'])}", pct(res["n_in"], res["n_hotel"])),
        ("Booking dihitung", f"{f(res['n_bf'])} / {f(res['n_all'])}", pct(res["n_bf"], res["n_all"])),
        ("Room night dihitung", f"{f(res['rn_bf'])} / {f(res['rn_all'])}", pct(res["rn_bf"], res["rn_all"])),
    ], "Hanya booking <b>termasuk sarapan</b>. Booking Room Only dan yang meal plan-nya tidak tertulis tidak dihitung.")
    if not res["ready"]:
        st.info("Belum ada booking termasuk sarapan yang bisa dibandingkan dengan corp rate.")
        return
    corp, m, pools, summary = res["corp"], res["m"], res["pools"], res["summary"]
    leak100, quality, city10 = res["leak100"], res["quality"], res["city10"]

    sub1, sub2 = st.tabs(["Ringkasan", "Analisis lanjutan"])

    # ================= RINGKASAN =================
    with sub1:
        topn = st.columns([1, 4])[0].selectbox("Hotel teratas (room night)", [25, 50, 100], index=2, key="ca_topn")
        s = top_hotels(summary, topn)
        if s.empty:
            st.info("Belum ada booking yang cocok dengan daftar corp rate.")
        else:
            _cumulative_view(s, topn, can_download, excel_fn)

            h1, h3, h2 = st.columns([2.2, 1.6, 1], vertical_alignment="bottom")
            h1.markdown("**Detail per hotel & tipe kamar**")
            show_nc = h3.toggle("Tampilkan tipe kamar yang belum ada corp rate", value=True, key="ca_show_nocorp",
                                help="Tipe kamar yang dibeli di hotel ber-kontrak tetapi tidak punya corp rate. "
                                     "Hanya ditampilkan, tidak ikut dihitung.")
            sort_by = h2.selectbox("Urutkan", ["Nama hotel (A–Z)", "Room night", "Potensi avoidance", "Selisih %"],
                                   key="ca_sort")
            u = res.get("uncovered")
            if show_nc and u is not None and not u.empty:
                hk = ["_hotel_key", "_city_key"]
                hrn = s.groupby(hk)["_hotel_rn"].first()
                ux = u.join(hrn, on=hk, how="inner")
                if not ux.empty:
                    ux = pd.DataFrame({"_hotel_key": ux["_hotel_key"], "_city_key": ux["_city_key"],
                                       "Hotel": ux["Hotel"], "Kota": ux["Kota"], "Room type": ux["Tipe kamar"],
                                       "Meal plan": ANALYSIS_MEAL, "Corp": np.nan, "Rata-rata dibayar": ux["Rata-rata dibayar"],
                                       "Selisih %": np.nan, "RN": ux["RN"], "Potensi": np.nan,
                                       "Room Night < Corp %": np.nan, "Posisi kota": np.nan,
                                       "Daya saing": NO_CORP, "_hotel_rn": ux["_hotel_rn"]})
                    s = pd.concat([s, ux], ignore_index=True)
            col = {"Room night": "RN", "Potensi avoidance": "Potensi", "Selisih %": "Selisih %"}.get(sort_by)
            if sort_by == "Nama hotel (A–Z)":   # abjad hotel, lalu kota, lalu tipe kamar dengan room night terbesar
                d = s.assign(_h=s["Hotel"].astype(str).str.casefold(), _k=s["Kota"].astype(str).str.casefold()) \
                     .sort_values(["_h", "_k", "RN"], ascending=[True, True, False])
            elif sort_by == "Room night":
                d = s.sort_values(["_hotel_rn", "RN"], ascending=False)
            else:
                d = s.sort_values(col, ascending=False)
            out = d[["Hotel", "Kota", "Room type", "Meal plan", "Corp", "Rata-rata dibayar", "Selisih %", "RN",
                     "Potensi", "Room Night < Corp %", "Posisi kota", "Daya saing"]].reset_index(drop=True)
            _detail_table(out)
            R = STATUS_RULES
            with st.expander("Arti status, Room Night < Corp % dan Posisi kota", icon=":material/help:"):
                st.markdown('<div class="mx-guide">'
                f"• <span style='color:{COMPETITIVE_GREEN};font-weight:600'>Kompetitif (hijau)</span>: Room Night &lt; Corp % "
                f"di bawah {R['below_tinjau']}% dan posisi paling tinggi P{R['pos_tinjau']}.<br>"
                f"• <span style='color:{NOT_COMPETITIVE_RED};font-weight:600'>Tidak kompetitif (merah)</span>: Room Night "
                f"&lt; Corp % di atas {R['below_tidak']}%, atau posisi di atas P{R['pos_tidak']}.<br>"
                "• <b>Perlu ditinjau (abu)</b>: di antara keduanya.<br>"
                f"• <span style='color:{ui.MUTED};font-style:italic'>○ Belum ada corp rate</span>: tipe kamar dibeli di hotel "
                "ber-kontrak tetapi tidak ada di kontrak. Ditampilkan sebagai informasi, <b>tidak dihitung</b> di potensi, "
                "kepatuhan, grafik, maupun status. Layak diusulkan masuk kontrak bila room night-nya besar.<br>"
                "<b>Room Night &lt; Corp %</b> = persentase room night yang dibayar lebih murah dari corp rate. "
                "Rate per malam tiap booking = Invoice Amount ÷ Number of Rooms Night; bila lebih rendah dari corp rate, "
                "semua room night booking itu dihitung. Contoh: 3 dari 5 room night dibayar di bawah corp rate = 60%. "
                "Makin kecil angkanya, makin bagus corp rate-nya.<br>"
                + _position_guide() + "</div>", unsafe_allow_html=True)
            if can_download:
                st.download_button("Unduh Excel", _excel(out.drop(columns=["Posisi kota"]).rename(columns={"Corp": "Corp rate", "RN": "Room night"}),
                                   "Cost Avoidance", excel_fn,
                                   money_cols=("Corp rate", "Rata-rata dibayar", "Potensi"),
                                   int_cols=("Room night",), dec_cols=("Selisih %", "Room Night < Corp %")),
                                   file_name=f"cost_avoidance_{datetime.now():%Y%m%d}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ca_dl_detail")

        nc = (m[~m["_contract_hotel"]].groupby(["_hotel_key", "_city_key"])
              .agg(Hotel=("Hotel Name", "first"), Kota=("City", "first"), RN=("Number of Rooms Night", "sum"),
                   Inv=("Invoice Amount", "sum")).reset_index().sort_values("RN", ascending=False).head(10))
        with st.expander("Kandidat corp rate baru — hotel volume besar tanpa corp rate"):
            if nc.empty:
                st.caption("Tidak ada.")
            else:
                nc["Rata-rata rate"] = (nc["Inv"] / nc["RN"]).map(_rp)
                nc["RN"] = nc["RN"].map(lambda v: f"{v:,.0f}".replace(",", "."))
                st.dataframe(nc[["Hotel", "Kota", "RN", "Rata-rata rate"]].rename(columns={"RN": "Room night"}),
                             hide_index=True, width="stretch")
        with st.expander("Cara perhitungan", icon=":material/menu_book:"):
            st.markdown(
                "- **Pencocokan**: hotel + kota + tipe kamar + meal plan, dan tanggal check-in di dalam masa berlaku.\n"
                "- **Potensi avoidance** = Σ (rate dibayar per booking − corp rate) × room night, hanya booking di atas corp rate.\n"
                f"- **Room Night < Corp %**: porsi room night yang lebih murah dari corp rate. **Posisi kota**: persentil corp rate "
                f"terhadap rate dibayar tipe kamar & meal plan sama di kota yang sama (minimal {MIN_MARKET_SAMPLE} booking).\n"
                f"- **Status**: Tidak kompetitif bila Room Night < Corp % > {STATUS_RULES['below_tidak']}% atau posisi > "
                f"P{STATUS_RULES['pos_tidak']}; Perlu ditinjau bila ≥ {STATUS_RULES['below_tinjau']}% atau > "
                f"P{STATUS_RULES['pos_tinjau']}.")

    # ================= ANALISIS LANJUTAN =================
    with sub2:
        lk = leak100
        manual_df = load_manual()
        manual = dict(zip(manual_df["key"].astype(str), pd.to_numeric(manual_df["persen"], errors="coerce")))
        pct = st.session_state.get("ca_pct", 40)
        rg = renegotiation(summary, pools, pct, manual, 100)
        q, suspects = quality
        matched_pct = float(q.loc[q["Status"] == "Cocok", "Porsi room night %"].sum()) if not q.empty else 0.0

        ui.kpi_row([
            ui.kpi("Cost avoidance (penegakan kontrak)", _jt(summary["Potensi"].sum() if not summary.empty else 0),
                   "booking di atas corp rate", dark=True),
            ui.kpi("Kebocoran ke hotel non-kontrak", _jt(lk["Potensi"].sum() if not lk.empty else 0), "100 hotel teratas"),
            ui.kpi("Hard savings (renegosiasi)", _jt(rg["Hard savings"].sum() if not rg.empty else 0)),
            ui.kpi("Room night cocok ke corp rate", _pct(matched_pct))])

        # 1. Kebocoran
        ui.section("1 · Kebocoran ke hotel non-kontrak")
        st.caption("100 hotel teratas tanpa corp rate (room night), di kota yang sudah punya hotel ber-corp rate.")
        if lk.empty:
            st.info("Tidak ada booking non-kontrak di kota ber-corp rate.")
        else:
            out = lk[["Hotel", "Kota", "Room type", "Meal plan", "RN", "Rata-rata dibayar", "Setara", "Selisih %", "Potensi"]]
            show = out.copy()
            show["RN"] = out["RN"].map(lambda v: f"{v:,.0f}".replace(",", "."))
            for c_ in ("Rata-rata dibayar", "Setara"):
                show[c_] = out[c_].map(_rp)
            show["Selisih %"] = out["Selisih %"].map(lambda v: _pct(v, True))
            show["Potensi"] = out["Potensi"].map(_jt)
            show.index = show.index + 1
            st.dataframe(show.rename(columns={"RN": "Room night", "Room type": "Tipe kamar", "Meal plan": "Meal plan", "Setara": "Corp rate setara (median kota)",
                                              "Potensi": "Potensi pengalihan"}),
                         width="stretch", height=460)
            st.warning("Angka ini **potensi**: data tidak mencatat apakah hotel kontrak sedang penuh atau lokasinya jauh "
                       "dari tempat tugas, dan kelas hotel belum ada di data.")
            if can_download:
                st.download_button("Unduh Excel", _excel(out.rename(columns={"RN": "Room night", "Setara": "Corp rate setara"}),
                                   "Kebocoran non-kontrak", excel_fn,
                                   money_cols=("Rata-rata dibayar", "Corp rate setara", "Potensi"), int_cols=("Room night",),
                                   dec_cols=("Selisih %",)),
                                   file_name=f"kebocoran_non_kontrak_{datetime.now():%Y%m%d}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ca_dl_leak")

        # 2. Renegosiasi
        ui.section("2 · Target renegosiasi corp rate")
        pct = st.slider("Target persentil (P20 agresif · P60 lunak)", 20, 60, pct, 5, key="ca_pct")
        if rg.empty:
            st.info("Tidak ada corp rate berstatus Tidak kompetitif / Perlu ditinjau.")
        else:
            ed = rg[["_key", "Hotel", "Kota", "Room type", "Meal plan", "Daya saing", "Corp", "Target dipakai",
                     "Sumber target", "Turun %", "RN", "Hard savings", "Target turun manual (%)"]].copy()
            prev = manual_df.set_index("key")["alasan"].to_dict() if not manual_df.empty else {}
            ed["Alasan"] = ed["_key"].map(lambda k: prev.get(k, ""))
            ed.index = ed.index + 1
            disabled = [c_ for c_ in ed.columns if c_ not in ("Target turun manual (%)", "Alasan")] if can_edit else True
            edited = st.data_editor(
                ed, key="ca_editor", height=500, width="stretch", disabled=disabled,
                column_order=[c_ for c_ in ed.columns if c_ != "_key"],
                column_config={
                    "Corp": st.column_config.NumberColumn("Corp rate", format="Rp %d"),
                    "Target dipakai": st.column_config.NumberColumn(format="Rp %d"),
                    "Turun %": st.column_config.NumberColumn(format="%.1f%%"),
                    "RN": st.column_config.NumberColumn("Room night", format="%d"),
                    "Hard savings": st.column_config.NumberColumn(format="Rp %d"),
                    "Target turun manual (%)": st.column_config.NumberColumn(
                        min_value=0.0, max_value=99.0, step=0.5, format="%.1f",
                        help="Kosongkan untuk memakai target persentil"),
                    "Alasan": st.column_config.TextColumn(max_chars=200),
                })
            st.metric("Total hard savings", _jt(rg["Hard savings"].sum()),
                      f"{int(rg['Sumber target'].eq('manual').sum())} target manual", delta_color="off")
            if can_edit and st.button("Simpan target manual", key="ca_save_manual", icon=":material/save:"):
                keep = edited[edited["Target turun manual (%)"].notna() & (edited["Target turun manual (%)"] > 0)]
                others = manual_df[~manual_df["key"].isin(ed["_key"])] if not manual_df.empty else manual_df
                new = pd.DataFrame({"key": keep["_key"], "persen": keep["Target turun manual (%)"],
                                    "alasan": keep["Alasan"].fillna(""), "oleh": username,
                                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M")})
                save_manual(pd.concat([others, new], ignore_index=True))
                st.success(f"{len(new)} target manual disimpan.")
                st.rerun()
            if not can_edit:
                st.caption("Hanya Super Admin & Admin yang bisa mengubah target manual.")
            if can_download:
                xl = rg.drop(columns=["_key", "_hotel_key", "_city_key", "_hotel_rn"], errors="ignore")
                xl = xl[["Hotel", "Kota", "Room type", "Meal plan", "Daya saing", "Corp", "Target persentil",
                         "Target turun manual (%)", "Target dipakai", "Sumber target", "Turun %", "RN", "Hard savings"]]
                st.download_button("Unduh Excel", _excel(xl.rename(columns={"Corp": "Corp rate", "RN": "Room night"}),
                                   "Target renegosiasi", excel_fn,
                                   money_cols=("Corp rate", "Target persentil", "Target dipakai", "Hard savings"),
                                   int_cols=("Room night",), dec_cols=("Turun %", "Target turun manual (%)")),
                                   file_name=f"target_renegosiasi_P{pct}_{datetime.now():%Y%m%d}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="ca_dl_reneg")

        # 3. Kepatuhan per kota
        ui.section("3 · Kepatuhan per kota")
        cc = city10
        if cc.empty:
            st.info("Belum ada data.")
        else:
            cc.index = cc.index + 1
            st.dataframe(cc[["Kota", "RN", "Kepatuhan %", "Non-kontrak %"]], width="stretch",
                         column_config={
                             "RN": st.column_config.NumberColumn("Room night", format="%d"),
                             "Kepatuhan %": st.column_config.ProgressColumn("Room night ≤ corp rate", format="%.0f%%",
                                                                           min_value=0, max_value=100),
                             "Non-kontrak %": st.column_config.NumberColumn(format="%.0f%%")})
            st.caption("10 kota teratas berdasarkan room night, di kota yang punya hotel ber-corp rate.")

        # 4. Kualitas pencocokan
        ui.section("4 · Kualitas pencocokan data")
        if not q.empty:
            q = q.sort_values("Porsi room night %", ascending=False)
            st.dataframe(q, hide_index=True, width="stretch",
                         column_config={"Porsi room night %": st.column_config.ProgressColumn(
                             format="%.1f%%", min_value=0, max_value=100)})
        fx = st.session_state.get("_fx_info") or {}
        if fx.get("converted"):
            st.caption(f"{fx['converted']:,} booking non-Rupiah dikonversi dengan kurs.".replace(",", "."))
        if not suspects.empty:
            with st.expander(f"Nama hotel yang mungkin tidak cocok ({len(suspects)})"):
                st.caption("Hotel di invoice yang dianggap non-kontrak tetapi namanya mirip hotel kontrak di kota yang sama. "
                           "Periksa, lalu samakan penulisannya di daftar corp rate atau kamus nama hotel.")
                st.dataframe(suspects, hide_index=True, width="stretch")
