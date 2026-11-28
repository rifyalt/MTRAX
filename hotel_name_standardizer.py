"""
hotel_name_standardizer.py
==========================
Penyeragaman otomatis penulisan nama hotel untuk kolom "Hotel Name" dan
"Supplier Name".

Dua lapisan aturan:

1. ATURAN MEREK (BRAND_RULES) — selalu diterapkan ke setiap nama, termasuk
   nama yang tidak ada di file standar:
       Melia                          -> Meliá
       Movenpick / Moevenpick         -> Mövenpick
       El Hotel                       -> éL Hotel
       Marriot (salah ketik)          -> Marriott
   ditambah penggantian nama penuh (FULL_NAME_OVERRIDES), mis.
       Courtyard Bandung Dago         -> Courtyard by Marriott Bandung Dago

2. KAMUS STANDAR dari file Excel (data/unique_hotels.xlsx, kolom
   "HOTEL NAME" -> "STADARD NAME"). Pencocokan tidak peka huruf besar/kecil,
   aksen, tanda baca, spasi ganda, "&" vs "and", dan "&amp;".
   Jika satu nama punya beberapa standar berbeda di Excel, dipilih standar
   yang paling sering muncul; jika seri, yang paling mirip dengan nama
   aslinya. Penulisan standar yang hanya beda kapitalisasi/aksen juga
   diseragamkan (dipilih yang paling sering, utamakan yang beraksen).

Fungsi aman dipanggil berulang (idempoten): hasil standar jika
distandarkan lagi tidak berubah.

Pemakaian di app:
    from hotel_name_standardizer import standardize_hotel_columns
    df = standardize_hotel_columns(df)

Pemakaian dari command line (menyeragamkan file Excel transaksi):
    python hotel_name_standardizer.py input.xlsx -o output.xlsx
    python hotel_name_standardizer.py --report laporan_konflik.xlsx
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DEFAULT_STANDARD_FILE = Path(__file__).resolve().parent / "data" / "unique_hotels.xlsx"
STANDARD_FILE = Path(os.environ.get("HOTEL_STANDARD_FILE", DEFAULT_STANDARD_FILE))

SOURCE_COL = "HOTEL NAME"
TARGET_COL = "STADARD NAME"      # sesuai ejaan header di file Excel

# Kolom data yang diseragamkan.
#   brand  = aturan merek (Meliá, Mövenpick, éL Hotel, Marriott)
#   lookup = cocokkan ke kamus standar Excel
COLUMNS_TO_STANDARDIZE = {
    "Hotel Name":    {"brand": True, "lookup": True},
    "Supplier Name": {"brand": True, "lookup": True},
}

# (pola regex, pengganti). Pola tidak peka huruf besar/kecil.
BRAND_RULES = [
    (r"(?<!\w)meli[aáà](?!\w)",            "Meliá"),
    (r"(?<!\w)m(?:ö|o|oe)venpick(?!\w)",   "Mövenpick"),
    (r"(?<!\w)[eéÉ]l\s+hotel(?!\w)",       "éL Hotel"),
    (r"(?<!\w)marriot(?!\w)",              "Marriott"),
]

# Penggantian nama penuh (dicocokkan tanpa peka huruf/aksen/tanda baca).
# Kiri: nama yang ditemui; kanan: nama standar yang dipakai.
FULL_NAME_OVERRIDES = {
    "Courtyard Bandung Dago":              "Courtyard by Marriott Bandung Dago",
    "Courtyard by Marriot Bandung Dago":   "Courtyard by Marriott Bandung Dago",
    "Courtyard by Marriott Bandung Dago":  "Courtyard by Marriott Bandung Dago",
}

_BRAND_RES = [(re.compile(p, re.IGNORECASE), r) for p, r in BRAND_RULES]

# Gabungkan otomatis nama standar yang hanya berbeda kata-kata ini (atau urutannya),
# mis. "Hotel Aston Manado" = "ASTON Manado Hotel", "Akmani Legian" = "The Akmani Legian".
MERGE_STANDARD_VARIANTS = True
FOLLOW_CHAINS = False          # True = ikuti semua rantai A->B->C di Excel (perilaku lama)
SAME_HOTEL_RATIO = 0.85        # rantai hanya diikuti bila nama B & C semirip ini (beda ejaan saja)
MERGE_IGNORE_WORDS = {"hotel", "the"}
STRICT_PREFIX = "~"            # penanda kunci cadangan (tidak mungkin muncul di name_key)
MERGE_REPORT = [pd.DataFrame()]


# ---------------------------------------------------------------------------
# UTILITAS
# ---------------------------------------------------------------------------
def _clean_spaces(s: str) -> str:
    s = str(s).replace(" ", " ").replace("&amp;", "&").replace("&AMP;", "&")
    return re.sub(r"\s+", " ", s).strip()


def apply_brand_rules(name: str) -> str:
    """Terapkan aturan merek (Meliá, Mövenpick, éL Hotel, Marriott)."""
    s = _clean_spaces(name)
    for rx, repl in _BRAND_RES:
        s = rx.sub(repl, s)
    return s


def name_key(name) -> str:
    """Kunci pencocokan: huruf kecil, tanpa aksen, tanpa tanda baca, '&' = 'and'."""
    s = _clean_spaces(name)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9@]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strict_key(name) -> str:
    """name_key tanpa kata 'hotel'/'the' dan tanpa memperhatikan posisi kata itu."""
    return " ".join(w for w in name_key(name).split() if w not in MERGE_IGNORE_WORDS)


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in ("", "nan", "none", "null")


def _spelling_score(s: str):
    """Urutan prioritas memilih ejaan: beraksen > bukan HURUF BESAR SEMUA."""
    has_accent = any(ord(c) > 127 for c in s)
    not_all_caps = not (s.upper() == s and any(c.isalpha() for c in s))
    return (has_accent, not_all_caps)


# ---------------------------------------------------------------------------
# MEMBANGUN KAMUS
# ---------------------------------------------------------------------------
def _read_standard_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]
    src, tgt = SOURCE_COL.upper(), TARGET_COL.upper()
    if tgt not in df.columns and "STANDARD NAME" in df.columns:
        tgt = "STANDARD NAME"
    if src not in df.columns or tgt not in df.columns:
        raise ValueError(f"File standar harus punya kolom '{SOURCE_COL}' dan '{TARGET_COL}'. "
                         f"Kolom yang ada: {list(df.columns)}")
    out = df[[src, tgt]].rename(columns={src: "raw", tgt: "std"})
    out = out[~out["raw"].map(_is_missing) & ~out["std"].map(_is_missing)]
    out["raw"] = out["raw"].map(apply_brand_rules)
    out["std"] = out["std"].map(apply_brand_rules)
    return out.reset_index(drop=True)


def build_mapping(table: pd.DataFrame):
    """
    Bangun kamus {name_key -> nama standar} dari tabel (raw, std).
    Mengembalikan (mapping, conflicts_df).
    """
    # 1) Ejaan kanonik untuk setiap nama standar (seragamkan kapitalisasi/aksen)
    std_variants = defaultdict(Counter)
    for s in table["std"]:
        std_variants[name_key(s)][s] += 1
    canon_spelling = {}
    for k, cnt in std_variants.items():
        canon_spelling[k] = max(cnt.items(),
                                key=lambda kv: (kv[1], _spelling_score(kv[0]), kv[0]))[0]

    # 2) Untuk setiap nama (kunci), pilih standar lewat voting
    votes = defaultdict(Counter)
    for raw, std in zip(table["raw"], table["std"]):
        votes[name_key(raw)][name_key(std)] += 1

    mapping, conflict_rows = {}, []
    for k, cnt in votes.items():
        best = max(cnt.items(), key=lambda kv: (
            kv[1], difflib.SequenceMatcher(None, k, kv[0]).ratio(), kv[0]))[0]
        mapping[k] = best
        if len(cnt) > 1:
            conflict_rows.append({
                "Nama (kunci)": k,
                "Pilihan di Excel": " | ".join(f"{canon_spelling[c]} ({n}x)" for c, n in cnt.most_common()),
                "Dipilih": canon_spelling[best],
            })

    # 3) Nama standar sendiri juga menjadi kunci (agar idempoten)
    for k in canon_spelling:
        mapping.setdefault(k, k)

    # 4) Rantai A -> B -> C di Excel hanya diikuti bila B dan C memang hotel yang sama
    #    (beda ejaan saja, mis. "De_Rain Hotel Bandung" -> "DeRain Hotel Bandung").
    #    Rantai ke hotel lain (mis. "The Alana ... Solo by ASTON" -> "... Malioboro Yogyakarta",
    #    "Arthama Hotel Wahid-Hasyim Jakarta" -> "Arthama Hotel Makassar") TIDAK diikuti:
    #    nama yang sudah menjadi nama standar di baris lain tetap dirinya sendiri.
    def _same(a, b):
        sa, sb = strict_key(a), strict_key(b)
        ta, tb = set(sa.split()), set(sb.split())
        if min(len(ta), len(tb)) >= 2 and (ta <= tb or tb <= ta):   # hanya menambah kata (mis. nama kota)
            return True
        return difflib.SequenceMatcher(None, sa, sb).ratio() >= SAME_HOTEL_RATIO
    resolved = {}
    for k in mapping:
        seen, cur = [k], mapping[k]
        while (cur in mapping and mapping[cur] != cur and cur not in seen
               and (FOLLOW_CHAINS or _same(cur, mapping[cur]))):
            seen.append(cur)
            cur = mapping[cur]
        resolved[k] = cur
    # Idempoten: setiap nama standar dipetakan ke dirinya sendiri. Bila nama standar itu
    # di Excel diarahkan ke ejaan lain dari hotel yang SAMA, semua yang menuju ke sana ikut
    # dipindah ke ejaan tersebut; bila ke hotel LAIN, nama standar tetap dirinya sendiri.
    for _ in range(3):
        moved = False
        for v in set(resolved.values()):
            w = resolved.get(v, v)
            if w == v:
                continue
            moved = True
            if _same(v, w):
                for k in [k for k, c in resolved.items() if c == v]:
                    resolved[k] = w
            else:
                resolved[v] = v
        if not moved:
            break

    final = {k: canon_spelling.get(v, v) for k, v in resolved.items()}

    # 4b) Gabungkan nama standar yang hanya beda kata "Hotel"/"The" atau urutannya
    #     (mis. "Hotel Aston Manado" = "ASTON Manado Hotel"). Wakil kelompok: yang paling
    #     banyak dipakai, lalu ejaan terbaik, lalu nama terpanjang.
    if MERGE_STANDARD_VARIANTS:
        usage = Counter(final.values())
        groups = defaultdict(set)
        for v in usage:
            sk = strict_key(v)
            if sk:
                groups[sk].add(v)
        rep_of, merge_rows = {}, []
        for sk, names in groups.items():
            rep = max(names, key=lambda n: (usage[n], _spelling_score(n), len(n), n))
            for n in names:
                rep_of[n] = rep
            if len(names) > 1:
                merge_rows.append({"Digabung menjadi": rep,
                                   "Variasi nama standar": " | ".join(sorted(names - {rep}))})
        final = {k: rep_of.get(v, v) for k, v in final.items()}
        for sk, names in groups.items():                  # cadangan untuk nama di luar Excel
            final[STRICT_PREFIX + sk] = rep_of[next(iter(names))]
        MERGE_REPORT[0] = pd.DataFrame(merge_rows)

    # 5) Override nama penuh (selalu menang)
    for src, dst in FULL_NAME_OVERRIDES.items():
        dst = apply_brand_rules(dst)
        final[name_key(src)] = dst
        final[name_key(dst)] = dst
    ov_keys = {name_key(src): apply_brand_rules(dst) for src, dst in FULL_NAME_OVERRIDES.items()}
    for k, v in list(final.items()):          # arahkan ulang HANYA yang menuju nama yang di-override
        ov = ov_keys.get(name_key(v))
        if ov is not None and ov != v:
            final[k] = ov

    return final, pd.DataFrame(conflict_rows)


@lru_cache(maxsize=4)
def _load_mapping_cached(path_str: str, mtime: float):
    table = _read_standard_table(Path(path_str))
    return build_mapping(table)


def load_mapping(path: str | Path | None = None):
    """Muat kamus standar (di-cache, otomatis dimuat ulang jika file berubah)."""
    p = Path(path) if path else STANDARD_FILE
    if not p.exists():
        return {}, pd.DataFrame()
    return _load_mapping_cached(str(p), p.stat().st_mtime)


# ---------------------------------------------------------------------------
# API UTAMA
# ---------------------------------------------------------------------------
def standardize_name(name, mapping: dict | None = None, use_lookup: bool = True):
    """Seragamkan satu nama hotel/supplier."""
    if _is_missing(name):
        return name
    if mapping is None:
        mapping = load_mapping()[0]
    s = apply_brand_rules(name)
    k = name_key(s)
    if use_lookup and k in mapping:
        return mapping[k]
    if use_lookup and MERGE_STANDARD_VARIANTS:
        sk = STRICT_PREFIX + strict_key(s)
        if sk in mapping:
            return mapping[sk]
    ov = {name_key(a): apply_brand_rules(b) for a, b in FULL_NAME_OVERRIDES.items()}
    return ov.get(k, s)


def standardize_series(series: pd.Series, mapping: dict | None = None,
                       use_lookup: bool = True) -> pd.Series:
    """Seragamkan satu kolom (diproses per nilai unik, jadi cepat)."""
    if mapping is None:
        mapping = load_mapping()[0]
    uniques = pd.unique(series.dropna())
    lut = {u: standardize_name(u, mapping, use_lookup) for u in uniques}
    return series.map(lambda v: lut.get(v, v) if not _is_missing(v) else v)


def standardize_hotel_columns(df: pd.DataFrame, columns: dict | None = None,
                              standard_file: str | Path | None = None,
                              keep_original: bool = False) -> pd.DataFrame:
    """
    Seragamkan kolom "Hotel Name" dan "Supplier Name" pada DataFrame.
    keep_original=True menyimpan nilai lama di kolom "<nama kolom> (Raw)".
    """
    if df is None or df.empty:
        return df
    mapping, _ = load_mapping(standard_file)
    cols = columns or COLUMNS_TO_STANDARDIZE
    out = df.copy()
    for col, opt in cols.items():
        if col not in out.columns:
            continue
        if keep_original and f"{col} (Raw)" not in out.columns:
            out[f"{col} (Raw)"] = out[col]
        if opt.get("brand", True) or opt.get("lookup", True):
            out[col] = standardize_series(out[col], mapping, use_lookup=opt.get("lookup", True))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Seragamkan nama Hotel Name / Supplier Name.")
    ap.add_argument("input", nargs="?", help="File Excel/CSV transaksi yang akan diseragamkan")
    ap.add_argument("-o", "--output", help="File hasil (default: <input>_standard.xlsx)")
    ap.add_argument("--standard-file", help="File kamus standar (default: data/unique_hotels.xlsx)")
    ap.add_argument("--report", help="Simpan laporan konflik kamus ke file Excel ini")
    a = ap.parse_args()

    mapping, conflicts = load_mapping(a.standard_file)
    print(f"Kamus standar: {sum(1 for k in mapping if not k.startswith(STRICT_PREFIX)):,} nama, {len(MERGE_REPORT[0]):,} kelompok nama mirip digabung, {len(conflicts):,} konflik di Excel diselesaikan otomatis.")
    if a.report:
        with pd.ExcelWriter(a.report) as w:
            conflicts.to_excel(w, index=False, sheet_name="Konflik di Excel")
            MERGE_REPORT[0].to_excel(w, index=False, sheet_name="Digabung otomatis")
        print(f"Laporan konflik: {a.report}")
    if a.input:
        src = Path(a.input)
        df = pd.read_csv(src) if src.suffix.lower() == ".csv" else pd.read_excel(src)
        res = standardize_hotel_columns(df, standard_file=a.standard_file)
        dst = Path(a.output) if a.output else src.with_name(src.stem + "_standard.xlsx")
        res.to_excel(dst, index=False)
        for c in COLUMNS_TO_STANDARDIZE:
            if c in df.columns:
                print(f"{c}: {df[c].nunique():,} -> {res[c].nunique():,} nama unik")
        print(f"Hasil: {dst}")


if __name__ == "__main__":
    _main()
