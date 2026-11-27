"""
mask_data.py — Anonimisasi data transaksi hotel mtrax sebelum dipakai dashboard.

Apa yang dilakukan:
  1. Hanya menyimpan 13 kolom yang dibutuhkan dashboard (whitelist).
     Kolom lain (nama, email, Employee Id, dsb.) dibuang dan tidak ditulis ke file hasil.
  2. Travel Request Number dan Booking ID diganti kode acak.
     - Setiap kali skrip dijalankan dibuat KUNCI ACAK BARU yang hanya ada di memori
       dan tidak pernah disimpan, jadi kode tidak bisa dibalik ke nomor asli.
     - Dalam satu kali proses, nilai yang sama selalu jadi kode yang sama
       (hitungan booking & travel request tetap benar).
     - Proses berikutnya menghasilkan kode yang berbeda.
  3. Memeriksa sisa kolom teks: memberi peringatan bila ada isi yang mirip
     email, nomor HP, atau NIK (16 digit). Nilainya tidak ditampilkan.

Cara pakai (dijalankan oleh pemilik data, di komputer/server perusahaan):
    pip install pandas openpyxl
    python mask_data.py --input "folder_atau_file.xlsx" --output "mtrax_masked.xlsx"

    --input bisa satu file (.xlsx / .xls / .csv) atau folder berisi banyak file.
    Semua file dalam satu folder diproses dengan kunci yang sama, sehingga satu
    Travel Request yang muncul di beberapa file tetap mendapat kode yang sama.

Bisa juga dipakai dari dashboard (lihat fungsi mask_dataframe di bawah).
"""

import argparse
import hashlib
import hmac
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
KEEP_COLUMNS = [
    "Travel Request Number",
    "Booking ID",
    "Issue Time",
    "Room Category",
    "Check in Date",
    "Meal Plan",
    "Check out Date",
    "Hotel Name",
    "City",
    "Country",
    "Number of Rooms Night",
    "Currency",
    "Invoice Amount",
]

# nama lain kolom di file sumber (dibandingkan tanpa beda huruf/spasi); nama yang DIAWALI teks ini juga diterima
COLUMN_ALIASES = {
    "Meal Plan": ["meal plan (room only / breakfast included)", "meal plan", "meal"],
    "Room Category": ["room category", "room type"],
}

# kolom -> awalan kode
MASK_COLUMNS = {
    "Travel Request Number": "TR",
    "Booking ID": "BK",
}

CODE_LENGTH = 16  # jumlah karakter heksadesimal pada kode (64 bit)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+62|\b0)8\d{7,11}\b")
_NIK = re.compile(r"\b\d{16}\b")


# ---------------------------------------------------------------------------
# FUNGSI INTI
# ---------------------------------------------------------------------------
def _norm_col(name) -> str:
    """Nama kolom dibandingkan tanpa beda huruf besar/kecil & spasi ganda."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def _norm_value(v) -> str:
    """Samakan bentuk nilai sebelum di-hash: '12345.0' == '12345' == ' 12345 '."""
    s = str(v).strip()
    return re.sub(r"\.0+$", "", s)


def select_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Ambil hanya KEEP_COLUMNS (toleran beda huruf/spasi). Return (df, dibuang, hilang)."""
    lookup = {_norm_col(c): c for c in df.columns}
    keep, missing = {}, []
    for target in KEEP_COLUMNS:
        src = lookup.get(_norm_col(target))
        for alias in COLUMN_ALIASES.get(target, []):
            if src is not None:
                break
            src = lookup.get(alias) or next((orig for n, orig in lookup.items()
                                             if n.startswith(alias) and orig not in keep), None)
        if src is None:
            missing.append(target)
        else:
            keep[src] = target
    dropped = [c for c in df.columns if c not in keep]
    out = df[list(keep)].rename(columns=keep)
    return out, dropped, missing


def mask_dataframe(df: pd.DataFrame, key: bytes | None = None) -> pd.DataFrame:
    """
    Pilih 13 kolom lalu ganti kolom ID dengan kode acak.
    key=None  -> kunci acak baru dibuat untuk pemanggilan ini lalu dibuang.
    Panggil SEKALI untuk seluruh data gabungan (jangan per file) agar kode konsisten.
    """
    out, _, _ = select_columns(df)
    out = out.copy()
    run_key = key if key is not None else secrets.token_bytes(32)
    try:
        for col, prefix in MASK_COLUMNS.items():
            if col not in out.columns:
                continue
            cache = {}

            def _code(v, _prefix=prefix, _col=col):
                if pd.isna(v) or str(v).strip() == "":
                    return pd.NA
                n = _norm_value(v)
                if n not in cache:
                    mac = hmac.new(run_key, f"{_col}|{n}".encode(), hashlib.sha256)
                    cache[n] = f"{_prefix}-{mac.hexdigest()[:CODE_LENGTH].upper()}"
                return cache[n]

            before = out[col].dropna().map(_norm_value).nunique()
            out[col] = out[col].map(_code)
            after = out[col].dropna().nunique()
            if before != after:  # praktis tidak mungkin, tapi tetap dicek
                raise RuntimeError(f"Tabrakan kode pada kolom {col}; jalankan ulang.")
            cache.clear()
    finally:
        del run_key  # kunci tidak disimpan di mana pun
    return out


def scan_for_personal_data(df: pd.DataFrame) -> list[str]:
    """Cari pola email / no. HP / NIK di kolom teks. Hanya melaporkan nama kolom & jumlah."""
    warnings = []
    for col in df.columns:
        if col in MASK_COLUMNS or df[col].dtype.kind not in "OSU":
            continue
        s = df[col].dropna().astype(str)
        for label, pat in (("email", _EMAIL), ("nomor HP", _PHONE), ("NIK 16 digit", _NIK)):
            n = int(s.str.contains(pat).sum())
            if n:
                warnings.append(f"Kolom '{col}': {n} sel mirip {label}")
    return warnings


# ---------------------------------------------------------------------------
# BACA FILE
# ---------------------------------------------------------------------------
def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    # dtype=str menjaga nomor panjang (TR/Booking ID) tidak berubah jadi angka ilmiah
    return pd.read_excel(path, dtype={c: str for c in MASK_COLUMNS})


def collect_files(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    files = sorted(p for p in inp.rglob("*")
                   if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .xlsx/.xls/.csv di {inp}")
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Anonimisasi data transaksi hotel untuk mtrax")
    ap.add_argument("--input", required=True, help="File atau folder data mentah")
    ap.add_argument("--output", help="File hasil (.xlsx). Default: mtrax_masked_<waktu>.xlsx")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    out_path = Path(args.output or f"mtrax_masked_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    if out_path.resolve() == inp.resolve():
        print("Output tidak boleh menimpa file input.", file=sys.stderr)
        return 1

    files = collect_files(inp)
    frames, dropped_all, missing_all = [], set(), set()
    for f in files:
        raw = read_any(f)
        sel, dropped, missing = select_columns(raw)
        dropped_all.update(map(str, dropped))
        missing_all.update(missing)
        frames.append(sel)
        del raw
    data = pd.concat(frames, ignore_index=True)
    del frames

    masked = mask_dataframe(data)  # satu kunci untuk semua file dalam proses ini
    del data

    masked.to_excel(out_path, index=False, sheet_name="Data")

    print(f"Selesai: {len(files)} file, {len(masked):,} baris -> {out_path}")
    for col in MASK_COLUMNS:
        if col in masked.columns:
            print(f"  {col}: {masked[col].nunique():,} kode unik")
    if dropped_all:
        print(f"  Kolom dibuang ({len(dropped_all)}): {', '.join(sorted(dropped_all))}")
    if missing_all:
        print(f"  PERINGATAN, kolom tidak ditemukan: {', '.join(sorted(missing_all))}")
    for w in scan_for_personal_data(masked):
        print(f"  PERINGATAN data pribadi: {w}")
    print("  Kunci masking proses ini sudah dibuang; kode tidak bisa dibalik.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
