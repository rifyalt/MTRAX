"""
Standarisasi penulisan Room Category (tipe kamar hotel).

Contoh:
    "DOUBLE ROOM KING BED - DE LUXE"   -> Kelas "Deluxe", Bed "King",  Tipe "Deluxe King"
    "Deluxe Room"                      -> Kelas "Deluxe", Bed "",      Tipe "Deluxe"
    "TWIN/DOUBLE ROOM - SUPERIOR"      -> Kelas "Superior", Bed "Twin/Double"
    "HARRIS ROOM"                      -> Kelas "Standard"

Pemakaian:
    from room_category_standardizer import standardize_room_columns
    df = standardize_room_columns(df)   # menambah kolom Room Class, Bed Type, Room Type Std

Aturan bersifat deterministik (tanpa AI) sehingga hasilnya stabil antar periode data.
"""
import re
import pandas as pd

# ── Kelas kamar, urut dari prioritas tertinggi ────────────────────────────
# Kata pertama yang cocok menentukan kelas. "Deluxe Suite" -> Suite,
# "Executive Deluxe" -> Executive, "Superior Deluxe" -> Deluxe.
ROOM_CLASS_RULES = [
    ("Non-Specific Room Type", None),     # Run of House / Assigned on Arrival (lihat _NONSPEC_RX)
    ("Presidential Suite", r"\bpresidential\b"),
    ("Junior Suite",       r"\bjunior suites?\b|\bjr\.? suites?\b"),
    ("Suite",              r"\bsuites?\b"),
    ("Villa",              r"\bvillas?\b|\bbungalows?\b|\bcottages?\b|\bchalets?\b|\bcabana\b"),
    ("Apartment",          r"\bapartments?\b|\b\d\s*bed ?rooms?\b|\b(one|two|three) bed ?rooms?\b|\bresidence\b"),
    ("Club",               r"\bclub\b|\blounge access\b"),
    ("Executive",          r"\bexecutive\b|\beksekutif\b|\bbusiness\b"),
    ("Premier",            r"\bpremi(er|ere|um)\b"),
    ("Grand Deluxe",       r"\bgrand deluxe\b|\bdeluxe grand\b|\bsuper deluxe\b|\bdeluxe premier\b"),
    ("Deluxe",             r"\bdeluxe\b|\bde luxe\b"),
    ("Superior",           r"\bsuperior\b"),
    ("Studio",             r"\bstudio\b"),
    ("Family",             r"\bfamily\b|\btriple\b|\bquad(ruple)?\b"),
    ("Standard",           r"\bstandard\b|\bclassic\b|\bguest ?room\b|\bharris room\b|\bmoxie\b"
                           r"|\bfave ?room\b|\bfresh ?room\b|\bfab ?room\b|\bsmart\b|\byello\b"
                           r"|\bharris\b|\bcomfort\b|\bkamar\b|\bcondotel\b|^room\b|^\d? ?(king|queen|twin|double|single)( bed)?( room)?$"),
]
ROOM_CLASS_ORDER = [c for c, _ in ROOM_CLASS_RULES] + ["Lainnya"]
# Kamar tanpa tipe spesifik (ditentukan hotel saat check-in)
_NONSPEC_RX = re.compile(
    r"\brun of (the )?house\b|\broh\b|\bassign(ed)?\b.*\b(on|at|upon) arrival\b"
    r"|\b(on|upon) arrival\b|\bnon ?specific\b|\bany room type\b")
_COMPILED = [(c, _NONSPEC_RX if p is None else re.compile(p)) for c, p in ROOM_CLASS_RULES]


def _clean(text) -> str:
    s = str(text).lower()
    s = s.replace("de luxe", "deluxe").replace("de-luxe", "deluxe")
    s = re.sub(r"\bdelux\b|\bdlx\b|\bdeluxed\b", "deluxe", s)
    s = re.sub(r"\bsup\b|\bspr\b", "superior", s)
    s = re.sub(r"\bexec\b", "executive", s)
    s = re.sub(r"\bstd\b", "standard", s)
    s = re.sub(r"\bkingroom\b", "king room", s)
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Kata "umum" yang tidak menyebut kelas: jika nama kamar HANYA berisi kata-kata ini
# (mis. "DOUBLE ROOM KING BED - POOL VIEW"), kamar dianggap kelas Standard.
_GENERIC_WORDS = set("""
room rooms bed beds double twin twins king queen single size with sofa or and for the a
view city pool garden sea seaview ocean lagoon mountain river riverview lake park bay beach
balcony terrace access breakfast include included incl only non smoking nonsmoking smooking
person pax adults large larger new newly modern side hill inner floor high
""".split())


def room_class(text) -> str:
    s = _clean(text)
    if not s or s in ("nan", "none", "-"):
        return "Lainnya"
    for cls, rx in _COMPILED:
        if rx.search(s):
            return cls
    words = [w for w in re.split(r"[ /]+", s) if w and not w.isdigit()]
    if words and all(w in _GENERIC_WORDS for w in words):
        return "Standard"
    return "Lainnya"


def bed_type(text) -> str:
    s = _clean(text)
    has = lambda w: re.search(rf"\b{w}\b", s) is not None
    twin = has("twins?") or re.search(r"\b(2|two) singles?\b", s) is not None
    if twin and (has("double") or has("king") or has("queen") or "/" in s and "double" in s):
        return "Twin/Double"
    if has("king"):
        return "King"
    if has("queen"):
        return "Queen"
    if twin:
        return "Twin"
    if has("double"):
        return "Double"
    if has("single"):
        return "Single"
    return ""


_NO_ROOM_CLASSES = {"Non-Specific Room Type", "Presidential Suite", "Junior Suite", "Suite", "Villa", "Apartment", "Studio", "Lainnya"}


def standardize_room_category(text):
    """Kembalikan (Room Class, Bed Type, Room Type Std)."""
    cls = room_class(text)
    bed = bed_type(text)
    # Tipe bed TIDAK memisahkan tipe kamar: Superior / Superior Double / Superior Twin -> "Superior Room".
    # Informasi bed tetap tersimpan di kolom Bed Type.
    std = cls if cls in _NO_ROOM_CLASSES else f"{cls} Room"
    return cls, bed, std


# ── Nama tipe kamar rinci per hotel ───────────────────────────────────────
# Berbeda dari Room Type Std (kelas umum), nama ini mempertahankan nama khas hotel
# ("Studio Style", "Studio Executive Plus", "Deluxe Premier") tetapi penulisannya
# diseragamkan: huruf, "DE LUXE" -> Deluxe, kata pengisi & tipe bed dibuang.
_NOISE_PHRASES = [
    r"\bwith breakfast\b", r"\bbreakfast included\b", r"\binclude(d)? breakfast\b",
    r"\bincl(ude|uding|uded)?\b", r"\bbreakfast\b", r"\broom only\b", r"\bwith\b",
    r"\bnon ?smoking\b", r"\bno smoking\b", r"\bsmoo?king\b", r"\bsingle use\b",
    r"\bpromo(tion(al)?)?( rate)?\b", r"\b(non )?refundable\b", r"\bbest (flexible )?rate\b",
    r"\bstaycation( offer)?\b", r"\bspecial offer\b", r"\boffer\b", r"\brate\b",
    r"\bfree wi ?fi\b", r"\bwi ?fi\b", r"\b(for )?\d+ (person|pax|adults?)\b",
    r"\brooms?\b", r"\bbeds?\b", r"\bsize\b", r"\bor\b", r"\band\b",
]
_NOISE_RX = re.compile("|".join(_NOISE_PHRASES))
_BED_WORDS = {"king", "queen", "twin", "twins", "double", "single", "super"}
_VIEW_RX = re.compile(r"\b((?:\w+ )?view|pool access|balcony|terrace)\b")


def _title(w: str) -> str:
    return w.upper() if len(w) <= 2 and w.isalpha() and w not in ("of", "by", "in") else w.capitalize()


def clean_room_name(text) -> str:
    raw = _clean(text)
    if not raw or raw in ("nan", "none", "-"):
        return "Lainnya"
    if _NONSPEC_RX.search(raw):
        return "Non-Specific Room Type"
    s = re.sub(r"\bpremiere\b", "premier", raw)
    s = re.sub(r"\b(\d) ?(king|queen|twin|double|single)s?\b", r"\2", s)
    views = [m.group(0) for m in _VIEW_RX.finditer(s)]
    s = _VIEW_RX.sub(" ", s)
    s = _NOISE_RX.sub(" ", s)
    toks = [t for t in re.split(r"[ /]+", s) if t and not t.isdigit()]
    core = [t for t in toks if t not in _BED_WORDS]
    seen, core_u = set(), []
    for t in core:
        if t not in seen:
            seen.add(t); core_u.append(t)
    # kata kelas di depan: "double room king bed - de luxe" -> "Deluxe King"
    cls_words = {"deluxe", "superior", "standard", "executive", "premier", "premiere", "premium",
                 "grand", "junior", "suite", "studio", "family", "club", "classic", "business"}
    lead = [t for t in core_u if t in cls_words]
    rest = [t for t in core_u if t not in cls_words]
    if lead and rest and core_u[0] not in cls_words:
        core_u = lead + rest
    # Tipe bed (Double/Twin/King/Queen/Single) sengaja tidak dimasukkan ke nama:
    # Deluxe, Deluxe Double, Deluxe Twin -> "Deluxe Room".
    if not core_u:
        core_u = ["standard"]         # "Double Room King Bed" -> "Standard Room"
    parts = [" ".join(_title(w) for w in core_u)]
    _no_room = {"suite", "studio", "villa", "apartment", "bungalow", "cottage", "cabana", "chalet", "loft"}
    _cls_end = {"deluxe", "superior", "standard", "executive", "business", "premier", "premium",
                "classic", "family", "club"}
    if not (set(core_u) & _no_room) and (len(core_u) == 1 or core_u[-1] in _cls_end):
        parts.append("Room")          # "Deluxe" -> "Deluxe Room", "Grand Deluxe" -> "Grand Deluxe Room"
    vw = " ".join(dict.fromkeys(v.strip() for v in views))
    if vw:
        parts.append(" ".join(_title(w) for w in vw.split()))
    name = " ".join(p for p in parts if p).strip()
    return name or "Standard"


def standardize_room_columns(df: pd.DataFrame, col: str = "Room Category") -> pd.DataFrame:
    """Tambahkan kolom Room Class, Bed Type, Room Type Std, Room Name Clean ke salinan df."""
    out = df.copy()
    if col not in out.columns:
        return out
    uniq = out[col].astype(str).unique()
    m = {u: standardize_room_category(u) + (clean_room_name(u),) for u in uniq}   # per nilai unik
    raw = out[col].astype(str)
    out["Room Class"] = raw.map(lambda v: m[v][0])
    out["Bed Type"] = raw.map(lambda v: m[v][1])
    out["Room Type Std"] = raw.map(lambda v: m[v][2])
    out["Room Name Clean"] = raw.map(lambda v: m[v][3])
    out["Room Name Clean"] = merge_word_order(out["Room Name Clean"])
    return out


def _word_key(name: str) -> str:
    return " ".join(sorted(w for w in str(name).lower().split() if w != "room"))


def merge_word_order(names: pd.Series) -> pd.Series:
    """Nama dengan kata yang sama tetapi urutan berbeda dianggap satu tipe:
    "Executive Suite" = "Suite Executive", "Premier Deluxe Room" = "Deluxe Premier Room".
    Nama yang ditampilkan = penulisan yang paling sering muncul di data."""
    key = names.map(_word_key)
    cnt = (pd.DataFrame({"_k": key.values, "_name": names.values})
             .groupby(["_k", "_name"]).size().rename("n").reset_index())
    best = (cnt.sort_values(["_k", "n", "_name"], ascending=[True, False, True])
               .drop_duplicates("_k").set_index("_k")["_name"])
    return key.map(best)


# ═════════════════════════════════════════════════════════════════════════
#  STREAMLIT SECTION — ROOM TYPE PRICE BENCHMARK (tab Room Intelligence)
# ═════════════════════════════════════════════════════════════════════════
def _mode(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    return s.value_counts().index[0] if len(s) else ""


def _top_variants(s: pd.Series, n: int = 3) -> str:
    vc = s.astype(str).str.strip().value_counts()
    txt = " · ".join(vc.index[:n].tolist())
    return txt + (f" (+{len(vc) - n} lainnya)" if len(vc) > n else "")


def build_room_type_tables(df: pd.DataFrame, rn_col: str = "Number of Rooms Night",
                           hotel_col: str = "Hotel / Properti"):
    """Hitung (data baris, ringkasan per tipe, ringkasan per hotel×tipe, mapping)."""
    d = standardize_room_columns(df)
    d[rn_col] = pd.to_numeric(d[rn_col], errors="coerce")
    d["Invoice Amount"] = pd.to_numeric(d["Invoice Amount"], errors="coerce")
    d = d[(d[rn_col] > 0) & (d["Invoice Amount"] > 0)].copy()
    d["_rate"] = d["Invoice Amount"] / d[rn_col]
    if hotel_col not in d.columns:
        hotel_col = "Hotel Name"
    _h = d[hotel_col].astype(str).str.strip()
    _hk = _h.str.lower().str.replace(r"\s+", " ", regex=True)
    _disp = _h.groupby(_hk).agg(lambda x: x.value_counts().index[0])
    d["_hotel"] = _hk.map(_disp)

    def _agg(g):
        r = g.agg(**{
            "Transaksi": ("_rate", "size"),
            "Room Nights": (rn_col, "sum"),
            "Total Spend": ("Invoice Amount", "sum"),
            "Median Rate/Night": ("_rate", "median"),
            "Min Rate/Night": ("_rate", "min"),
            "Max Rate/Night": ("_rate", "max"),
        })
        r["Avg Rate/Night"] = r["Total Spend"] / r["Room Nights"]
        return r

    by_type = _agg(d.groupby(["Room Type Std", "Room Class"]))
    by_type["Jumlah Hotel"] = d.groupby(["Room Type Std", "Room Class"])["_hotel"].nunique()
    by_type = by_type.reset_index().sort_values(["Transaksi", "Room Nights"], ascending=False)

    by_hotel = _agg(d.groupby(["_hotel", "Room Type Std", "Room Class"]))
    by_hotel["Penulisan Asli"] = d.groupby(["_hotel", "Room Type Std", "Room Class"])["Room Category"].agg(_top_variants)
    by_hotel = by_hotel.reset_index().rename(columns={"_hotel": "Hotel"})
    if "City" in d.columns:
        city = d.groupby("_hotel")["City"].agg(_mode)
        by_hotel.insert(1, "City", by_hotel["Hotel"].map(city))
    by_hotel = by_hotel.sort_values(["Transaksi", "Room Nights"], ascending=False)

    mapping = (d.groupby(["Room Category", "Room Type Std", "Room Class"])
                 .size().rename("Transaksi").reset_index()
                 .rename(columns={"Room Category": "Room Category (Asli)"})
                 .sort_values("Transaksi", ascending=False))
    # Hotel -> tipe kamar rinci (Room Name Clean)
    g = d.groupby(["_hotel", "Room Name Clean"])
    by_hotel_room = g.agg(**{
        "Transaksi": ("_rate", "size"),
        "Total Room Nights": (rn_col, "sum"),
        "Total Spend": ("Invoice Amount", "sum"),
        "Room Class": ("Room Class", _mode),
    })
    by_hotel_room["Penulisan Asli"] = g["Room Category"].agg(lambda s: _top_variants(s, n=50))
    by_hotel_room = by_hotel_room.reset_index().rename(columns={"_hotel": "Hotel", "Room Name Clean": "Tipe Kamar"})
    by_hotel_room["Avg Rate / Room Night"] = by_hotel_room["Total Spend"] / by_hotel_room["Total Room Nights"]
    by_hotel_room["Avg Room Night / Transaksi"] = by_hotel_room["Total Room Nights"] / by_hotel_room["Transaksi"]
    if "City" in d.columns:
        by_hotel_room["City"] = by_hotel_room["Hotel"].map(d.groupby("_hotel")["City"].agg(_mode))
    return d, by_type, by_hotel, mapping, by_hotel_room


def grouped_hotel_excel_bytes(tbl: pd.DataFrame, cols, sheet: str = "Hotel - Tipe Kamar"):
    """Excel berformat kelompok: baris hotel (tebal, berwarna) diikuti baris tipe kamarnya."""
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        wb = w.book
        ws = wb.add_worksheet(sheet)
        w.sheets[sheet] = ws
        f_head = wb.add_format({"bold": True, "bg_color": "#062440", "font_color": "white", "border": 1})
        f_hotel = wb.add_format({"bold": True, "bg_color": "#DDEFFA", "border": 1})
        f_hotel_n = wb.add_format({"bold": True, "bg_color": "#DDEFFA", "border": 1, "num_format": "#,##0"})
        f_hotel_d = wb.add_format({"bold": True, "bg_color": "#DDEFFA", "border": 1, "num_format": "#,##0.00"})
        f_hotel_m = wb.add_format({"bold": True, "bg_color": "#DDEFFA", "border": 1, "num_format": '"Rp"#,##0'})
        f_room = wb.add_format({"indent": 2, "border": 1})
        f_n = wb.add_format({"num_format": "#,##0", "border": 1})
        f_d = wb.add_format({"num_format": "#,##0.00", "border": 1})
        f_m = wb.add_format({"num_format": '"Rp"#,##0', "border": 1})
        f_t = wb.add_format({"border": 1, "font_color": "#6a8fa0"})
        headers = ["Hotel / Tipe Kamar"] + list(cols)
        for j, h in enumerate(headers):
            ws.write(0, j, h, f_head)
        r = 1
        for hotel, grp in tbl.groupby("Hotel", sort=False):
            city = grp["City"].iloc[0] if "City" in grp.columns else ""
            ws.write(r, 0, f"{hotel}" + (f"  ·  {city}" if city else ""), f_hotel)
            tot = {"Transaksi": grp["Transaksi"].sum(), "Total Room Nights": grp["Total Room Nights"].sum()}
            tot["Avg Room Night / Transaksi"] = tot["Total Room Nights"] / tot["Transaksi"]
            tot["Avg Rate / Room Night"] = grp["Total Spend"].sum() / tot["Total Room Nights"]
            for j, c in enumerate(cols, start=1):
                v = tot.get(c, "")
                fm = f_hotel_m if "Rate" in c else f_hotel_d if "/ Transaksi" in c else f_hotel_n if c in tot else f_hotel
                ws.write(r, j, v, fm)
            r += 1
            for _, row in grp.iterrows():
                ws.write(r, 0, row["Tipe Kamar"], f_room)
                for j, c in enumerate(cols, start=1):
                    fm = f_m if "Rate" in c else f_d if "/ Transaksi" in c else f_n if c in ("Transaksi", "Total Room Nights") else f_t
                    ws.write(r, j, row[c], fm)
                r += 1
        ws.set_column(0, 0, 48)
        ws.set_column(1, len(cols), 18)
        if "Penulisan Asli" in cols:
            ws.set_column(len(cols), len(cols), 60)
        ws.freeze_panes(1, 1)
    buf.seek(0)
    return buf


def render_room_type_benchmark(df: pd.DataFrame, rn_col: str = "Number of Rooms Night",
                               hotel_col: str = "Hotel / Properti", filter_label: str = "",
                               is_admin: bool = False, excel_fn=None, key: str = "rtb"):
    import streamlit as st
    import plotly.graph_objects as go
    from datetime import datetime

    st.markdown('<div class="pi-section-divider">Room Type Price Benchmark</div>', unsafe_allow_html=True)

    need = ["Room Category", "Invoice Amount", rn_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.markdown('<div class="pi-insight">ℹ️ Analisis tipe kamar membutuhkan kolom '
                    + ", ".join(f"<b>{c}</b>" for c in miss) + ".</div>", unsafe_allow_html=True)
        return

    d, by_type, by_hotel, mapping, by_hotel_room = build_room_type_tables(df, rn_col, hotel_col)
    if d.empty:
        st.markdown('<div class="pi-empty">⚠️ Tidak ada transaksi dengan room night & nominal valid.</div>',
                    unsafe_allow_html=True)
        return

    n_raw = d["Room Category"].astype(str).str.strip().nunique()
    n_std = d["Room Type Std"].nunique()
    pct_ok = (d["Room Class"] != "Lainnya").mean() * 100
    st.markdown(f"""
    <div style='background:#f0f8ff;border:1px solid #cce4f4;border-left:3px solid #1BA0E2;
                border-radius:6px;padding:10px 16px;margin-bottom:14px;font-size:0.82em;color:#0D7FCC;'>
        🛏️ <b>{n_raw:,}</b> variasi penulisan Room Category diseragamkan menjadi
        <b>{n_std:,}</b> tipe kamar standar · <b>{pct_ok:.1f}%</b> transaksi dikenali kelasnya
        {('· ' + filter_label) if filter_label else ''}
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.6, 2, 1, 2])
    with c1:
        view = st.radio("Tampilan", ["Tabel Hotel → Tipe Kamar", "Per Tipe Kamar", "Per Hotel × Tipe Kamar",
                                     "Profil Kamar per Hotel"],
                        horizontal=True, key=f"{key}_view")
    is_profile = view in ("Profil Kamar per Hotel", "Tabel Hotel → Tipe Kamar")
    with c2:
        classes = [c for c in ROOM_CLASS_ORDER if c in set(d["Room Class"])]
        sel_cls = st.multiselect("Kelas kamar", classes, default=[], placeholder="Semua kelas",
                                 key=f"{key}_cls")
    with c3:
        min_trx = st.number_input("Min. transaksi", min_value=1, value=3, step=1, key=f"{key}_min",
                                  disabled=is_profile,
                                  help="Tidak berlaku di Tabel Hotel → Tipe Kamar & Profil Kamar per Hotel (semua tipe kamar ditampilkan).")
        if is_profile:
            min_trx = 1
    with c4:
        q = st.text_input("Cari hotel / tipe kamar", value="", placeholder="mis. Deluxe King, Aston…",
                          key=f"{key}_q").strip()

    money = {c: "Rp{:,.0f}" for c in ["Avg Rate/Night", "Median Rate/Night", "Min Rate/Night",
                                      "Max Rate/Night", "Total Spend"]}
    ints = {c: "{:,.0f}" for c in ["Transaksi", "Room Nights", "Jumlah Hotel"]}

    def _filt(t, cols):
        t = t[t["Transaksi"] >= min_trx]
        if sel_cls:
            t = t[t["Room Class"].isin(sel_cls)]
        if q:
            m = pd.Series(False, index=t.index)
            for c in cols:
                m |= t[c].astype(str).str.contains(q, case=False, na=False, regex=False)
            t = t[m]
        return t.reset_index(drop=True)

    if view == "Tabel Hotel → Tipe Kamar":
        import html as _html
        hr = by_hotel_room.copy()
        if sel_cls:
            hr = hr[hr["Room Class"].isin(sel_cls)]
        if q:
            m = pd.Series(False, index=hr.index)
            for c in ["Hotel", "Tipe Kamar", "City", "Penulisan Asli"]:
                if c in hr.columns:
                    m |= hr[c].astype(str).str.contains(q, case=False, na=False, regex=False)
            hr = hr[m]
        if hr.empty:
            st.markdown('<div class="pi-empty">⚠️ Tidak ada hotel / tipe kamar yang sesuai filter.</div>',
                        unsafe_allow_html=True)
            return
        hot_rn = hr.groupby("Hotel")["Total Room Nights"].sum().sort_values(ascending=False)
        n_hotel = len(hot_rn)
        opts = [o for o in (10, 20, 50, 100) if o < n_hotel] + [n_hotel]
        n_show = st.selectbox("Jumlah hotel ditampilkan (urut Total Room Nights terbesar)", opts,
                              index=min(1, len(opts) - 1),
                              format_func=lambda v: f"Semua ({v:,} hotel)" if v == n_hotel else f"Top {v}",
                              key=f"{key}_nhotel")
        order = {h: i for i, h in enumerate(hot_rn.index)}
        hr = hr.assign(_o=hr["Hotel"].map(order)).sort_values(["_o", "Total Room Nights", "Transaksi"],
                                                              ascending=[True, False, False])
        shown = hr[hr["_o"] < n_show]

        cols_t = ["Transaksi", "Total Room Nights", "Avg Room Night / Transaksi", "Avg Rate / Room Night"]
        th = "".join(f"<th>{c}</th>" for c in ["Hotel / Tipe Kamar"] + cols_t + ["Penulisan Asli"])
        rows = []
        for hotel, grp in shown.groupby("Hotel", sort=False):
            city = grp["City"].iloc[0] if "City" in grp.columns else ""
            t_trx, t_rn = grp["Transaksi"].sum(), grp["Total Room Nights"].sum()
            t_rate = grp["Total Spend"].sum() / t_rn if t_rn else float("nan")
            rows.append(
                f"<tr class='h'><td>🏨 {_html.escape(str(hotel))}"
                f"<span class='c'>{_html.escape(str(city))} · {len(grp)} tipe kamar</span></td>"
                f"<td>{t_trx:,.0f}</td><td>{t_rn:,.0f}</td><td>{t_rn / t_trx:,.2f}</td>"
                f"<td>Rp{t_rate:,.0f}</td><td></td></tr>")
            for _, r in grp.iterrows():
                rows.append(
                    f"<tr><td class='r'>{_html.escape(str(r['Tipe Kamar']))}</td>"
                    f"<td>{r['Transaksi']:,.0f}</td><td>{r['Total Room Nights']:,.0f}</td>"
                    f"<td>{r['Avg Room Night / Transaksi']:,.2f}</td>"
                    f"<td>Rp{r['Avg Rate / Room Night']:,.0f}</td>"
                    f"<td class='o'>{_html.escape(str(r['Penulisan Asli']))}</td></tr>")
        st.markdown(f"""
        <style>
        .rtb-wrap {{ max-height: 640px; overflow: auto; border: 1px solid #cce4f4; border-radius: 8px; }}
        .rtb {{ width: 100%; border-collapse: collapse; font-size: 0.84em; }}
        .rtb th {{ position: sticky; top: 0; background: #062440; color: #fff; text-align: right;
                  padding: 8px 12px; font-weight: 600; white-space: nowrap; z-index: 1; }}
        .rtb th:first-child, .rtb th:last-child {{ text-align: left; }}
        .rtb td {{ padding: 6px 12px; border-bottom: 1px solid #e8f4fd; text-align: right; white-space: nowrap; }}
        .rtb td:first-child {{ text-align: left; }}
        .rtb tr.h td {{ background: #DDEFFA; font-weight: 700; color: #062440; border-top: 2px solid #1BA0E2; }}
        .rtb tr.h .c {{ font-weight: 400; color: #6a8fa0; margin-left: 10px; font-size: 0.9em; }}
        .rtb td.r {{ padding-left: 34px; color: #0D3B5E; }}
        .rtb td.o {{ text-align: left; color: #6a8fa0; font-size: 0.92em; white-space: normal; min-width: 260px; }}
        </style>
        <div class="rtb-wrap"><table class="rtb"><thead><tr>{th}</tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div>
        """, unsafe_allow_html=True)
        st.caption(f"Menampilkan {min(n_show, n_hotel):,} dari {n_hotel:,} hotel · "
                   "Avg Rate / Room Night = total Invoice Amount ÷ total Room Nights · "
                   "Avg Room Night / Transaksi = rata-rata lama menginap per transaksi.")
        if is_admin:
            st.download_button(
                "⬇️ Download Excel (semua hotel sesuai filter)",
                data=grouped_hotel_excel_bytes(hr, cols_t + ["Penulisan Asli"]),
                file_name=f"hotel_tipe_kamar_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key}_dl_grp")
        is_admin = False   # tombol download standar di bawah tidak dipakai untuk tampilan ini
        t = None
    elif view == "Per Tipe Kamar":
        t = _filt(by_type, ["Room Type Std", "Room Class"])
        cols = ["Room Type Std", "Room Class", "Transaksi", "Room Nights", "Jumlah Hotel",
                "Avg Rate/Night", "Median Rate/Night", "Min Rate/Night", "Max Rate/Night"]
        t = t[cols].rename(columns={"Room Type Std": "Tipe Kamar Standar", "Room Class": "Kelas"})
        if t.empty:
            st.markdown('<div class="pi-empty">⚠️ Tidak ada tipe kamar yang sesuai filter.</div>',
                        unsafe_allow_html=True)
            return
        top = t.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top["Avg Rate/Night"], y=top["Tipe Kamar Standar"], orientation="h",
            marker_color="#1BA0E2", customdata=top[["Transaksi", "Room Nights", "Median Rate/Night"]],
            text=[f"Rp{v/1e6:,.2f} jt" for v in top["Avg Rate/Night"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg Rp%{x:,.0f}/malam<br>Median Rp%{customdata[2]:,.0f}"
                          "<br>%{customdata[0]:,} transaksi · %{customdata[1]:,} RN<extra></extra>"))
        fig.update_layout(template="plotly_white", height=max(320, 28 * len(top) + 90),
                          title=dict(text="Avg Rate/Night · 15 Tipe Kamar Paling Sering Dipakai", font=dict(size=13)),
                          xaxis=dict(title="Rata-rata harga per malam (Rp)", gridcolor="#e8f4fd",
                                     range=[0, top["Avg Rate/Night"].max() * 1.18]),
                          margin=dict(l=10, r=30, t=50, b=40), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(t.style.format({**money, **ints}, na_rep="-"),
                     use_container_width=True, hide_index=True, height=420)
        sheet, fname, mcols, icols = "Room Type", "room_type_benchmark", \
            ("Avg Rate/Night", "Median Rate/Night", "Min Rate/Night", "Max Rate/Night"), \
            ("Transaksi", "Room Nights", "Jumlah Hotel")
    elif is_profile:
        bh = _filt(by_hotel, ["Hotel", "Room Type Std", "Room Class", "City"] if "City" in by_hotel else
                   ["Hotel", "Room Type Std", "Room Class"])
        if bh.empty:
            st.markdown('<div class="pi-empty">⚠️ Tidak ada hotel yang sesuai filter.</div>',
                        unsafe_allow_html=True)
            return
        rank = {c: i for i, c in enumerate(ROOM_CLASS_ORDER)}
        bh = bh.assign(_r=bh["Room Class"].map(rank)).sort_values(["Hotel", "_r", "Transaksi"],
                                                                 ascending=[True, True, False])
        prof = bh.groupby("Hotel").agg(**{
            "Jumlah Tipe Kamar": ("Room Type Std", "nunique"),
            "Transaksi": ("Transaksi", "sum"),
            "Room Nights": ("Room Nights", "sum"),
            "Total Spend": ("Total Spend", "sum"),
            "Rate Tipe Termurah": ("Avg Rate/Night", "min"),
            "Rate Tipe Termahal": ("Avg Rate/Night", "max"),
        })
        prof["Tipe Kamar Terdeteksi"] = bh.groupby("Hotel").apply(
            lambda g: " · ".join(f"{r} ({n})" for r, n in zip(g["Room Type Std"], g["Transaksi"])))
        prof["Avg Rate/Night"] = prof["Total Spend"] / prof["Room Nights"]
        if "City" in bh.columns:
            prof.insert(0, "City", bh.groupby("Hotel")["City"].first())
        prof = prof.reset_index().sort_values(["Transaksi", "Room Nights"], ascending=False)
        pcols = ["Hotel"] + (["City"] if "City" in prof.columns else []) + \
                ["Jumlah Tipe Kamar", "Tipe Kamar Terdeteksi", "Transaksi", "Room Nights",
                 "Avg Rate/Night", "Rate Tipe Termurah", "Rate Tipe Termahal"]
        t = prof[pcols].rename(columns={"Hotel": "Hotel / Properti"}).reset_index(drop=True)
        money.update({c: "Rp{:,.0f}" for c in ["Rate Tipe Termurah", "Rate Tipe Termahal"]})
        ints["Jumlah Tipe Kamar"] = "{:,.0f}"
        st.caption(f"{len(t):,} hotel · angka dalam kurung = jumlah transaksi tipe kamar tersebut · "
                   "hanya tipe kamar yang pernah dipesan pada periode & filter terpilih")
        st.dataframe(t.style.format({**money, **ints}, na_rep="-"),
                     use_container_width=True, hide_index=True, height=380,
                     column_config={"Tipe Kamar Terdeteksi": st.column_config.TextColumn(width="large")})

        st.markdown("<div style='font-size:0.72em;font-weight:600;color:#6a8fa0;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin:18px 0 6px;'>Detail tipe kamar per hotel</div>",
                    unsafe_allow_html=True)
        _n = dict(zip(t["Hotel / Properti"], t["Jumlah Tipe Kamar"]))
        sel_h = st.selectbox("Pilih hotel", t["Hotel / Properti"].tolist(),
                             format_func=lambda h: f"{h}  ·  {_n[h]} tipe kamar",
                             label_visibility="collapsed", key=f"{key}_hotel")
        dh = bh[bh["Hotel"] == sel_h]
        dsub = d[d["_hotel"] == sel_h]
        if sel_cls:
            dsub = dsub[dsub["Room Class"].isin(sel_cls)]
        allvar = dsub.groupby("Room Type Std")["Room Category"].agg(lambda s: _top_variants(s, n=50))
        dt = dh.assign(**{"Semua Penulisan Asli": dh["Room Type Std"].map(allvar)})
        dt = dt[["Room Type Std", "Room Class", "Transaksi", "Room Nights", "Avg Rate/Night",
                 "Median Rate/Night", "Min Rate/Night", "Max Rate/Night", "Semua Penulisan Asli"]] \
            .rename(columns={"Room Type Std": "Tipe Kamar Standar", "Room Class": "Kelas"})
        if len(dt) > 1:
            fig = go.Figure(go.Bar(
                x=dt["Tipe Kamar Standar"], y=dt["Avg Rate/Night"], marker_color="#1BA0E2",
                text=[f"Rp{v/1e6:,.2f} jt" for v in dt["Avg Rate/Night"]], textposition="outside",
                customdata=dt[["Transaksi"]],
                hovertemplate="<b>%{x}</b><br>Avg Rp%{y:,.0f}/malam<br>%{customdata[0]:,} transaksi<extra></extra>"))
            fig.update_layout(template="plotly_white", height=320,
                              title=dict(text=f"Avg Rate/Night per Tipe Kamar · {sel_h}", font=dict(size=13)),
                              yaxis=dict(title="Rp / malam", gridcolor="#e8f4fd",
                                         range=[0, dt["Avg Rate/Night"].max() * 1.2]),
                              margin=dict(l=60, r=30, t=50, b=40), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dt.style.format({**money, **ints}, na_rep="-"),
                     use_container_width=True, hide_index=True)
        sheet, fname, mcols, icols = "Profil Kamar Hotel", "hotel_room_profile", \
            ("Avg Rate/Night", "Rate Tipe Termurah", "Rate Tipe Termahal"), \
            ("Jumlah Tipe Kamar", "Transaksi", "Room Nights")
    else:
        t = _filt(by_hotel, ["Hotel", "Room Type Std", "Room Class", "City"] if "City" in by_hotel else
                  ["Hotel", "Room Type Std", "Room Class"])
        cols = ["Hotel"] + (["City"] if "City" in t.columns else []) + \
               ["Room Type Std", "Room Class", "Transaksi", "Room Nights", "Avg Rate/Night",
                "Median Rate/Night", "Min Rate/Night", "Max Rate/Night", "Penulisan Asli"]
        t = t[cols].rename(columns={"Hotel": "Hotel / Properti", "Room Type Std": "Tipe Kamar Standar",
                                    "Room Class": "Kelas"})
        if t.empty:
            st.markdown('<div class="pi-empty">⚠️ Tidak ada kombinasi hotel × tipe kamar yang sesuai filter.</div>',
                        unsafe_allow_html=True)
            return
        st.caption(f"{len(t):,} kombinasi hotel × tipe kamar · diurutkan dari transaksi terbanyak")
        st.dataframe(t.style.format({**money, **ints}, na_rep="-"),
                     use_container_width=True, hide_index=True, height=480)
        sheet, fname, mcols, icols = "Hotel x Room Type", "hotel_room_type_benchmark", \
            ("Avg Rate/Night", "Median Rate/Night", "Min Rate/Night", "Max Rate/Night"), \
            ("Transaksi", "Room Nights")

    st.caption("Avg Rate/Night = total Invoice Amount ÷ total Number of Rooms Night. "
               "Median/Min/Max dihitung dari harga per malam tiap transaksi.")

    if is_admin and excel_fn is not None:
        st.download_button("⬇️ Download Excel", data=excel_fn(t, sheet, money_cols=mcols, int_cols=icols),
                           file_name=f"{fname}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"{key}_dl")

    with st.expander("🔍 Cek hasil standarisasi: Room Category asli → Tipe Kamar Standar", expanded=False):
        mq = st.text_input("Cari penulisan asli / tipe standar", key=f"{key}_mq").strip()
        mp = mapping
        if mq:
            mp = mp[mp["Room Category (Asli)"].astype(str).str.contains(mq, case=False, regex=False)
                    | mp["Room Type Std"].str.contains(mq, case=False, regex=False)]
        st.dataframe(mp.rename(columns={"Room Type Std": "Tipe Kamar Standar", "Room Class": "Kelas"}),
                     use_container_width=True, hide_index=True, height=360)
        st.caption("Kelas 'Lainnya' = nama kamar khas hotel yang tidak menyebut kelas (mis. PRIVILEGE ROOM, NEO ROOM).")
