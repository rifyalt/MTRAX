import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotel_name_standardizer import (  # noqa: E402
    apply_brand_rules, load_mapping, standardize_hotel_columns, standardize_name,
)


@pytest.mark.parametrize("raw,expected", [
    ("Melia Bali", "Meliá Bali"),
    ("MELIA HANOI", "Meliá Hanoi"),
    ("Gran Melia Jakarta", "Hotel Gran Meliá Jakarta"),
    ("MOVENPICK SURABAYA CITY", "Mövenpick Surabaya City"),
    ("MOVENPICK RESORT AND SPA BALI", "Mövenpick Resort & Spa Jimbaran Bali"),
    ("EL HOTEL BANDUNG", "éL Hotel Bandung"),
    ("eL Hotel Banyuwangi", "éL Hotel Banyuwangi"),
    ("Courtyard by Marriot Bandung Dago", "Courtyard by Marriott Bandung Dago"),
    ("COURTYARD BY MARRIOTT BANDUNG DAGO", "Courtyard by Marriott Bandung Dago"),
    ("Courtyard Bandung Dago", "Courtyard by Marriott Bandung Dago"),
])
def test_known_names(raw, expected):
    assert standardize_name(raw) == expected


@pytest.mark.parametrize("raw", ["Parmelia Hilton Perth", "The Melio Enim Hotel", "Avenzel Hotel and Convention Cibubur"])
def test_brand_rules_do_not_touch_other_words(raw):
    assert apply_brand_rules(raw) == raw


def test_unknown_name_only_gets_brand_rules():
    assert standardize_name("Melia Hotel Antah Berantah") == "Meliá Hotel Antah Berantah"
    assert standardize_name("Hotel Tidak Dikenal") == "Hotel Tidak Dikenal"


def test_idempotent():
    mapping, _ = load_mapping()
    for v in set(mapping.values()):
        assert standardize_name(v, mapping) == v


def test_dataframe_columns_and_missing_values():
    df = pd.DataFrame({
        "Hotel Name": ["melia bali", None, "EL HOTEL MALANG"],
        "Supplier Name": ["Movenpick Hanoi", "", None],
        "City": ["Bali", "Jakarta", "Malang"],
    })
    out = standardize_hotel_columns(df)
    assert out.loc[0, "Hotel Name"] == "Meliá Bali"
    assert out.loc[2, "Hotel Name"] == "éL Hotel Malang"
    assert out.loc[0, "Supplier Name"].startswith("Mövenpick")
    assert pd.isna(out.loc[1, "Hotel Name"])
    assert out["City"].tolist() == df["City"].tolist()


@pytest.mark.parametrize("a,b", [
    ("Aryaduta Bandung", "ARYADUTA HOTEL BANDUNG"),
    ("Hotel Aston Manado", "Aston Manado Hotel"),
    ("The Akmani Legian", "AKMANI LEGIAN"),
    ("Alpha Hotel Pekanbaru", "Hotel Alpha Pekanbaru"),
])
def test_variants_only_differing_by_hotel_or_the_are_merged(a, b):
    assert standardize_name(a) == standardize_name(b)


def test_different_hotels_stay_separate():
    assert standardize_name("AC Hotel Kuantan") != standardize_name("Patra Jakarta")
    assert standardize_name("Patra Bandung") != standardize_name("Patra Comfort Palembang")
