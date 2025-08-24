# data.py
from __future__ import annotations

import os
import glob
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# ---- Konstanter ----
# Merk: vi fyller IKKE rank/rank_points her – rader uten rank droppes under.
DEFAULTS = {
    "winner_age": 25.0,
    "loser_age": 25.0,
    "winner_ht": 180.0,
    "loser_ht": 180.0,
}

MIN_COLUMNS = [
    "tourney_date",
    "tourney_level",
    "surface",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "winner_age",
    "loser_age",
    "winner_ht",
    "loser_ht",
]


def load_matches(
    data_dir: str = "data/raw/tennis_atp",
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Laster ATP-kamper (atp_matches_YYYY.csv):
    - normaliserer kolonnenavn
    - parser 'tourney_date' (YYYYMMDD)
    - legger til 'year'
    - fjerner rader uten 'winner_rank' / 'loser_rank'
    - fyller alder/høyde med nøytrale defaults
    - returnerer én DataFrame
    """
    pattern = os.path.join(data_dir, "atp_matches_*.csv")
    files = sorted(glob.glob(pattern))
    if years:
        years = {int(y) for y in years}
        files = [f for f in files if _year_from_filename(f) in years]

    if not files:
        raise FileNotFoundError(f"Fant ingen filer i {pattern}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        year_from_name = _year_from_filename(f)
        df = pd.read_csv(f, low_memory=False)
        df = _normalize_columns(df)

        # tourney_date finnes typisk som int YYYYMMDD
        if "tourney_date" in df.columns:
            df["tourney_date"] = _parse_tourney_date(df["tourney_date"])

        # year: fra filnavn først, ellers fra dato
        if "year" not in df.columns:
            if year_from_name is not None:
                df["year"] = int(year_from_name)
            elif "tourney_date" in df.columns:
                df["year"] = df["tourney_date"].dt.year
            else:
                raise ValueError("Kan ikke avgjøre 'year' for fil: " + f)

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = _ensure_min_columns(out)

    # ---- Kvalitetsfilter: behold kun rader med gyldig ranking ----
    out = out.dropna(subset=["winner_rank", "loser_rank"])

    # ---- Fyll nøytrale defaults for alder/høyde ----
    out = _fill_missing_defaults(out)

    return out


def list_players(df: pd.DataFrame) -> List[str]:
    """Returnerer sortert unikeliste av spillernavn i datasettet."""
    players = pd.concat([df["winner_name"], df["loser_name"]], ignore_index=True)
    return sorted(players.dropna().unique().tolist())


def train_test_years(df: pd.DataFrame) -> Tuple[List[int], int]:
    """Bruk alle år bortsett fra siste som trening, siste som test."""
    years_sorted = sorted(int(y) for y in df["year"].dropna().unique())
    if len(years_sorted) < 2:
        raise ValueError("Trenger minst to år for å lage train/test-splitt.")
    return years_sorted[:-1], years_sorted[-1]


# ---- Interne hjelpere ----
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _parse_tourney_date(col: pd.Series) -> pd.Series:
    col_str = col.astype(str).str.slice(0, 8)
    return pd.to_datetime(col_str, format="%Y%m%d", errors="coerce")


def _year_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    try:
        return int(base.split("_")[-1].split(".")[0])
    except Exception:
        return None


def _ensure_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MIN_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _fill_missing_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default
    return df
