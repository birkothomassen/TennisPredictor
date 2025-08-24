# features.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Iterable, Optional

import pandas as pd


# -------------------- Elo (surface-spesifikk) --------------------
class SurfaceElo:
    def __init__(self, k: float = 32.0, base: float = 1500.0):
        self.k = k
        self.base = base
        # ratings[surface][player] -> rating (vanlige dicts, pickle-vennlig)
        self.ratings: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def get(self, player_a: str, player_b: str, surface: str) -> Tuple[float, float]:
        surf_dict = self.ratings.get(surface, {})
        ra = surf_dict.get(player_a, self.base)
        rb = surf_dict.get(player_b, self.base)
        return ra, rb

    def update(self, winner: str, loser: str, surface: str) -> None:
        surf_dict = self.ratings.setdefault(surface, {})
        ra = surf_dict.get(winner, self.base)
        rb = surf_dict.get(loser, self.base)
        ea = self._expected(ra, rb)
        surf_dict[winner] = ra + self.k * (1 - ea)
        surf_dict[loser] = rb - self.k * (1 - ea)
        self.ratings[surface] = surf_dict


# -------------------- Fabrikker (pickle-vennlige) --------------------
def _h2h_leaf() -> Dict[str, int]:
    return {"wins": 0, "losses": 0, "total": 0}

def _recent_leaf() -> Dict[str, int]:
    return {"wins": 0, "losses": 0}


# -------------------- State --------------------
@dataclass
class StatsState:
    """
    Holder all historikk som trengs for featurebygging og senere prediksjon.
    (Kun toppnivå-funksjoner/vanlige dicts → pickle-vennlig.)
    """
    # h2h[A][B] -> {"wins","losses","total"}
    h2h: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    surface_elo: SurfaceElo = field(default_factory=SurfaceElo)
    # recent_form[player] -> {"wins","losses"}
    recent_form: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Viktig: rekkefølge må matche treningsdata
    feature_order: List[str] = field(default_factory=lambda: [
        # Ranking (lavere tall = bedre; inkluder diff + råverdier)
        "rank_diff",
        "winner_rank",
        "loser_rank",
        # ATP rank points (høyere er bedre)
        "points_diff",
        "winner_pts",
        "loser_pts",
        # Demografi/antropometri
        "age_diff",
        "ht_diff",
        # Historikk
        "h2h_net",
        "surface_elo_diff",
        "tourney_importance",
        "recent_form_diff",
    ])


TL_MAP = {"G": 4, "M": 3, "A": 2, "C": 1, "F": 1}


# -------------------- Hjelpere for state --------------------
def _ensure_h2h_pair(h2h: Dict[str, Dict[str, Dict[str, int]]], a: str, b: str) -> None:
    if a not in h2h:
        h2h[a] = {}
    if b not in h2h[a]:
        h2h[a][b] = _h2h_leaf()
    if b not in h2h:
        h2h[b] = {}
    if a not in h2h[b]:
        h2h[b][a] = _h2h_leaf()

def _ensure_recent(recent: Dict[str, Dict[str, int]], p: str) -> None:
    if p not in recent:
        recent[p] = _recent_leaf()


# -------------------- Featurebygging for én kamp --------------------
def _recent_rate(recent_form: Dict[str, Dict[str, int]], player: str) -> float:
    if player not in recent_form:
        _ensure_recent(recent_form, player)
    w = recent_form[player]["wins"]
    l = recent_form[player]["losses"]
    tot = max(w + l, 1)
    return w / tot


def _h2h_net(h2h: Dict[str, Dict[str, Dict[str, int]]], a: str, b: str) -> int:
    _ensure_h2h_pair(h2h, a, b)
    return h2h[a][b]["wins"] - h2h[b][a]["wins"]


def _surface_elo_diff(elo: SurfaceElo, a: str, b: str, surface: str) -> float:
    ra, rb = elo.get(a, b, surface)
    return ra - rb


def _tourney_importance(level: Optional[str]) -> int:
    return TL_MAP.get((level or "A"), 1)


def build_feature_row(
    state: StatsState,
    winner: str,
    loser: str,
    *,
    surface: str = "unknown",
    tourney_level: str = "A",
    # Ranking
    winner_rank: float = 1000,
    loser_rank: float = 1000,
    # Rank points
    winner_rank_points: float = 0.0,
    loser_rank_points: float = 0.0,
    # Demografi/antropometri
    winner_age: float = 25.0,
    loser_age: float = 25.0,
    winner_ht: float = 180.0,
    loser_ht: float = 180.0,
) -> Dict[str, float]:
    """
    Lager ett feature-row gitt gjeldende state (uten å oppdatere state).
    Brukes både i trening og i runtime-prediksjon.
    """
    feat = {
        # Ranking (lavere tall = bedre). Positiv rank_diff betyr at vinner har bedre (lavere) rank.
        "rank_diff": (loser_rank - winner_rank),
        "winner_rank": winner_rank,
        "loser_rank": loser_rank,

        # ATP points (høyere = bedre)
        "points_diff": (winner_rank_points - loser_rank_points),
        "winner_pts": winner_rank_points,
        "loser_pts": loser_rank_points,

        # Demografi/antropometri
        "age_diff": winner_age - loser_age,
        "ht_diff": winner_ht - loser_ht,

        # Historikk
        "h2h_net": _h2h_net(state.h2h, winner, loser),
        "surface_elo_diff": _surface_elo_diff(state.surface_elo, winner, loser, surface),
        "tourney_importance": _tourney_importance(tourney_level),
        "recent_form_diff": _recent_rate(state.recent_form, winner) - _recent_rate(state.recent_form, loser),
    }
    return feat


def update_state_after_match(
    state: StatsState,
    winner: str,
    loser: str,
    *,
    surface: str = "unknown",
) -> None:
    """Oppdaterer H2H, Elo og recent form etter én kamp."""
    # H2H
    _ensure_h2h_pair(state.h2h, winner, loser)
    state.h2h[winner][loser]["wins"] += 1
    state.h2h[winner][loser]["total"] += 1
    state.h2h[loser][winner]["losses"] += 1
    state.h2h[loser][winner]["total"] += 1
    # Elo
    state.surface_elo.update(winner, loser, surface)
    # Recent form
    _ensure_recent(state.recent_form, winner)
    _ensure_recent(state.recent_form, loser)
    state.recent_form[winner]["wins"] += 1
    state.recent_form[loser]["losses"] += 1


# -------------------- Batch-bygging (trening/test) --------------------
def build_datasets_temporal(
    df: pd.DataFrame,
    train_years: Iterable[int],
    test_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, StatsState]:
    """
    Bygger features kronologisk. State oppdateres etter hver treningskamp.
    For test bygges features basert på state fra treningsperioden (ingen look-ahead),
    men state oppdateres fortløpende gjennom teståret for realistisk drift.
    Returnerer: (train_features_df, test_features_df, endelig_state)
    """
    state = StatsState()
    df_sorted = df.sort_values("tourney_date").reset_index(drop=True)

    train_df = df_sorted[df_sorted["year"].isin(train_years)].copy()
    test_df = df_sorted[df_sorted["year"] == test_year].copy()

    # ----- Treningsfeatures -----
    train_feats: List[Dict[str, float]] = []
    for _, row in train_df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        surf = row.get("surface", "unknown")
        tl = row.get("tourney_level", "A")

        feat = build_feature_row(
            state,
            w, l,
            surface=surf,
            tourney_level=tl,
            winner_rank=row.get("winner_rank", 1000),
            loser_rank=row.get("loser_rank", 1000),
            winner_rank_points=row.get("winner_rank_points", 0.0),
            loser_rank_points=row.get("loser_rank_points", 0.0),
            winner_age=row.get("winner_age", 25.0),
            loser_age=row.get("loser_age", 25.0),
            winner_ht=row.get("winner_ht", 180.0),
            loser_ht=row.get("loser_ht", 180.0),
        )
        feat["Target"] = 1
        train_feats.append(feat)

        update_state_after_match(state, w, l, surface=surf)

    # ----- Testfeatures -----
    test_feats: List[Dict[str, float]] = []
    for _, row in test_df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        surf = row.get("surface", "unknown")
        tl = row.get("tourney_level", "A")

        feat = build_feature_row(
            state,
            w, l,
            surface=surf,
            tourney_level=tl,
            winner_rank=row.get("winner_rank", 1000),
            loser_rank=row.get("loser_rank", 1000),
            winner_rank_points=row.get("winner_rank_points", 0.0),
            loser_rank_points=row.get("loser_rank_points", 0.0),
            winner_age=row.get("winner_age", 25.0),
            loser_age=row.get("loser_age", 25.0),
            winner_ht=row.get("winner_ht", 180.0),
            loser_ht=row.get("loser_ht", 180.0),
        )
        feat["Target"] = 1
        test_feats.append(feat)

        # Oppdater state også i teståret for realistisk tidsdrift
        update_state_after_match(state, w, l, surface=surf)

    train_features = pd.DataFrame(train_feats)
    test_features = pd.DataFrame(test_feats)

    # Behold kolonnerekkefølge
    feature_cols = state.feature_order
    train_features = train_features[[*feature_cols, "Target"]]
    test_features = test_features[[*feature_cols, "Target"]]

    return train_features, test_features, state
