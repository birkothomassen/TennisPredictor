# tournament_sim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd

from predict import MatchSimulator


ROUND_LABELS_MAP = {
    2:  ["Final"],
    4:  ["Semi-final", "Final"],
    8:  ["Quarter-final", "Semi-final", "Final"],
    16: ["R16", "Quarter-final", "Semi-final", "Final"],
    32: ["R32", "R16", "Quarter-final", "Semi-final", "Final"],
    64: ["R64", "R32", "R16", "Quarter-final", "Semi-final", "Final"],
    128:["R128","R64","R32","R16","Quarter-final","Semi-final","Final"],
}


def _is_power_of_two(n: int) -> bool:
    return n > 1 and (n & (n-1) == 0)


def _round_labels(n_players: int) -> List[str]:
    if n_players not in ROUND_LABELS_MAP:
        # generisk fallback
        labels = []
        m = n_players
        while m >= 2:
            if m == 2:
                labels.append("Final")
            elif m == 4:
                labels.append("Semi-final")
            elif m == 8:
                labels.append("Quarter-final")
            else:
                labels.append(f"R{m}")
            m //= 2
        return labels
    return ROUND_LABELS_MAP[n_players]


@dataclass
class SimConfig:
    surface: str = "Hard"
    tourney_level: str = "A"
    n_sims: int = 2000
    seed: int = 42


def simulate_tournament(
    sim: MatchSimulator,
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    cfg: SimConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Kjør Monte Carlo for KO-turnering.
    Returnerer:
      - summary_df: per-spiller sannsynlighet for Champion/Final/SF/…
      - raw_counts: rå teller for antall ganger spiller nådde hver runde
    CSV for draw kan bare være to kolonner: slot,player (slot=1..N, power-of-two N).
    """

    n = len(entrants)
    assert _is_power_of_two(n), f"Antall spillere må være 2^k (fikk {n})."
    labels = _round_labels(n)

    rng = np.random.default_rng(cfg.seed)

    # tellere
    reach_counts: Dict[str, Dict[str, int]] = {}
    champion_counts: Dict[str, int] = {}

    def _inc_reach(label: str, p: str):
        if p not in reach_counts:
            reach_counts[p] = {}
        reach_counts[p][label] = reach_counts[p].get(label, 0) + 1

    def _inc_champion(p: str):
        champion_counts[p] = champion_counts.get(p, 0) + 1

    # Hent stats med fallbacks
    def _stats(p: str) -> Dict[str, float]:
        s = stats_lookup.get(p, {})
        return {
            "rank": float(s.get("rank", 100.0)),
            "points": float(s.get("points", 0.0)),
            "age": float(s.get("age", 25.0)),
            "ht": float(s.get("ht", 185.0)),
        }

    # Simuler n_sims turneringer
    for _ in range(cfg.n_sims):
        bracket = list(entrants)

        for label in labels:
            winners = []
            # Alle i denne runden "reacher" denne runden
            for i in range(0, len(bracket), 2):
                A = bracket[i]
                B = bracket[i+1]
                _inc_reach(label, A)
                _inc_reach(label, B)

                As = _stats(A)
                Bs = _stats(B)

                pA, _ = sim.predict_proba(
                    A, B,
                    surface=cfg.surface, tourney_level=cfg.tourney_level,
                    A_rank=As["rank"], B_rank=Bs["rank"],
                    A_points=As["points"], B_points=Bs["points"],
                    A_age=As["age"], B_age=Bs["age"],
                    A_ht=As["ht"], B_ht=Bs["ht"],
                )
                # Sample vinner
                winA = rng.random() < pA
                winners.append(A if winA else B)

            bracket = winners  # neste runde

        # Champion
        assert len(bracket) == 1
        _inc_champion(bracket[0])

    # Bygg DataFrame med sannsynligheter
    players = sorted({*list(reach_counts.keys()), *list(champion_counts.keys())})
    cols = [*labels, "Champion"]
    data = []
    for p in players:
        row = {"Player": p}
        for lbl in labels:
            row[lbl] = reach_counts.get(p, {}).get(lbl, 0) / cfg.n_sims
        row["Champion"] = champion_counts.get(p, 0) / cfg.n_sims
        data.append(row)

    df = pd.DataFrame(data)
    # sorter etter tittel-sjanse
    df = df.sort_values("Champion", ascending=False).reset_index(drop=True)
    return df, {"reach": reach_counts, "champion": champion_counts}


def parse_draw_csv(file_like) -> List[str]:
    """
    Forventet format:
        slot,player
        1, Novak Djokovic
        2, Carlos Alcaraz
        ...
        N, Spiller N
    Returnerer liste med spillere i slot-rekkefølge.
    """
    df = pd.read_csv(file_like)
    # fleks: tillat 'player' eller 'Player'
    col = "player" if "player" in df.columns else ("Player" if "Player" in df.columns else None)
    if col is None:
        raise ValueError("CSV må ha kolonnen 'player' (eller 'Player').")
    # valgfritt: slot for sortering
    if "slot" in df.columns:
        df = df.sort_values("slot")
    entrants = [str(x).strip() for x in df[col].dropna().tolist()]
    if not _is_power_of_two(len(entrants)):
        raise ValueError(f"Antall spillere må være 2^k (fikk {len(entrants)}).")
    return entrants
