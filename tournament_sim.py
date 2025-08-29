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

def round_labels(n_players: int) -> List[str]:
    if n_players in ROUND_LABELS_MAP:
        return ROUND_LABELS_MAP[n_players]
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

@dataclass
class SimConfig:
    surface: str = "Hard"
    tourney_level: str = "A"
    n_sims: int = 2000
    seed: int = 42

# ------- Monte Carlo sammendrag (som før) -------
def simulate_tournament(
    sim: MatchSimulator,
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    cfg: SimConfig,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    n = len(entrants)
    assert _is_power_of_two(n), f"Antall spillere må være 2^k (fikk {n})."
    labels = round_labels(n)

    rng = np.random.default_rng(cfg.seed)
    reach_counts: Dict[str, Dict[str, int]] = {}
    champion_counts: Dict[str, int] = {}

    def _inc_reach(label: str, p: str):
        reach_counts.setdefault(p, {})
        reach_counts[p][label] = reach_counts[p].get(label, 0) + 1

    def _inc_champion(p: str):
        champion_counts[p] = champion_counts.get(p, 0) + 1

    def _stats(p: str) -> Dict[str, float]:
        s = stats_lookup.get(p, {})
        return {
            "rank": float(s.get("rank", 100.0)),
            "points": float(s.get("points", 0.0)),
            "age": float(s.get("age", 25.0)),
            "ht": float(s.get("ht", 185.0)),
        }

    for _ in range(cfg.n_sims):
        bracket = list(entrants)
        for label in labels:
            winners = []
            for i in range(0, len(bracket), 2):
                A, B = bracket[i], bracket[i+1]
                _inc_reach(label, A); _inc_reach(label, B)
                As, Bs = _stats(A), _stats(B)
                pA, _ = sim.predict_proba(
                    A, B,
                    surface=cfg.surface, tourney_level=cfg.tourney_level,
                    A_rank=As["rank"], B_rank=Bs["rank"],
                    A_points=As["points"], B_points=Bs["points"],
                    A_age=As["age"], B_age=Bs["age"],
                    A_ht=As["ht"],  B_ht=Bs["ht"],
                )
                winners.append(A if rng.random() < pA else B)
            bracket = winners
        champion_counts[bracket[0]] = champion_counts.get(bracket[0], 0) + 1

    players = sorted({*reach_counts.keys(), *champion_counts.keys()})
    cols = [*labels, "Champion"]
    data = []
    for p in players:
        row = {"Player": p}
        for lbl in labels:
            row[lbl] = reach_counts.get(p, {}).get(lbl, 0) / cfg.n_sims
        row["Champion"] = champion_counts.get(p, 0) / cfg.n_sims
        data.append(row)
    df = pd.DataFrame(data).sort_values("Champion", ascending=False).reset_index(drop=True)
    return df, {"reach": reach_counts, "champion": champion_counts}

# ------- EN enkelt bracket (for visuell progresjon) -------
def simulate_bracket_path(
    sim: MatchSimulator,
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    cfg: SimConfig,
    mode: str = "most_likely",  # "most_likely" eller "sample"
) -> Dict[str, object]:
    """
    Kjør én turnering og returner runde-for-runde resultater:
    {
      "rounds": [
         {"label": "R16", "matches":[{"A":..., "B":..., "pA":0.62, "winner":"..."}]},
         ...
      ],
      "champion": "..."
    }
    """
    assert _is_power_of_two(len(entrants)), "Antall spillere må være 2^k."
    labels = round_labels(len(entrants))
    rng = np.random.default_rng(cfg.seed)

    def _stats(p: str) -> Dict[str, float]:
        s = stats_lookup.get(p, {})
        return {
            "rank": float(s.get("rank", 100.0)),
            "points": float(s.get("points", 0.0)),
            "age": float(s.get("age", 25.0)),
            "ht": float(s.get("ht", 185.0)),
        }

    rounds: List[Dict[str, object]] = []
    bracket = list(entrants)

    for label in labels:
        matches: List[Dict[str, object]] = []
        winners = []
        for i in range(0, len(bracket), 2):
            A, B = bracket[i], bracket[i+1]
            As, Bs = _stats(A), _stats(B)
            pA, _ = sim.predict_proba(
                A, B,
                surface=cfg.surface, tourney_level=cfg.tourney_level,
                A_rank=As["rank"], B_rank=Bs["rank"],
                A_points=As["points"], B_points=Bs["points"],
                A_age=As["age"], B_age=Bs["age"],
                A_ht=As["ht"],  B_ht=Bs["ht"],
            )
            if mode == "sample":
                winner = A if rng.random() < pA else B
            else:
                winner = A if pA >= 0.5 else B
            winners.append(winner)
            matches.append({"A": A, "B": B, "pA": float(pA), "winner": winner})
        rounds.append({"label": label, "matches": matches})
        bracket = winners

    return {"rounds": rounds, "champion": bracket[0]}
