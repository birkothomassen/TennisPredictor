# predict.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from features import StatsState, build_feature_row

MODELS_DIR = Path("models")


class MatchSimulator:
    """
    Laster:
      - models/best_model.pkl  (kalibrert estimator med predict_proba)
      - models/stats_state.pkl (StatsState med h2h/elo/recent_form/feature_order)
    Bruk:
        sim = MatchSimulator()
        proba, feats = sim.predict_proba(
            "Novak Djokovic", "Carlos Alcaraz",
            surface="Hard", tourney_level="G",
            A_rank=1, B_rank=2, A_points=12000, B_points=10000
        )
    """
    def __init__(self,
                 model_path: str | Path = MODELS_DIR / "best_model.pkl",
                 state_path: str | Path = MODELS_DIR / "stats_state.pkl"):
        self.model = joblib.load(model_path)
        with open(state_path, "rb") as f:
            self.state: StatsState = pickle.load(f)

        # behold kolonnerekkefølgen som ble brukt ved trening
        self.feature_order = list(self.state.feature_order)

    # ---------- bygg features ----------
    def _row(
        self,
        A: str, B: str, *,
        surface: str = "Hard",
        tourney_level: str = "A",
        A_rank: float = 100.0, B_rank: float = 100.0,
        A_points: float = 0.0, B_points: float = 0.0,
        A_age: float = 25.0,  B_age: float = 25.0,
        A_ht: float = 185.0,  B_ht: float = 185.0,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        feat = build_feature_row(
            self.state,
            winner=A, loser=B,
            surface=surface, tourney_level=tourney_level,
            winner_rank=A_rank, loser_rank=B_rank,
            winner_rank_points=A_points, loser_rank_points=B_points,
            winner_age=A_age, loser_age=B_age,
            winner_ht=A_ht, loser_ht=B_ht,
        )
        X = pd.DataFrame([feat])[self.feature_order]
        return X, feat

    # ---------- prediksjon ----------
    def predict_proba(
        self,
        A: str, B: str, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returnerer sannsynlighet for at A slår B (klasse=1) + feature-dict.
        Støtter kwargs: surface, tourney_level, A_rank, B_rank, A_points, B_points, A_age, B_age, A_ht, B_ht
        """
        X, feat = self._row(A, B, **kwargs)
        proba = float(self.model.predict_proba(X)[:, 1])
        return proba, feat

    def predict(self, A: str, B: str, threshold: float = 0.5, **kwargs) -> Dict[str, object]:
        """
        Gir vinner basert på threshold + proba og features.
        """
        p, feat = self.predict_proba(A, B, **kwargs)
        winner = A if p >= threshold else B
        return {"winner": winner, "prob_A_wins": p, "features": feat}

    # ---------- helper ----------
    @staticmethod
    def format_matchup(A: str, B: str, surface: str, tl: str, p: float) -> str:
        return f"{A} vs {B} [{surface}/{tl}] → P({A} wins)={p:.1%}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("A")
    parser.add_argument("B")
    parser.add_argument("--surface", "-s", default="Hard")
    parser.add_argument("--tl", default="A", help="Turneringsnivå: G/M/A/C/F")
    parser.add_argument("--A_rank", type=float, default=100.0)
    parser.add_argument("--B_rank", type=float, default=100.0)
    parser.add_argument("--A_points", type=float, default=0.0)
    parser.add_argument("--B_points", type=float, default=0.0)
    parser.add_argument("--A_age", type=float, default=25.0)
    parser.add_argument("--B_age", type=float, default=25.0)
    parser.add_argument("--A_ht", type=float, default=185.0)
    parser.add_argument("--B_ht", type=float, default=185.0)
    args = parser.parse_args()

    sim = MatchSimulator()
    p, _ = sim.predict_proba(
        args.A, args.B,
        surface=args.surface, tourney_level=args.tl,
        A_rank=args.A_rank, B_rank=args.B_rank,
        A_points=args.A_points, B_points=args.B_points,
        A_age=args.A_age, B_age=args.B_age,
        A_ht=args.A_ht, B_ht=args.B_ht,
    )
    print(MatchSimulator.format_matchup(args.A, args.B, args.surface, args.tl, p))
