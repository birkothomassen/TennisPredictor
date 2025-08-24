# train.py
from __future__ import annotations

import json
import joblib
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, log_loss, brier_score_loss

from data import load_matches, train_test_years
from features import build_datasets_temporal, StatsState

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _mirror_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Lag speil-eksempler ved å bytte perspektiv A↔B korrekt."""
    flip = df.copy()

    # 1) Neger differanse-features
    for col in [
        "rank_diff",
        "points_diff",
        "age_diff",
        "ht_diff",
        "h2h_net",
        "surface_elo_diff",
        "recent_form_diff",
    ]:
        if col in flip.columns:
            flip[col] = -flip[col]

    # 2) Bytt winner_* ↔ loser_* for råverdier
    swap_pairs = [
        ("winner_rank", "loser_rank"),
        ("winner_pts",  "loser_pts"),
    ]
    for a, b in swap_pairs:
        if a in flip.columns and b in flip.columns:
            tmp = flip[a].copy()
            flip[a] = flip[b]
            flip[b] = tmp

    # 3) Samme turnerings-importance (uendret)
    # 4) Ny target
    flip["Target"] = 0

    out = pd.concat([df, flip], ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, proba)),
    }


def train_and_save(data_dir: str = "data/raw/tennis_atp") -> Tuple[str, str]:
    """
    Trener RF på 2018..(n-1), tester på år n.
    Lagrer:
      - models/best_model.pkl  (kalibrert RF)
      - models/stats_state.pkl (StatsState)
      - models/metadata.json   (info)
    """
    # 1) Data
    df = load_matches(data_dir)

    # 2) År-splitt
    train_years, test_year = train_test_years(df)
    print(f"Trener på år: {train_years} | Tester på: {test_year}")

    # 3) Features + state
    train_features, test_features, state = build_datasets_temporal(df, train_years, test_year)

    # 4) Balanser
    train_bal = _mirror_examples(train_features)
    test_bal = _mirror_examples(test_features)
    feature_cols = [c for c in train_bal.columns if c != "Target"]

    X_train, y_train = train_bal[feature_cols], train_bal["Target"].astype(int)
    X_test, y_test = test_bal[feature_cols], test_bal["Target"].astype(int)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("Feature columns:", feature_cols)

    # 5) Random Forest + kalibrering
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf, cv=3, method="isotonic")
    rf_cal.fit(X_train, y_train)

    # 6) Eval
    scores = _evaluate(rf_cal, X_test, y_test)
    print(f"[RandomForest] acc={scores['accuracy']:.3f}  prec={scores['precision']:.3f}  "
          f"logloss={scores['logloss']:.3f}  brier={scores['brier']:.3f}")

    # 7) Lagre
    model_path = MODELS_DIR / "best_model.pkl"
    state_path = MODELS_DIR / "stats_state.pkl"
    meta_path = MODELS_DIR / "metadata.json"

    # sikre feature-rekkefølge lagres i state
    if isinstance(state, StatsState) and not state.feature_order:
        state.feature_order = feature_cols

    joblib.dump(rf_cal, model_path)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    metadata = {
        "best_model": "RandomForest",
        "scores": {"RandomForest": scores},
        "chosen_scores": scores,
        "train_years": list(map(int, train_years)),
        "test_year": int(test_year),
        "feature_order": feature_cols,
        "has_xgboost": False,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Lagret beste modell: RandomForest")
    print(f"  - Model: {model_path}")
    print(f"  - State: {state_path}")
    print(f"  - Meta : {meta_path}")

    return str(model_path), str(state_path)


if __name__ == "__main__":
    train_and_save()
