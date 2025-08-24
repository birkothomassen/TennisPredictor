# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data import load_matches, list_players
from predict import MatchSimulator


# =============== Konfig ===============
st.set_page_config(page_title="Tennis Prediction AI", page_icon="🎾", layout="wide")

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.pkl"
STATE_PATH = MODELS_DIR / "stats_state.pkl"
META_PATH = MODELS_DIR / "metadata.json"


# =============== Cache-funksjoner ===============
@st.cache_data(show_spinner=False)
def _load_df(data_dir: str = "data/raw/tennis_atp") -> pd.DataFrame:
    return load_matches(data_dir)

@st.cache_data(show_spinner=False)
def _player_list(df: pd.DataFrame) -> List[str]:
    return list_players(df)

@st.cache_data(show_spinner=False)
def _latest_player_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Siste KJENTE (ikke-NaN) rank / rank_points / age / ht per spiller.
    Overskriver ikke med manglende verdier, og lagrer datoen poengene er fra.
    """
    stats: Dict[str, Dict[str, float]] = {}
    df_sorted = df.sort_values("tourney_date")  # eldste -> nyeste

    def _update(p: str, rank, pts, age, ht, date):
        cur = stats.get(p, {})
        if pd.notna(rank):
            cur["rank"] = float(rank)
        if pd.notna(pts):
            cur["points"] = float(pts)
            if pd.notna(date):
                cur["points_date"] = pd.to_datetime(date).date()
        if pd.notna(age):
            cur["age"] = float(age)
        if pd.notna(ht):
            cur["ht"] = float(ht)
        stats[p] = cur

    for _, row in df_sorted.iterrows():
        d = row.get("tourney_date")
        w = row.get("winner_name")
        l = row.get("loser_name")
        if isinstance(w, str):
            _update(
                w,
                row.get("winner_rank"),
                row.get("winner_rank_points"),
                row.get("winner_age"),
                row.get("winner_ht"),
                d,
            )
        if isinstance(l, str):
            _update(
                l,
                row.get("loser_rank"),
                row.get("loser_rank_points"),
                row.get("loser_age"),
                row.get("loser_ht"),
                d,
            )

    # Fallbacks
    for p, v in stats.items():
        v.setdefault("rank", 100.0)
        v.setdefault("points", 0.0)
        v.setdefault("points_date", None)
        v.setdefault("age", 25.0)
        v.setdefault("ht", 185.0)

    return stats

@st.cache_resource(show_spinner=False)
def _load_simulator() -> MatchSimulator | None:
    if not MODEL_PATH.exists() or not STATE_PATH.exists():
        return None
    return MatchSimulator(model_path=MODEL_PATH, state_path=STATE_PATH)

@st.cache_data(show_spinner=False)
def _load_metadata(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _raw_tree_estimator(model):
    """
    Hent underliggende tre-estimator (RandomForest) fra en kalibrert modell.
    """
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            return model.calibrated_classifiers_[0].base_estimator
        if hasattr(model, "base_estimator"):
            return model.base_estimator
    except Exception:
        pass
    return model


# =============== Header ===============
st.title("🎾 Tennis Prediction AI — Matchup-simulator")

meta = _load_metadata(META_PATH)
if not MODEL_PATH.exists() or not STATE_PATH.exists():
    st.error("Modell mangler. Kjør først:\n\n```bash\npython train.py\n```")
    st.stop()

if meta:
    st.caption(
        f"**Modell:** {meta.get('best_model','RandomForest')} • "
        f"**Train-år:** {', '.join(map(str, meta.get('train_years', [])))} • "
        f"**Test-år:** {meta.get('test_year','?')}"
    )

st.divider()


# =============== Data & simulator ===============
df = _load_df()
players = _player_list(df)
stats_lookup = _latest_player_stats(df)
sim = _load_simulator()

# Tabs
tab_sim, tab_model = st.tabs(["🆚 Matchup-simulator", "📊 Modell & innsikt"])


# ====================== SIMULATOR ======================
with tab_sim:
    st.subheader("🆚 Velg spillere og kontekst")

    def _safe_index(name: str, default: int = 0) -> int:
        try:
            return players.index(name)
        except Exception:
            return default

    col_a, col_b = st.columns(2)
    with col_a:
        A = st.selectbox("Spiller A", options=players, index=_safe_index("Novak Djokovic", 0))
    with col_b:
        B = st.selectbox("Spiller B", options=players, index=_safe_index("Carlos Alcaraz", 1))

    col_c, col_d = st.columns(2)
    with col_c:
        surface = st.radio("Underlag", options=["Hard", "Clay", "Grass", "unknown"], index=0, horizontal=True)
    with col_d:
        tl = st.selectbox("Turneringsnivå", options=["G", "M", "A", "C", "F"], index=2)

    go = st.button("🔮 Simuler kamp", type="primary")

    if go and sim:
        if A == B:
            st.warning("Velg to forskjellige spillere.")
            st.stop()

        # Hent auto-verdier
        A_s = stats_lookup.get(A, {"rank": 100.0, "points": 0.0, "age": 25.0, "ht": 185.0, "points_date": None})
        B_s = stats_lookup.get(B, {"rank": 100.0, "points": 0.0, "age": 25.0, "ht": 185.0, "points_date": None})

        with st.spinner("Beregner sannsynlighet..."):
            p, feat = sim.predict_proba(
                A, B,
                surface=surface, tourney_level=tl,
                A_rank=A_s["rank"], B_rank=B_s["rank"],
                A_points=A_s["points"], B_points=B_s["points"],
                A_age=A_s["age"],  B_age=B_s["age"],
                A_ht=A_s["ht"],    B_ht=B_s["ht"],
            )

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.metric(f"Sannsynlighet for {A} å slå {B}", f"{p:.1%}")
            capA = f"{A} (rank={int(A_s['rank'])}, pts={int(A_s['points'])}" + (f" as of {A_s['points_date']}" if A_s.get("points_date") else "") + ")"
            capB = f"{B} (rank={int(B_s['rank'])}, pts={int(B_s['points'])}" + (f" as of {B_s['points_date']}" if B_s.get("points_date") else "") + ")"
            st.caption(f"{capA}  vs  {capB}  [{surface}/{tl}]")
            st.progress(min(max(int(round(p * 100)), 0), 100))
        with c2:
            st.metric("Forventet vinner", A if p >= 0.5 else B)
        with c3:
            if meta:
                st.metric("Modell", meta.get("best_model", "RF"))

        # Brukte features (penere labels for A/B)
        st.subheader("📎 Brukte features")
        label_map = {
            "winner_rank": "A_rank",
            "loser_rank": "B_rank",
            "winner_pts":  "A_points",
            "loser_pts":   "B_points",
        }
        pretty_feat = {label_map.get(k, k): v for k, v in feat.items()}
        feat_df = pd.DataFrame([pretty_feat]).T.reset_index()
        feat_df.columns = ["feature", "value"]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Viktigste parametre (per-prediksjon)
        st.subheader("🔥 Viktigste parametre (denne prediksjonen)")
        tree_model = _raw_tree_estimator(sim.model)
        shown = False
        try:
            import shap  # valgfritt
            explainer = shap.TreeExplainer(tree_model)  # bruk rå RF, ikke kalibrert wrapper
            X_row, _ = sim._row(
                A, B,
                surface=surface, tourney_level=tl,
                A_rank=A_s["rank"], B_rank=B_s["rank"],
                A_points=A_s["points"], B_points=B_s["points"],
                A_age=A_s["age"],  B_age=B_s["age"],
                A_ht=A_s["ht"],    B_ht=B_s["ht"],
            )
            sv = explainer.shap_values(X_row)
            if isinstance(sv, list):  # RF: liste per klasse; ta klasse 1
                sv = sv[1]
            vals = sv[0]
            order = (np.abs(vals)).argsort()[::-1][:7]
            names = [sim.feature_order[i] for i in order]
            contribs = [float(vals[i]) for i in order]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(range(len(names))[::-1], contribs[::-1])
            ax.set_yticks(range(len(names))[::-1])
            ax.set_yticklabels(names[::-1])
            ax.set_title("SHAP-bidrag (størst først)")
            ax.axvline(0, linewidth=1)
            st.pyplot(fig, use_container_width=True)
            shown = True
        except Exception:
            pass

        if not shown:
            # Fallback: global importance fra rå RF
            try:
                importances = getattr(tree_model, "feature_importances_", None)
                if importances is not None:
                    order = np.argsort(importances)[::-1][:7]
                    names = [sim.feature_order[i] for i in order]
                    vals = [float(importances[i]) for i in order]
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.barh(range(len(names))[::-1], vals[::-1])
                    ax.set_yticks(range(len(names))[::-1])
                    ax.set_yticklabels(names[::-1])
                    ax.set_title("Feature importance (RF)")
                    st.pyplot(fig, use_container_width=True)
            except Exception:
                st.caption("Kunne ikke vise viktighetsplot.")

        with st.expander("ℹ️ Datakilde for rank/points"):
            st.write(
                "- **rank** hentes fra kolonnene `winner_rank`/`loser_rank` i ATP-CSV.\n"
                "- **points** hentes fra `winner_rank_points`/`loser_rank_points`.\n"
                "- Verdiene er **siste kjente** fra datasettet (dato vist over)."
            )


# ====================== MODELL & INNSIKT ======================
with tab_model:
    st.subheader("📊 Modellresultater (fra metadata)")
    if meta is None:
        st.info("Ingen metadata funnet.")
    else:
        chosen = meta.get("chosen_scores", {})
        scores = meta.get("scores", {})
        if chosen:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{chosen.get('accuracy', 0):.3f}")
            c2.metric("Precision", f"{chosen.get('precision', 0):.3f}")
            c3.metric("LogLoss", f"{chosen.get('logloss', 0):.3f}")
            c4.metric("Brier", f"{chosen.get('brier', 0):.3f}")

        if scores:
            rows = []
            for name, sc in scores.items():
                rows.append({
                    "Model": name,
                    "Accuracy": sc.get("accuracy"),
                    "Precision": sc.get("precision"),
                    "LogLoss": sc.get("logloss"),
                    "Brier": sc.get("brier"),
                })
            score_df = pd.DataFrame(rows).sort_values("LogLoss")
            st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("🌍 Viktigste features (globalt)")
    if sim is None:
        st.info("Modell ikke lastet.")
    else:
        try:
            tree_model = _raw_tree_estimator(sim.model)
            importances = getattr(tree_model, "feature_importances_", None)
            if importances is not None:
                feat_imp = pd.DataFrame({
                    "feature": sim.feature_order,
                    "importance": importances
                }).sort_values("importance", ascending=True)

                # Kul, ren barplot (matplotlib)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(feat_imp["feature"], feat_imp["importance"])
                ax.set_title("Feature Importance — Random Forest", fontweight="bold")
                ax.set_xlabel("Viktighet")
                ax.set_ylabel("Feature")
                st.pyplot(fig, use_container_width=True)

                # Topp 3 som metrics
                top3 = feat_imp.sort_values("importance", ascending=False).head(3)
                c1, c2, c3 = st.columns(3)
                cols = [c1, c2, c3]
                for i, row in enumerate(top3.itertuples(index=False), start=0):
                    cols[i].metric(f"#{i+1} {row.feature}", f"{row.importance:.3f}")
        except Exception:
            st.caption("Kunne ikke beregne global feature importance.")

    with st.expander("Rå metadata"):
        st.json(meta or {})
