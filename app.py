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
from tournament_sim import (
    SimConfig,
    simulate_tournament,
    simulate_bracket_path,
    round_labels,
)

# =============== Konfig ===============
st.set_page_config(page_title="Tennis Prediction AI", page_icon=None, layout="wide")

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
    """Siste KJENTE (ikke-NaN) rank/points/age/ht per spiller + dato for points."""
    stats: Dict[str, Dict[str, float]] = {}
    df_sorted = df.sort_values("tourney_date")

    def _update(p: str, rank, pts, age, ht, date):
        cur = stats.get(p, {})
        if pd.notna(rank): cur["rank"] = float(rank)
        if pd.notna(pts):
            cur["points"] = float(pts)
            if pd.notna(date): cur["points_date"] = pd.to_datetime(date).date()
        if pd.notna(age): cur["age"] = float(age)
        if pd.notna(ht): cur["ht"] = float(ht)
        stats[p] = cur

    for _, row in df_sorted.iterrows():
        d = row.get("tourney_date")
        w, l = row.get("winner_name"), row.get("loser_name")
        if isinstance(w, str):
            _update(w, row.get("winner_rank"), row.get("winner_rank_points"),
                    row.get("winner_age"), row.get("winner_ht"), d)
        if isinstance(l, str):
            _update(l, row.get("loser_rank"), row.get("loser_rank_points"),
                    row.get("loser_age"), row.get("loser_ht"), d)

    for _, v in stats.items():
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

# =============== Header ===============
st.title("Tennis Prediction")

if not MODEL_PATH.exists() or not STATE_PATH.exists():
    st.error("Modell mangler. Kjør først:\n\n```bash\npython train.py\n```")
    st.stop()

st.divider()

# =============== Data & simulator ===============
df = _load_df()
players = _player_list(df)
stats_lookup = _latest_player_stats(df)
sim = _load_simulator()

# Tabs
tab_sim, tab_tour = st.tabs(
    ["Matchup-simulator", "Turnering-simulator"]
)

# ====================== SIMULATOR ======================
with tab_sim:
    st.subheader("Velg spillere og kontekst")

    def _safe_index(name: str, default: int = 0) -> int:
        try: return players.index(name)
        except Exception: return default

    col_a, col_b = st.columns(2)
    with col_a:
        A = st.selectbox("Spiller A", players, index=_safe_index("Novak Djokovic", 0), key="sim_player_a")
    with col_b:
        B = st.selectbox("Spiller B", players, index=_safe_index("Carlos Alcaraz", 1), key="sim_player_b")

    col_c, col_d = st.columns(2)
    with col_c:
        surface = st.radio("Underlag", ["Hard", "Clay", "Grass", "unknown"],
                           index=0, horizontal=True, key="sim_surface")
    with col_d:
        tl = st.selectbox("Turneringsnivå", ["G", "M", "A", "C", "F"], index=2, key="sim_tl")

    go = st.button("Simuler kamp", type="primary", key="sim_go")

    if go and sim:
        if A == B:
            st.warning("Velg to forskjellige spillere.")
            st.stop()

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
            st.progress(min(max(int(round(p * 100)), 0), 100))
        with c2:
            st.metric("Forventet vinner", A if p >= 0.5 else B)
        with c3:
            st.metric("Modell", "Aktiv")

        st.subheader("Brukte features")
        label_map = {"winner_rank": "A_rank", "loser_rank": "B_rank", "winner_pts": "A_points", "loser_pts": "B_points"}
        pretty_feat = {label_map.get(k, k): v for k, v in feat.items()}
        feat_df = pd.DataFrame([pretty_feat]).T.reset_index()
        feat_df.columns = ["feature", "value"]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ====================== TURNERING ======================
with tab_tour:
    st.subheader("Bygg & kjør turnering")

    size = st.selectbox("Antall spillere", [4, 8, 16, 32, 64], index=2, key="tour_size")
    labels = round_labels(size)

    st.caption("Fyll inn alle deltakerne (én per slot). Du kan auto-fylle med topp N fra datasettet.")

    def _top_n_players(n: int) -> List[str]:
        rows = [{"player": name, "points": s.get("points", 0.0), "rank": s.get("rank", 9999)}
                for name, s in stats_lookup.items()]
        sdf = pd.DataFrame(rows)
        if len(sdf) == 0: return []
        sdf = sdf.sort_values(["points", "rank"], ascending=[False, True])
        return sdf.head(n)["player"].tolist()

    cb1, cb2 = st.columns([1, 1])
    with cb1:
        if st.button(f"Fyll med topp {size}", key="btn_prefill"):
            top = _top_n_players(size)
            for i in range(size):
                st.session_state[f"slot_{size}_{i}"] = top[i] if i < len(top) else ""
    with cb2:
        if st.button("Tøm alle", key="btn_clear"):
            for i in range(size):
                st.session_state[f"slot_{size}_{i}"] = ""

    st.markdown("### R1 – sett inn spillere")
    cols = st.columns(4 if size >= 16 else 2)
    inputs: List[str] = []
    for i in range(size):
        col = cols[(i // 2) % len(cols)]
        with col:
            txt = st.text_input(f"Slot {i+1}", key=f"slot_{size}_{i}", placeholder="Spillernavn")
            inputs.append(txt.strip())

    if st.button("Valider deltakere", key="btn_validate"):
        if any(not x for x in inputs):
            st.error("Alle slots må fylles ut.")
        elif len(set(inputs)) != len(inputs):
            st.error("Dupliserte navn funnet. Hver spiller må være unik.")
        else:
            st.success("Alt ser bra ut! Du kan simulere under.")

    st.markdown("---")
    st.markdown("### Simuler")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        surface_t = st.selectbox("Underlag", ["Hard", "Clay", "Grass", "unknown"], index=0, key="tour_surface")
    with c2:
        tl_t = st.selectbox("Turneringsnivå", ["G", "M", "A", "C", "F"], index=2, key="tour_tl")
    with c3:
        mode = st.radio("Metode", ["Mest sannsynlig", "Én tilfeldig simulering"], index=0, key="tour_mode")
    with c4:
        seed = st.number_input("Seed", min_value=0, max_value=10**9, value=42, step=1, key="tour_seed")

    run = st.button("Kjør turnering", key="tour_run")

    if run:
        entrants = inputs
        if any(not x for x in entrants):
            st.error("Alle slots må fylles ut."); st.stop()
        if (len(entrants) & (len(entrants) - 1)) != 0 or len(entrants) != size:
            st.error(f"Antall spillere må være {size} (2^k)."); st.stop()
        if sim is None:
            st.error("Modell mangler – tren først (`python train.py`)."); st.stop()

        cfg = SimConfig(surface=surface_t, tourney_level=tl_t, seed=int(seed))

        path = simulate_bracket_path(
            sim, entrants, stats_lookup, cfg,
            mode="sample" if mode.startswith("Én") else "most_likely",
        )

        st.success(f"Ferdig! Vinner: {path['champion']}")

        st.markdown("### Runde for runde")
        round_cols = st.columns(len(path["rounds"]))
        for ci, rnd in enumerate(path["rounds"]):
            with round_cols[ci]:
                st.markdown(f"**{rnd['label']}**")
                for m in rnd["matches"]:
                    A, B, pA, W = m["A"], m["B"], m["pA"], m["winner"]
                    a_line = f"✔️ {A}  (p={pA:.2f})" if W == A else f"{A}  (p={pA:.2f})"
                    b_line = f"✔️ {B}  (p={1-pA:.2f})" if W == B else f"{B}  (p={1-pA:.2f})"
                    st.markdown(a_line); st.markdown(b_line); st.markdown("—")

        with st.expander("Monte Carlo-sammendrag (valgfritt)"):

            sims = st.slider("Antall simuleringer", 500, 10000, 2000, 500, key="tour_mc_sims")
            if st.button("Kjør MC", key="tour_mc_go"):
                cfg2 = SimConfig(surface=surface_t, tourney_level=tl_t, n_sims=int(sims), seed=int(seed))
                summary_df, _ = simulate_tournament(sim, entrants, stats_lookup, cfg2)
                show = summary_df.copy()
                show["Champion"] = (show["Champion"] * 100).round(1)
                st.dataframe(show, use_container_width=True, hide_index=True)
