# tournament_sim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import pandas as pd
import time

from predict import MatchSimulator

@dataclass
class SimConfig:
    surface: str = "Hard"
    tourney_level: str = "A"
    n_sims: int = 2000  # For bakoverkompatibilitet
    seed: Optional[int] = None  # For bakoverkompatibilitet

def _seed_positions(n: int) -> List[int]:
    """
    Klassisk seed-plassering for 2/4/8 spillere:
    - 2: [1 vs 2]
    - 4: [1 vs 4] og [2 vs 3]  => pos [0,3,1,2]
    - 8: [1 vs 8], [4 vs 5], [3 vs 6], [2 vs 7] => pos [0,7,3,4,2,5,1,6]
    """
    if n == 2:
        return [0, 1]
    elif n == 4:
        return [0, 3, 1, 2]
    elif n == 8:
        return [0, 7, 3, 4, 2, 5, 1, 6]
    else:
        # For andre størrelser, bare returner normal rekkefølge
        return list(range(n))

def get_round_name(n_players: int, round_num: int, total_rounds: int) -> str:
    """Gir intuitive navn til rundene basert på antall spillere"""
    if n_players == 2:
        return "Final"
    elif n_players == 4:
        if round_num == 0:
            return "Semi-final"
        else:
            return "Final"
    elif n_players == 8:
        if round_num == 0:
            return "Quarter-final"
        elif round_num == 1:
            return "Semi-final"
        else:
            return "Final"
    else:
        # Fallback for andre størrelser
        if round_num == total_rounds - 1:
            return "Final"
        elif round_num == total_rounds - 2:
            return "Semi-final"
        else:
            return f"Round {round_num + 1}"

def round_labels(n_players: int) -> List[str]:
    """Returner rundeetiketter for gitt antall spillere"""
    if n_players <= 2:
        return ["Final"]
    elif n_players <= 4:
        return ["Semi-final", "Final"]
    elif n_players <= 8:
        return ["Quarter-final", "Semi-final", "Final"]
    else:
        # For større turneringer
        rounds = []
        temp = n_players
        while temp > 2:
            if temp <= 4:
                rounds.append("Semi-final")
            elif temp <= 8:
                rounds.append("Quarter-final")
            else:
                rounds.append(f"Round of {temp}")
            temp //= 2
        rounds.append("Final")
        return rounds

def build_bracket(
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    draw: Literal["as_is", "seeded", "shuffle"] = "as_is",
) -> List[str]:
    """
    Bygg start-bracket (rekkefølgen nederst i treet).
    """
    n = len(entrants)
    base = list(entrants)

    if draw == "seeded":
        # Sorter spillere etter ranking (lavere rank = bedre)
        rows = []
        for p in base:
            s = stats_lookup.get(p, {})
            rows.append({
                "player": p,
                "points": float(s.get("points", 0.0)),
                "rank": float(s.get("rank", 999.0)),
            })
        df = pd.DataFrame(rows).sort_values(["rank", "points"], ascending=[True, False]).reset_index(drop=True)
        seeded = df["player"].tolist()
        
        # Plasser i klassisk seeding-posisjon
        positions = _seed_positions(n)
        bracket = [None] * n
        for idx, pos in enumerate(positions):
            if idx < len(seeded) and pos < len(bracket):
                bracket[pos] = seeded[idx]
        
        # Fyll eventuelle manglende plasser
        for i in range(n):
            if bracket[i] is None:
                bracket[i] = seeded[min(i, len(seeded)-1)]
        return bracket

    elif draw == "shuffle":
        # Bruk tiden som seed for å få forskjellige resultater hver gang
        rng = np.random.default_rng(int(time.time() * 1000000) % 2**32)
        rng.shuffle(base)
        return base

    # as_is - bruk som gitt
    return base

def get_top_players(stats_lookup: Dict[str, Dict[str, float]], n: int = 8) -> List[str]:
    """Hent de n beste spillerne basert på ranking"""
    if not stats_lookup:
        return []
    
    players = []
    for name, stats in stats_lookup.items():
        rank = float(stats.get("rank", 999))
        points = float(stats.get("points", 0))
        players.append({"name": name, "rank": rank, "points": points})
    
    # Sorter etter rank (lavere = bedre), så points (høyere = bedre)
    players.sort(key=lambda x: (x["rank"], -x["points"]))
    
    return [p["name"] for p in players[:n]]

def top_n_players(stats_lookup: Dict[str, Dict[str, float]], n: int = 8) -> List[str]:
    """Alias for get_top_players for bakoverkompatibilitet"""
    return get_top_players(stats_lookup, n)

def simulate_tournament(
    sim: MatchSimulator,
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    cfg: SimConfig,
    draw: Literal["as_is", "seeded", "shuffle"] = "as_is",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Simuler mange turneringer og returner sannsynligheter.
    Returnerer samme format som før for bakoverkompatibilitet.
    """
    n = len(entrants)
    if n < 2:
        raise ValueError("Må ha minst 2 spillere")
    
    labels = round_labels(n)
    reach_counts: Dict[str, Dict[str, int]] = {}
    champion_counts: Dict[str, int] = {}
    
    def _inc_reach(label: str, p: str):
        reach_counts.setdefault(p, {})
        reach_counts[p][label] = reach_counts[p].get(label, 0) + 1
    
    def _inc_champion(p: str):
        champion_counts[p] = champion_counts.get(p, 0) + 1
    
    def get_player_stats(player: str) -> Dict[str, float]:
        s = stats_lookup.get(player, {})
        return {
            "rank": float(s.get("rank", 100.0)),
            "points": float(s.get("points", 1000.0)),
            "age": float(s.get("age", 25.0)),
            "ht": float(s.get("ht", 185.0)),
        }
    
    # Kjør mange simuleringer
    for sim_num in range(cfg.n_sims):
        # Bruk forskjellig seed for hver simulering
        rng = np.random.default_rng(int(time.time() * 1000000 + sim_num) % 2**32)
        
        current_players = build_bracket(entrants, stats_lookup, draw=draw)
        
        # Spill turnering runde for runde
        label_idx = 0
        while len(current_players) > 1 and label_idx < len(labels):
            label = labels[label_idx]
            next_round_players = []
            
            # Alle spillere i denne runden når den
            for player in current_players:
                _inc_reach(label, player)
            
            # Spill kampene
            for i in range(0, len(current_players), 2):
                if i + 1 < len(current_players):
                    player1 = current_players[i]
                    player2 = current_players[i + 1]
                    
                    stats1 = get_player_stats(player1)
                    stats2 = get_player_stats(player2)
                    
                    prob1, _ = sim.predict_proba(
                        player1, player2,
                        surface=cfg.surface, 
                        tourney_level=cfg.tourney_level,
                        A_rank=stats1["rank"], B_rank=stats2["rank"],
                        A_points=stats1["points"], B_points=stats2["points"],
                        A_age=stats1["age"], B_age=stats2["age"],
                        A_ht=stats1["ht"], B_ht=stats2["ht"],
                    )
                    
                    winner = player1 if rng.random() < prob1 else player2
                    next_round_players.append(winner)
                else:
                    # Bye
                    next_round_players.append(current_players[i])
            
            current_players = next_round_players
            label_idx += 1
        
        # Vinneren
        if current_players:
            _inc_champion(current_players[0])
    
    # Bygg resultat DataFrame
    players = sorted({*reach_counts.keys(), *champion_counts.keys()})
    data = []
    for p in players:
        row = {"Player": p}
        for lbl in labels:
            row[lbl] = reach_counts.get(p, {}).get(lbl, 0) / cfg.n_sims
        row["Champion"] = champion_counts.get(p, 0) / cfg.n_sims
        data.append(row)
    
    df = pd.DataFrame(data).sort_values("Champion", ascending=False).reset_index(drop=True)
    return df, {"reach": reach_counts, "champion": champion_counts}

def simulate_bracket_path(
    sim: MatchSimulator,
    entrants: List[str],
    stats_lookup: Dict[str, Dict[str, float]],
    cfg: SimConfig,
    mode: Literal["most_likely", "sample"] = "sample",
    draw: Literal["as_is", "seeded", "shuffle"] = "as_is",
) -> Dict[str, object]:
    """
    Simuler EN enkelt turnering og returner hele bracket-strukturen.
    """
    n = len(entrants)
    if n < 2:
        raise ValueError("Må ha minst 2 spillere")
    
    # Bruk tiden som seed for å få forskjellige resultater hver gang
    rng = np.random.default_rng(int(time.time() * 1000000) % 2**32)
    
    def get_player_stats(player: str) -> Dict[str, float]:
        s = stats_lookup.get(player, {})
        return {
            "rank": float(s.get("rank", 100.0)),
            "points": float(s.get("points", 1000.0)),
            "age": float(s.get("age", 25.0)),
            "ht": float(s.get("ht", 185.0)),
        }
    
    rounds_data = []
    current_players = build_bracket(entrants, stats_lookup, draw=draw)
    
    # Beregn antall runder
    total_rounds = 0
    temp_n = n
    while temp_n > 1:
        total_rounds += 1
        temp_n //= 2
    
    round_num = 0
    
    # Spill turnering runde for runde
    while len(current_players) > 1:
        round_name = get_round_name(n, round_num, total_rounds)
        matches = []
        next_round_players = []
        
        # Spill alle kamper i denne runden
        for i in range(0, len(current_players), 2):
            if i + 1 < len(current_players):
                player1 = current_players[i]
                player2 = current_players[i + 1]
                
                stats1 = get_player_stats(player1)
                stats2 = get_player_stats(player2)
                
                prob1, prob2 = sim.predict_proba(
                    player1, player2,
                    surface=cfg.surface, 
                    tourney_level=cfg.tourney_level,
                    A_rank=stats1["rank"], B_rank=stats2["rank"],
                    A_points=stats1["points"], B_points=stats2["points"],
                    A_age=stats1["age"], B_age=stats2["age"],
                    A_ht=stats1["ht"], B_ht=stats2["ht"],
                )
                
                # Simuler kampen
                if mode == "sample":
                    winner = player1 if rng.random() < prob1 else player2
                else:  # most_likely
                    winner = player1 if prob1 >= 0.5 else player2
                
                next_round_players.append(winner)
                
                matches.append({
                    "A": player1,
                    "B": player2,
                    "pA": float(prob1),
                    "winner": winner
                })
            else:
                # Bye
                next_round_players.append(current_players[i])
        
        rounds_data.append({
            "label": round_name,
            "matches": matches
        })
        
        current_players = next_round_players
        round_num += 1
    
    return {
        "rounds": rounds_data,
        "champion": current_players[0] if current_players else None
    }

def format_match_result(match: Dict) -> str:
    """Formater kampresultat for visning"""
    p1, p2 = match["A"], match["B"]
    winner = match["winner"]
    prob = match["pA"] if winner == p1 else (1 - match["pA"])
    
    return f"{p1} vs {p2} → {winner} ({prob:.1%})"

def get_tournament_summary(result: Dict) -> str:
    """Lag et kort sammendrag av turneringen"""
    champion = result["champion"]
    n_rounds = len(result["rounds"])
    
    summary = f"🏆 Champion: {champion}\n"
    summary += f"📊 Rounds: {n_rounds}\n"
    
    return summary