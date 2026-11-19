"""Vectorized Common-Random-Numbers (CRN) Monte Carlo league & playoff simulator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .lineup import parse_starter_slots, solve_weekly_lineup

POS_VARIANCE_PARAMS: Dict[str, Tuple[float, float]] = {
    "QB": (0.34, 2.8),
    "RB": (0.44, 2.4),
    "WR": (0.52, 2.5),
    "TE": (0.55, 2.0),
    "K": (0.40, 2.0),
    "DEF": (0.40, 2.0),
}

ALL_POSITIONS: List[str] = ["QB", "RB", "WR", "TE", "K", "DEF"]


def _pos_sigma(pos: str, mu: float) -> float:
    slope, intercept = POS_VARIANCE_PARAMS.get(pos, (0.45, 2.4))
    return slope * mu + intercept


class SimulationContext:
    """Holds Common Random Numbers (CRN) player & streamer score arrays for fast trade evaluation."""

    def __init__(
        self,
        league_info: Dict[str, Any],
        players_master: Dict[str, Dict[str, Any]],
        weekly_projections: Dict[str, Dict[int, float]],
        streaming_baselines: Dict[str, Dict[int, float]],
        waiver_pool: List[Dict[str, Any]],
        num_sims: int = 5000,
        seed: int = 42,
        ros_totals: Optional[Dict[str, Dict[str, Any]]] = None,
        bye_week_credit: Optional[float] = None,
    ) -> None:
        self.num_sims = int(num_sims)
        self.seed = int(seed)
        self.current_week = int(league_info["current_week"])
        self.playoff_week_start = int(league_info.get("playoff_week_start") or 15)
        self.playoff_teams = int(league_info.get("playoff_teams") or 6)
        self.playoff_round_type = int(league_info.get("playoff_round_type") or 0)
        self.playoff_seed_type = int(league_info.get("playoff_seed_type") or 0)
        self.use_median_scoring = bool(league_info.get("use_median_scoring", True))
        self.starter_slots = parse_starter_slots(league_info.get("roster_positions") or [])

        if bye_week_credit is None:
            self.bye_week_credit = 0.85 if league_info.get("neutralize_byes", True) else 0.0
        else:
            self.bye_week_credit = float(bye_week_credit)

        # Determine simulation weeks (from current_week through Week 17)
        end_week = 17
        self.sim_weeks: List[int] = list(range(self.current_week, end_week + 1))
        if not self.sim_weeks:
            self.sim_weeks = [17]
        self.week_to_idx: Dict[int, int] = {w: idx for idx, w in enumerate(self.sim_weeks)}
        self.reg_weeks: List[int] = [w for w in self.sim_weeks if w < self.playoff_week_start]
        self.playoff_weeks: List[int] = [w for w in self.sim_weeks if w >= self.playoff_week_start]

        self.players_master = players_master
        self.raw_weekly_projections = weekly_projections

        # Apply bye-week credit so single regular-season byes don't artificially inflate bye-swap trades
        effective_proj: Dict[str, Dict[int, float]] = {}
        for pid, wk_map in weekly_projections.items():
            eff_map = dict(wk_map)
            if self.bye_week_credit > 0.0 and ros_totals:
                ros = ros_totals.get(pid) or {}
                bye_weeks = ros.get("bye_weeks") or []
                ppg = float(ros.get("ppg") or 0.0)
                if bye_weeks and ppg > 0.0:
                    imputed = round(self.bye_week_credit * ppg, 3)
                    for bw in bye_weeks:
                        if bw in self.week_to_idx:
                            eff_map[bw] = imputed
            effective_proj[pid] = eff_map

        self.effective_weekly_projections = effective_proj
        self.weekly_projections = self.effective_weekly_projections
        self.streaming_baselines = streaming_baselines

        # Collect all relevant player_ids (rostered + waiver pool + any projected players)
        relevant_pids: Set[str] = set()
        for t in league_info["teams"]:
            relevant_pids.update(t.get("players") or [])
        for w in waiver_pool:
            relevant_pids.add(w["player_id"])
        relevant_pids.update(self.weekly_projections.keys())

        self.player_ids: List[str] = sorted(relevant_pids)
        self.pid_to_idx: Dict[str, int] = {pid: idx for idx, pid in enumerate(self.player_ids)}

        num_players = len(self.player_ids)
        num_weeks = len(self.sim_weeks)

        self.mu = np.zeros((num_players, num_weeks), dtype=np.float32)
        self.sigma = np.zeros((num_players, num_weeks), dtype=np.float32)

        for p_idx, pid in enumerate(self.player_ids):
            pos = str((players_master.get(pid) or {}).get("position") or "WR")
            wk_map = self.weekly_projections.get(pid) or {}
            for w_idx, w in enumerate(self.sim_weeks):
                m = float(wk_map.get(w, 0.0))
                if m > 0.0:
                    self.mu[p_idx, w_idx] = m
                    self.sigma[p_idx, w_idx] = _pos_sigma(pos, m)

        rng = np.random.default_rng(self.seed)

        # Sample right-skewed Gamma distributions matching exact (mu, sigma^2)
        self.player_samples = np.zeros((self.num_sims, num_weeks, num_players), dtype=np.float32)
        active_mask = self.mu > 0.5
        if np.any(active_mask):
            m_act = self.mu[active_mask].astype(np.float64)
            s_act = self.sigma[active_mask].astype(np.float64)
            shapes = (m_act / s_act) ** 2
            scales = (s_act ** 2) / m_act
            # Broadcast shape/scale across num_sims
            sampled = rng.gamma(shape=shapes[None, :], scale=scales[None, :], size=(self.num_sims, m_act.size)).astype(np.float32)
            # Assign into (num_sims, num_weeks, num_players)
            p_indices, w_indices = np.where(active_mask)
            self.player_samples[:, w_indices, p_indices] = sampled

        # Include tiny deterministic points if 0 < mu <= 0.5
        tiny_mask = (self.mu > 0.0) & (self.mu <= 0.5)
        if np.any(tiny_mask):
            p_tiny, w_tiny = np.where(tiny_mask)
            self.player_samples[:, w_tiny, p_tiny] = self.mu[p_tiny, w_tiny][None, :]

        # Pre-sample bye-week streamer scores (90% of best unrostered player at pos)
        self.pos_to_idx: Dict[str, int] = {pos: idx for idx, pos in enumerate(ALL_POSITIONS)}
        self.streamer_mu = np.zeros((len(ALL_POSITIONS), num_weeks), dtype=np.float32)
        self.streamer_sigma = np.zeros((len(ALL_POSITIONS), num_weeks), dtype=np.float32)
        self.streamer_samples = np.zeros((self.num_sims, num_weeks, len(ALL_POSITIONS)), dtype=np.float32)

        for pos_idx, pos in enumerate(ALL_POSITIONS):
            for w_idx, w in enumerate(self.sim_weeks):
                m = float((streaming_baselines.get(pos) or {}).get(w, 0.0)) * 0.90
                if m > 0.5:
                    s = _pos_sigma(pos, m)
                    self.streamer_mu[pos_idx, w_idx] = m
                    self.streamer_sigma[pos_idx, w_idx] = s
                    k = (m / s) ** 2
                    theta = (s ** 2) / m
                    self.streamer_samples[:, w_idx, pos_idx] = rng.gamma(k, theta, size=self.num_sims).astype(np.float32)

        # Team indexing and baseline schedule mapping
        self.teams = league_info["teams"]
        self.num_teams = len(self.teams)
        self.roster_id_to_idx: Dict[int, int] = {
            int(t["roster_id"]): idx for idx, t in enumerate(self.teams)
        }
        self.idx_to_roster_id: Dict[int, int] = {
            idx: int(t["roster_id"]) for idx, t in enumerate(self.teams)
        }

        self.current_wins = np.array(
            [float(t["wins"]) + 0.5 * float(t.get("ties", 0)) for t in self.teams],
            dtype=np.float32,
        )
        self.current_pf = np.array(
            [float(t["points_for"]) for t in self.teams],
            dtype=np.float32,
        )

        # Pre-map remaining regular-season H2H schedule to team indices
        self.schedule_idx_by_week: Dict[int, List[Tuple[int, int]]] = {}
        raw_sched = league_info.get("schedule_by_week") or {}
        for w in self.reg_weeks:
            pairs = raw_sched.get(w) or []
            idx_pairs: List[Tuple[int, int]] = []
            for r1, r2 in pairs:
                if r1 in self.roster_id_to_idx and r2 in self.roster_id_to_idx:
                    idx_pairs.append((self.roster_id_to_idx[r1], self.roster_id_to_idx[r2]))
            self.schedule_idx_by_week[w] = idx_pairs


def compute_team_weekly_profile(
    roster_ids: List[str],
    reserve_ids: Set[str],
    sim_ctx: SimulationContext,
    include_lineups: bool = False,
) -> Dict[str, Any]:
    """Compute a team's optimal lineups, deterministic means/variances, and (num_sims, num_weeks) scores."""
    num_weeks = len(sim_ctx.sim_weeks)
    scores = np.zeros((sim_ctx.num_sims, num_weeks), dtype=np.float32)
    weekly_mu = np.zeros(num_weeks, dtype=np.float64)
    weekly_var = np.zeros(num_weeks, dtype=np.float64)
    lineups_by_week: Dict[int, Dict[str, Any]] = {}

    for w_idx, w in enumerate(sim_ctx.sim_weeks):
        total_pts, starters, bench = solve_weekly_lineup(
            roster_ids,
            w,
            sim_ctx.weekly_projections,
            sim_ctx.players_master,
            sim_ctx.streaming_baselines,
            sim_ctx.starter_slots,
            reserve_ids=reserve_ids,
        )
        p_indices: List[int] = []
        stream_indices: List[int] = []
        w_mu = 0.0
        w_var = 0.0

        for st in starters:
            pid = st["player_id"]
            if st["is_streamer"]:
                pos = st["position"]
                pos_idx = sim_ctx.pos_to_idx.get(pos, 2)
                stream_indices.append(pos_idx)
                m = float(sim_ctx.streamer_mu[pos_idx, w_idx])
                s = float(sim_ctx.streamer_sigma[pos_idx, w_idx])
                w_mu += m
                w_var += s * s
            else:
                p_idx = sim_ctx.pid_to_idx.get(pid)
                if p_idx is not None:
                    p_indices.append(p_idx)
                    m = float(sim_ctx.mu[p_idx, w_idx])
                    s = float(sim_ctx.sigma[p_idx, w_idx])
                    w_mu += m
                    w_var += s * s

        if p_indices:
            scores[:, w_idx] += sim_ctx.player_samples[:, w_idx, p_indices].sum(axis=1)
        if stream_indices:
            scores[:, w_idx] += sim_ctx.streamer_samples[:, w_idx, stream_indices].sum(axis=1)

        weekly_mu[w_idx] = w_mu if w_mu > 0 else total_pts
        weekly_var[w_idx] = max(w_var, 1.0)

        if include_lineups:
            lineups_by_week[w] = {
                "projected_pts": round(total_pts, 2),
                "std_dev": round(float(np.sqrt(weekly_var[w_idx])), 2),
                "starters": starters,
                "bench": bench,
            }

    return {
        "scores": scores,
        "weekly_mu": weekly_mu,
        "weekly_var": weekly_var,
        "lineups_by_week": lineups_by_week,
    }


def _simulate_playoffs(
    team_score_matrix: np.ndarray,
    order: np.ndarray,
    sim_ctx: SimulationContext,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 6-team, 4-team, or 8-team playoff brackets across remaining playoff weeks."""
    num_sims = sim_ctx.num_sims
    num_teams = sim_ctx.num_teams
    playoff_teams = min(sim_ctx.playoff_teams, num_teams)
    sim_idx = np.arange(num_sims)

    made_playoffs = np.zeros((num_sims, num_teams), dtype=bool)
    for k in range(playoff_teams):
        made_playoffs[sim_idx, order[:, k]] = True

    first_round_bye = np.zeros((num_sims, num_teams), dtype=bool)
    made_semis = np.zeros((num_sims, num_teams), dtype=bool)
    made_finals = np.zeros((num_sims, num_teams), dtype=bool)
    won_champ = np.zeros((num_sims, num_teams), dtype=bool)

    po_weeks = sim_ctx.playoff_weeks
    if not po_weeks:
        # Fallback if simulated after Week 17
        for k in range(min(4, playoff_teams)):
            made_semis[sim_idx, order[:, k]] = True
        champ = order[:, 0]
        made_finals[sim_idx, order[:, 0]] = True
        if num_teams > 1:
            made_finals[sim_idx, order[:, 1]] = True
        won_champ[sim_idx, champ] = True
        return made_playoffs, first_round_bye, made_semis, made_finals, won_champ

    # Helper to extract simulated scores for a playoff week (or fallback to last available week)
    def get_scores(wk_offset: int) -> np.ndarray:
        wk = po_weeks[min(wk_offset, len(po_weeks) - 1)]
        w_idx = sim_ctx.week_to_idx[wk]
        return team_score_matrix[:, w_idx, :]

    if playoff_teams == 6 and len(po_weeks) >= 3:
        s1, s2, s3, s4, s5, s6 = [order[:, k] for k in range(6)]
        first_round_bye[sim_idx, s1] = True
        first_round_bye[sim_idx, s2] = True

        # Week 15 Quarterfinals: 3v6 and 4v5
        sc_qf = get_scores(0)
        qf1_win_s3 = sc_qf[sim_idx, s3] >= sc_qf[sim_idx, s6]
        qf1_win = np.where(qf1_win_s3, s3, s6)
        qf1_seed = np.where(qf1_win_s3, 3, 6)

        qf2_win_s4 = sc_qf[sim_idx, s4] >= sc_qf[sim_idx, s5]
        qf2_win = np.where(qf2_win_s4, s4, s5)
        qf2_seed = np.where(qf2_win_s4, 4, 5)

        made_semis[sim_idx, s1] = True
        made_semis[sim_idx, s2] = True
        made_semis[sim_idx, qf1_win] = True
        made_semis[sim_idx, qf2_win] = True

        # Week 16 Semifinals
        if sim_ctx.playoff_seed_type == 1:
            # Re-seed: #1 plays lowest remaining seed (highest seed number)
            opp1 = np.where(qf1_seed >= qf2_seed, qf1_win, qf2_win)
            opp2 = np.where(qf1_seed >= qf2_seed, qf2_win, qf1_win)
        else:
            # Fixed bracket: #1 plays winner of 4v5, #2 plays winner of 3v6
            opp1 = qf2_win
            opp2 = qf1_win

        sc_sf = get_scores(1)
        sf1_win = np.where(sc_sf[sim_idx, s1] >= sc_sf[sim_idx, opp1], s1, opp1)
        sf2_win = np.where(sc_sf[sim_idx, s2] >= sc_sf[sim_idx, opp2], s2, opp2)

        made_finals[sim_idx, sf1_win] = True
        made_finals[sim_idx, sf2_win] = True

        # Week 17 Championship
        sc_ch = get_scores(2)
        champ = np.where(sc_ch[sim_idx, sf1_win] >= sc_ch[sim_idx, sf2_win], sf1_win, sf2_win)
        won_champ[sim_idx, champ] = True

    elif playoff_teams == 4 and len(po_weeks) >= 2:
        s1, s2, s3, s4 = [order[:, k] for k in range(4)]
        for s in (s1, s2, s3, s4):
            made_semis[sim_idx, s] = True
        sc_sf = get_scores(0)
        sf1_win = np.where(sc_sf[sim_idx, s1] >= sc_sf[sim_idx, s4], s1, s4)
        sf2_win = np.where(sc_sf[sim_idx, s2] >= sc_sf[sim_idx, s3], s2, s3)
        made_finals[sim_idx, sf1_win] = True
        made_finals[sim_idx, sf2_win] = True

        sc_ch = get_scores(1)
        champ = np.where(sc_ch[sim_idx, sf1_win] >= sc_ch[sim_idx, sf2_win], sf1_win, sf2_win)
        won_champ[sim_idx, champ] = True

    elif playoff_teams == 8 and len(po_weeks) >= 3:
        seeds = [order[:, k] for k in range(8)]
        sc_qf = get_scores(0)
        qf1 = np.where(sc_qf[sim_idx, seeds[0]] >= sc_qf[sim_idx, seeds[7]], seeds[0], seeds[7])
        qf2 = np.where(sc_qf[sim_idx, seeds[3]] >= sc_qf[sim_idx, seeds[4]], seeds[3], seeds[4])
        qf3 = np.where(sc_qf[sim_idx, seeds[1]] >= sc_qf[sim_idx, seeds[6]], seeds[1], seeds[6])
        qf4 = np.where(sc_qf[sim_idx, seeds[2]] >= sc_qf[sim_idx, seeds[5]], seeds[2], seeds[5])

        for qf in (qf1, qf2, qf3, qf4):
            made_semis[sim_idx, qf] = True

        sc_sf = get_scores(1)
        sf1 = np.where(sc_sf[sim_idx, qf1] >= sc_sf[sim_idx, qf2], qf1, qf2)
        sf2 = np.where(sc_sf[sim_idx, qf3] >= sc_sf[sim_idx, qf4], qf3, qf4)
        made_finals[sim_idx, sf1] = True
        made_finals[sim_idx, sf2] = True

        sc_ch = get_scores(2)
        champ = np.where(sc_ch[sim_idx, sf1] >= sc_ch[sim_idx, sf2], sf1, sf2)
        won_champ[sim_idx, champ] = True

    else:
        # Generic 2-team final fallback
        for k in range(min(4, playoff_teams)):
            made_semis[sim_idx, order[:, k]] = True
        s1, s2 = order[:, 0], order[:, 1]
        made_finals[sim_idx, s1] = True
        made_finals[sim_idx, s2] = True
        sc_ch = get_scores(len(po_weeks) - 1)
        champ = np.where(sc_ch[sim_idx, s1] >= sc_ch[sim_idx, s2], s1, s2)
        won_champ[sim_idx, champ] = True

    return made_playoffs, first_round_bye, made_semis, made_finals, won_champ


def run_league_simulation(
    team_score_matrix: np.ndarray,
    sim_ctx: SimulationContext,
    track_weekly_for_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run vectorized regular-season (H2H + Median) and playoff Monte Carlo simulation."""
    num_sims = sim_ctx.num_sims
    num_teams = sim_ctx.num_teams

    sim_wins = np.broadcast_to(sim_ctx.current_wins, (num_sims, num_teams)).copy()
    sim_pf = np.broadcast_to(sim_ctx.current_pf, (num_sims, num_teams)).copy()

    tracked = set(track_weekly_for_indices) if track_weekly_for_indices is not None else set(range(num_teams))
    h2h_win_probs: Dict[int, Dict[int, float]] = {idx: {} for idx in tracked}
    med_win_probs: Dict[int, Dict[int, float]] = {idx: {} for idx in tracked}
    h2h_opponents: Dict[int, Dict[int, int]] = {idx: {} for idx in tracked}

    for w in sim_ctx.reg_weeks:
        w_idx = sim_ctx.week_to_idx[w]
        scores = team_score_matrix[:, w_idx, :]
        sim_pf += scores

        for idx_i, idx_j in sim_ctx.schedule_idx_by_week.get(w, []):
            sc_i = scores[:, idx_i]
            sc_j = scores[:, idx_j]
            win_i = sc_i > sc_j
            win_j = sc_j > sc_i
            tie = sc_i == sc_j
            sim_wins[:, idx_i] += win_i.astype(np.float32) + 0.5 * tie.astype(np.float32)
            sim_wins[:, idx_j] += win_j.astype(np.float32) + 0.5 * tie.astype(np.float32)

            if idx_i in tracked:
                h2h_win_probs[idx_i][w] = round(
                    float(np.mean(win_i.astype(np.float32) + 0.5 * tie.astype(np.float32))) * 100.0,
                    2,
                )
                h2h_opponents[idx_i][w] = sim_ctx.idx_to_roster_id[idx_j]
            if idx_j in tracked:
                h2h_win_probs[idx_j][w] = round(float(np.mean(win_j.astype(np.float32) + 0.5 * tie.astype(np.float32))) * 100.0, 2)
                h2h_opponents[idx_j][w] = sim_ctx.idx_to_roster_id[idx_i]

        # League Median Scoring (top half beats weekly median)
        med = np.median(scores, axis=1, keepdims=True)
        med_win = scores > med
        if sim_ctx.use_median_scoring:
            sim_wins += med_win.astype(np.float32)
        for idx in tracked:
            med_win_probs[idx][w] = round(float(np.mean(med_win[:, idx])) * 100.0, 2)

    # Seed playoffs by total wins + Points For tiebreaker
    ranking_key = sim_wins + sim_pf * np.float32(1e-5)
    order = np.argsort(-ranking_key, axis=1)

    made_playoffs, first_round_bye, made_semis, made_finals, won_champ = _simulate_playoffs(
        team_score_matrix, order, sim_ctx
    )

    exp_wins = np.mean(sim_wins, axis=0)
    exp_pf = np.mean(sim_pf, axis=0)
    playoff_pct = np.mean(made_playoffs, axis=0) * 100.0
    bye_pct = np.mean(first_round_bye, axis=0) * 100.0
    semis_pct = np.mean(made_semis, axis=0) * 100.0
    finals_pct = np.mean(made_finals, axis=0) * 100.0
    champ_pct = np.mean(won_champ, axis=0) * 100.0

    # Expected remaining regular season games per team
    games_per_week = 2.0 if sim_ctx.use_median_scoring else 1.0
    rem_games = len(sim_ctx.reg_weeks) * games_per_week
    total_current_games = np.array(
        [float(t["wins"] + t["losses"] + t.get("ties", 0)) for t in sim_ctx.teams],
        dtype=np.float32,
    )

    team_results: List[Dict[str, Any]] = []
    for idx in range(num_teams):
        w_exp = float(exp_wins[idx])
        total_games = float(total_current_games[idx]) + rem_games
        l_exp = max(0.0, total_games - w_exp)
        team_results.append(
            {
                "team_idx": idx,
                "roster_id": sim_ctx.idx_to_roster_id[idx],
                "exp_wins": round(w_exp, 2),
                "exp_losses": round(l_exp, 2),
                "exp_pf": round(float(exp_pf[idx]), 1),
                "playoff_pct": round(float(playoff_pct[idx]), 2),
                "bye_pct": round(float(bye_pct[idx]), 2),
                "semis_pct": round(float(semis_pct[idx]), 2),
                "finals_pct": round(float(finals_pct[idx]), 2),
                "champ_pct": round(float(champ_pct[idx]), 2),
            }
        )

    return {
        "teams": team_results,
        "h2h_win_probs": h2h_win_probs,
        "med_win_probs": med_win_probs,
        "h2h_opponents": h2h_opponents,
    }


def fast_reevaluate_trade(
    baseline_team_matrix: np.ndarray,
    idx_a: int,
    new_scores_a: np.ndarray,
    idx_b: int,
    new_scores_b: np.ndarray,
    sim_ctx: SimulationContext,
    track_weekly_for_both: bool = True,
) -> Dict[str, Any]:
    """Clone baseline team score matrix for just 2 traded teams and re-run 5,000 CRN sims."""
    mod_matrix = baseline_team_matrix.copy()
    mod_matrix[:, :, idx_a] = new_scores_a
    mod_matrix[:, :, idx_b] = new_scores_b
    tracked = [idx_a, idx_b] if track_weekly_for_both else []
    return run_league_simulation(mod_matrix, sim_ctx, track_weekly_for_indices=tracked)
