"""3-Stage multi-player trade package finder and single-trade sandbox evaluator."""

from __future__ import annotations

import itertools
import math
import time
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.special import ndtr

from .fantasycalc import evaluate_perceived_trade, get_player_fc_value
from .lineup import (
    resolve_imbalanced_trade_rosters,
    solve_full_strength_ppg,
    solve_weekly_lineup,
)
from .persuasion import generate_trade_persuasion
from .simulator import (
    SimulationContext,
    _pos_sigma,
    compute_team_weekly_profile,
    fast_reevaluate_trade,
)

SKILL_POSITIONS: Set[str] = {"QB", "RB", "WR", "TE"}
DEFAULT_PACKAGE_SIZES: List[Tuple[int, int]] = [
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 1),
    (1, 3),
    (3, 2),
    (2, 3),
    (3, 3),
]


def _parse_package_sizes(raw_sizes: Optional[List[str]]) -> List[Tuple[int, int]]:
    if not raw_sizes:
        return list(DEFAULT_PACKAGE_SIZES)
    parsed: List[Tuple[int, int]] = []
    for s in raw_sizes:
        parts = str(s).replace("v", "-").split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            a, b = int(parts[0]), int(parts[1])
            if 1 <= a <= 3 and 1 <= b <= 3:
                pair = (a, b)
                if pair not in parsed:
                    parsed.append(pair)
    return parsed or list(DEFAULT_PACKAGE_SIZES)


def _compute_deterministic_profile(
    roster_ids: List[str],
    reserve_ids: Set[str],
    sim_ctx: SimulationContext,
) -> Dict[str, Any]:
    """Compute weekly deterministic starter means, variances, and set of players who ever start."""
    num_weeks = len(sim_ctx.sim_weeks)
    mu_arr = np.zeros(num_weeks, dtype=np.float64)
    var_arr = np.zeros(num_weeks, dtype=np.float64)
    ever_started: Set[str] = set()

    for w_idx, w in enumerate(sim_ctx.sim_weeks):
        total_pts, starters, _ = solve_weekly_lineup(
            roster_ids,
            w,
            sim_ctx.weekly_projections,
            sim_ctx.players_master,
            sim_ctx.streaming_baselines,
            sim_ctx.starter_slots,
            reserve_ids=reserve_ids,
        )
        w_mu = 0.0
        w_var = 0.0
        for st in starters:
            pid = st["player_id"]
            if st["is_streamer"]:
                pos_idx = sim_ctx.pos_to_idx.get(st["position"], 2)
                m = float(sim_ctx.streamer_mu[pos_idx, w_idx])
                s = float(sim_ctx.streamer_sigma[pos_idx, w_idx])
                w_mu += m
                w_var += s * s
            else:
                ever_started.add(pid)
                p_idx = sim_ctx.pid_to_idx.get(pid)
                if p_idx is not None:
                    m = float(sim_ctx.mu[p_idx, w_idx])
                    s = float(sim_ctx.sigma[p_idx, w_idx])
                    w_mu += m
                    w_var += s * s
                else:
                    pts = float(st["points"])
                    w_mu += pts
                    s = _pos_sigma(st["position"], pts)
                    w_var += s * s

        mu_arr[w_idx] = w_mu if w_mu > 0 else total_pts
        var_arr[w_idx] = max(w_var, 4.0)

    return {
        "mu": mu_arr,
        "var": var_arr,
        "ever_started": ever_started,
    }


def _analytical_team_metrics(
    team_idx: int,
    profile: Dict[str, Any],
    baseline_mu: np.ndarray,
    baseline_var: np.ndarray,
    baseline_med_mu: np.ndarray,
    baseline_med_var: np.ndarray,
    sim_ctx: SimulationContext,
    other_idx: Optional[int] = None,
    other_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Estimate regular-season wins (H2H + Median), playoff points, and starter PPG analytically."""
    mu = profile["mu"]
    var = profile["var"]

    exp_rem_wins = 0.0
    for w in sim_ctx.reg_weeks:
        w_idx = sim_ctx.week_to_idx[w]
        # Find scheduled opponent in week w
        opp_idx = None
        for i, j in sim_ctx.schedule_idx_by_week.get(w, []):
            if i == team_idx:
                opp_idx = j
                break
            if j == team_idx:
                opp_idx = i
                break

        if opp_idx is not None:
            if other_idx is not None and opp_idx == other_idx and other_profile is not None:
                opp_m = float(other_profile["mu"][w_idx])
                opp_v = float(other_profile["var"][w_idx])
            else:
                opp_m = float(baseline_mu[opp_idx, w_idx])
                opp_v = float(baseline_var[opp_idx, w_idx])
            z_h2h = (float(mu[w_idx]) - opp_m) / math.sqrt(max(float(var[w_idx]) + opp_v, 1.0))
            exp_rem_wins += float(ndtr(z_h2h))

        if sim_ctx.use_median_scoring:
            z_med = (float(mu[w_idx]) - float(baseline_med_mu[w_idx])) / math.sqrt(
                max(float(var[w_idx]) + float(baseline_med_var[w_idx]), 1.0)
            )
            exp_rem_wins += float(ndtr(z_med))

    po_indices = [sim_ctx.week_to_idx[w] for w in sim_ctx.playoff_weeks]
    po_pts = float(np.sum(mu[po_indices])) if po_indices else 0.0
    avg_ppg = float(np.mean(mu)) if len(mu) > 0 else 0.0

    return {
        "exp_rem_wins": exp_rem_wins,
        "playoff_pts": po_pts,
        "starter_ppg": avg_ppg,
    }


def _select_tradable_assets(
    team: Dict[str, Any],
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
    must_include: Set[str],
    exclude: Set[str],
    max_assets: int = 12,
) -> List[str]:
    """Select active skill players eligible for package generation."""
    reserve_set = set(team.get("reserve") or [])
    scored: List[Tuple[float, str]] = []
    forced: List[str] = []

    for pid in team.get("players") or []:
        if pid in exclude:
            continue
        pinfo = players_master.get(pid)
        if not pinfo:
            continue
        pos = str(pinfo.get("position") or "")
        if pos not in SKILL_POSITIONS:
            continue
        if pid in must_include:
            forced.append(pid)
            continue
        if pid in reserve_set:
            continue
        ros_pts = float((ros_totals.get(pid) or {}).get("total_pts") or 0.0)
        fc = float((fc_values.get(pid) or {}).get("fc_value") or 0.0)
        if ros_pts <= 10.0 and fc <= 250.0:
            continue
        score = fc + ros_pts * 20.0
        scored.append((score, pid))

    scored.sort(key=lambda item: item[0], reverse=True)
    result = list(forced)
    for _, pid in scored:
        if pid not in result:
            result.append(pid)
        if len(result) >= max_assets:
            break
    return result


def _package_combos_by_size(
    assets: List[str],
    sizes: Set[int],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
    required_subset: Optional[Set[str]] = None,
) -> Dict[int, List[Tuple[float, Tuple[str, ...]]]]:
    """Build sorted lists of (raw_fc_value, tuple_of_pids) per package size."""
    req = required_subset or set()
    by_size: Dict[int, List[Tuple[float, Tuple[str, ...]]]] = {}
    val_map = {pid: get_player_fc_value(pid, fc_values, ros_totals) for pid in assets}

    for k in sorted(sizes):
        if len(assets) < k:
            by_size[k] = []
            continue
        combos: List[Tuple[float, Tuple[str, ...]]] = []
        for comb in itertools.combinations(assets, k):
            if req and not req.issubset(comb):
                continue
            raw_val = sum(val_map[p] for p in comb)
            combos.append((raw_val, comb))
        combos.sort(key=lambda item: item[0])
        by_size[k] = combos
    return by_size


def _format_player_brief(
    pid: str,
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pinfo = players_master.get(pid) or {}
    ros = ros_totals.get(pid) or {}
    fc_entry = fc_values.get(pid) or {}
    return {
        "player_id": pid,
        "full_name": pinfo.get("full_name", pid),
        "position": pinfo.get("position", "?"),
        "team": pinfo.get("team", "FA"),
        "injury_status": pinfo.get("injury_status", ""),
        "ppg": round(float(ros.get("ppg") or 0.0), 2),
        "ros_pts": round(float(ros.get("total_pts") or 0.0), 1),
        "playoff_pts": round(float(ros.get("playoff_pts") or 0.0), 1),
        "next_bye": ros.get("next_bye"),
        "fc_value": round(get_player_fc_value(pid, fc_values, ros_totals), 0),
        "overall_rank": int(fc_entry.get("overall_rank") or 999),
    }


def _acceptance_multiplier(partner_gain_pct: float) -> float:
    """Weight acceptance realism: peaks around +3% to +10% in partner's favor."""
    if partner_gain_pct >= 0.0:
        if partner_gain_pct <= 12.0:
            return 1.0 + 0.015 * partner_gain_pct
        return max(0.75, 1.18 - 0.012 * (partner_gain_pct - 12.0))
    return max(0.45, 1.0 + 0.035 * partner_gain_pct)


def find_viable_trades(
    session: Dict[str, Any],
    my_roster_id: int,
    partner_roster_id: Optional[int] = None,
    package_sizes: Optional[List[str]] = None,
    min_perceived_pct: float = -12.0,
    max_perceived_pct: float = 20.0,
    mode: str = "mutual_champ",
    must_include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    require_both_champ_up: bool = True,
    allow_same_pos_1for1: bool = False,
    max_results: int = 40,
) -> Dict[str, Any]:
    """Run the 3-stage funnel to discover mutually beneficial multi-player trades."""
    t0 = time.perf_counter()

    league_info = session["league_info"]
    players_master = session["players_master"]
    fc_values = session["fc_values"]
    waiver_cal = session["waiver_calibration"]
    ros_totals = session["ros_totals"]
    sim_ctx: SimulationContext = session["sim_ctx"]
    baseline_matrix: np.ndarray = session["baseline_team_matrix"]
    baseline_sim: Dict[str, Any] = session["baseline_sim"]
    baseline_mu: np.ndarray = session["baseline_mu"]
    baseline_var: np.ndarray = session["baseline_var"]
    baseline_med_mu: np.ndarray = session["baseline_med_mu"]
    baseline_med_var: np.ndarray = session["baseline_med_var"]
    baseline_det_metrics: Dict[int, Dict[str, float]] = session["baseline_det_metrics"]

    teams_by_rid = {int(t["roster_id"]): t for t in league_info["teams"]}
    if my_roster_id not in teams_by_rid:
        raise ValueError(f"Roster ID {my_roster_id} not found.")

    my_team = teams_by_rid[my_roster_id]
    my_idx = sim_ctx.roster_id_to_idx[my_roster_id]
    my_roster_set = set(my_team.get("players") or [])

    must_set = set(str(x) for x in (must_include or []))
    excl_set = set(str(x) for x in (exclude or []))
    must_a = must_set & my_roster_set

    sizes_list = _parse_package_sizes(package_sizes)
    sizes_a_needed = {a for a, _ in sizes_list}
    sizes_b_needed = {b for _, b in sizes_list}

    my_assets = _select_tradable_assets(
        my_team, players_master, fc_values, ros_totals, must_a, excl_set, max_assets=12
    )
    my_combos = _package_combos_by_size(
        my_assets, sizes_a_needed, fc_values, ros_totals, required_subset=must_a
    )

    if partner_roster_id is not None:
        partner_rids = [int(partner_roster_id)] if int(partner_roster_id) != my_roster_id else []
    else:
        partner_rids = [int(t["roster_id"]) for t in league_info["teams"] if int(t["roster_id"]) != my_roster_id]

    waiver_adj_1 = float(waiver_cal["waiver_adj_1"])
    waiver_adj_2 = float(waiver_cal["waiver_adj_2"])
    g_min = min_perceived_pct / 100.0
    g_max = max_perceived_pct / 100.0

    # Stage 1: Meet-in-the-Middle Value-Band Pruning
    stage1_candidates: List[Tuple[int, Tuple[str, ...], Tuple[str, ...], Dict[str, Any]]] = []
    stage1_total_examined = 0

    for p_rid in partner_rids:
        p_team = teams_by_rid.get(p_rid)
        if not p_team:
            continue
        p_roster_set = set(p_team.get("players") or [])
        must_b = must_set & p_roster_set
        # If user required specific players from another team, skip partners without those players
        unmatched_other_must = must_set - my_roster_set - p_roster_set
        if unmatched_other_must:
            continue

        p_assets = _select_tradable_assets(
            p_team, players_master, fc_values, ros_totals, must_b, excl_set, max_assets=12
        )
        p_combos = _package_combos_by_size(
            p_assets, sizes_b_needed, fc_values, ros_totals, required_subset=must_b
        )

        for sz_a, sz_b in sizes_list:
            list_a = my_combos.get(sz_a) or []
            list_b = p_combos.get(sz_b) or []
            if not list_a or not list_b:
                continue

            raw_b_keys = [item[0] for item in list_b]
            spot_diff = sz_a - sz_b
            abs_diff = abs(spot_diff)
            w_adj = 0.0 if abs_diff == 0 else (waiver_adj_1 if abs_diff == 1 else waiver_adj_2)
            stage1_total_examined += len(list_a) * len(list_b)

            for raw_a, pkg_a in list_a:
                if raw_a < 150.0:
                    continue

                if spot_diff > 0:
                    # adjusted_a = raw_a; adjusted_b = raw_b + w_adj
                    adj_a = raw_a
                    # Safe broad bounds accounting for denom floor 500
                    if 1.0 + g_max > 0.05:
                        lo_b = min(
                            adj_a - g_max * 500.0 - w_adj,
                            adj_a / (1.0 + g_max) - w_adj,
                        )
                    else:
                        lo_b = 0.0
                    if 1.0 + g_min > 0.05:
                        hi_b = max(
                            adj_a - g_min * 500.0 - w_adj,
                            adj_a / (1.0 + g_min) - w_adj,
                        )
                    else:
                        hi_b = 1e9
                else:
                    # adjusted_a = raw_a + w_adj; adjusted_b = raw_b
                    adj_a = raw_a + w_adj
                    if 1.0 + g_max > 0.05:
                        lo_b = min(adj_a - g_max * 500.0, adj_a / (1.0 + g_max))
                    else:
                        lo_b = 0.0
                    if 1.0 + g_min > 0.05:
                        hi_b = max(adj_a - g_min * 500.0, adj_a / (1.0 + g_min))
                    else:
                        hi_b = 1e9

                idx_lo = bisect_left(raw_b_keys, lo_b - 5.0)
                idx_hi = bisect_right(raw_b_keys, hi_b + 5.0)

                for j in range(idx_lo, idx_hi):
                    raw_b, pkg_b = list_b[j]
                    if raw_b < 150.0:
                        continue
                    if not allow_same_pos_1for1 and sz_a == 1 and sz_b == 1:
                        pos_a = (players_master.get(pkg_a[0]) or {}).get("position")
                        pos_b = (players_master.get(pkg_b[0]) or {}).get("position")
                        if pos_a == pos_b:
                            continue

                    if spot_diff > 0:
                        adj_b = raw_b + w_adj
                    else:
                        adj_b = raw_b
                    pct = 100.0 * (adj_a - adj_b) / (adj_b if adj_b > 500.0 else 500.0)
                    if min_perceived_pct <= pct <= max_perceived_pct:
                        stage1_candidates.append((p_rid, pkg_a, pkg_b, pct))

    t1 = time.perf_counter()

    # Stage 2: Fast Analytical Lineup & Win-Delta Pruning
    # Stratified cap across (partner, size_a, size_b) so 3-for-2/3-for-3 never crowd out 1-for-1/2-for-1
    if len(stage1_candidates) > 2200:
        buckets: Dict[Tuple[int, int, int], List[Tuple[int, Tuple[str, ...], Tuple[str, ...], float]]] = {}
        for item in stage1_candidates:
            bkey = (item[0], len(item[1]), len(item[2]))
            buckets.setdefault(bkey, []).append(item)

        stratified: List[Tuple[int, Tuple[str, ...], Tuple[str, ...], float]] = []
        for (p_rid, sz_a, sz_b), items in buckets.items():
            # Keep smaller packages generously; cap 3-player buckets so Stage 2 stays fast
            cap = 48 if (sz_a + sz_b) <= 4 else 14
            if len(items) > cap:
                items.sort(key=lambda x: abs(x[3] - 4.0))
                items = items[:cap]
            stratified.extend(items)
        stage1_candidates = stratified

    stage2_scored: List[Tuple[float, Dict[str, Any]]] = []
    my_reserve = set(my_team.get("reserve") or [])
    base_det_a = baseline_det_metrics[my_idx]
    allow_stash_relax = len(sim_ctx.sim_weeks) <= 3
    det_cache: Dict[Tuple[frozenset, frozenset], Dict[str, Any]] = {}
    full_ppg_cache: Dict[Tuple[frozenset, frozenset], float] = {}

    base_full_ppg: Dict[int, float] = {
        idx: solve_full_strength_ppg(
            t["players"],
            ros_totals,
            players_master,
            sim_ctx.starter_slots,
            reserve_ids=set(t.get("reserve") or []),
        )
        for idx, t in enumerate(league_info["teams"])
    }

    def _get_det(r_ids: List[str], res_ids: Set[str]) -> Dict[str, Any]:
        ckey = (frozenset(r_ids), frozenset(res_ids))
        cached_det = det_cache.get(ckey)
        if cached_det is None:
            cached_det = _compute_deterministic_profile(r_ids, res_ids, sim_ctx)
            det_cache[ckey] = cached_det
        return cached_det

    def _get_full_ppg(r_ids: List[str], res_ids: Set[str]) -> float:
        ckey = (frozenset(r_ids), frozenset(res_ids))
        cached_val = full_ppg_cache.get(ckey)
        if cached_val is None:
            cached_val = solve_full_strength_ppg(
                r_ids, ros_totals, players_master, sim_ctx.starter_slots, reserve_ids=res_ids
            )
            full_ppg_cache[ckey] = cached_val
        return cached_val

    for p_rid, pkg_a, pkg_b, pct in stage1_candidates:
        perceived = evaluate_perceived_trade(
            list(pkg_a), list(pkg_b), fc_values, waiver_cal, ros_totals
        )
        p_team = teams_by_rid[p_rid]
        p_idx = sim_ctx.roster_id_to_idx[p_rid]
        p_reserve = set(p_team.get("reserve") or [])
        base_det_b = baseline_det_metrics[p_idx]

        tx = resolve_imbalanced_trade_rosters(
            my_team["players"],
            p_team["players"],
            list(pkg_a),
            list(pkg_b),
            my_reserve,
            p_reserve,
            players_master,
            fc_values,
            ros_totals,
            waiver_cal["waiver_pool"],
            sim_ctx.starter_slots,
        )
        if tx["invalid_padding"]:
            continue

        det_a = _get_det(tx["final_a_ids"], tx["reserve_a"])
        det_b = _get_det(tx["final_b_ids"], tx["reserve_b"])

        # Redundant sub-package check: multi-player side must contribute starters to recipient
        redundant = False
        if len(pkg_a) >= 2:
            for pid in pkg_a:
                if pid not in det_b["ever_started"]:
                    if not (allow_stash_relax and get_player_fc_value(pid, fc_values, ros_totals) >= 1200.0):
                        redundant = True
                        break
        if not redundant and len(pkg_b) >= 2:
            for pid in pkg_b:
                if pid not in det_a["ever_started"]:
                    if not (allow_stash_relax and get_player_fc_value(pid, fc_values, ros_totals) >= 1200.0):
                        redundant = True
                        break
        if redundant:
            continue

        d_full_ppg_a = _get_full_ppg(tx["final_a_ids"], tx["reserve_a"]) - base_full_ppg[my_idx]
        d_full_ppg_b = _get_full_ppg(tx["final_b_ids"], tx["reserve_b"]) - base_full_ppg[p_idx]

        # Full-Strength Lineup Gate: reject trades that downgrade either team's non-bye starting lineup
        if d_full_ppg_a < -0.10 or d_full_ppg_b < -0.10:
            continue

        met_a = _analytical_team_metrics(
            my_idx, det_a, baseline_mu, baseline_var, baseline_med_mu, baseline_med_var, sim_ctx,
            other_idx=p_idx, other_profile=det_b,
        )
        met_b = _analytical_team_metrics(
            p_idx, det_b, baseline_mu, baseline_var, baseline_med_mu, baseline_med_var, sim_ctx,
            other_idx=my_idx, other_profile=det_a,
        )

        d_wins_a = met_a["exp_rem_wins"] - base_det_a["exp_rem_wins"]
        d_wins_b = met_b["exp_rem_wins"] - base_det_b["exp_rem_wins"]
        d_po_a = met_a["playoff_pts"] - base_det_a["playoff_pts"]
        d_po_b = met_b["playoff_pts"] - base_det_b["playoff_pts"]
        d_ppg_a = met_a["starter_ppg"] - base_det_a["starter_ppg"]
        d_ppg_b = met_b["starter_ppg"] - base_det_b["starter_ppg"]

        benefit_a = d_wins_a * 4.5 + d_po_a * 0.30 + d_full_ppg_a * 1.6 + d_ppg_a * 0.6
        benefit_b = d_wins_b * 4.5 + d_po_b * 0.30 + d_full_ppg_b * 1.6 + d_ppg_b * 0.6

        if benefit_a <= 0.05 or benefit_b <= 0.05:
            continue
        if d_wins_a < -0.35 or d_wins_b < -0.35:
            continue

        acc_mult = _acceptance_multiplier(perceived["partner_gain_pct"])
        if mode == "win_now":
            s2_score = (d_wins_a * 5.0 + d_wins_b * 3.0 + 0.25 * (d_ppg_a + d_ppg_b)) * acc_mult
        elif mode == "my_champ":
            s2_score = (benefit_a * 1.4 + benefit_b * 0.45) * acc_mult
        else:
            s2_score = (math.sqrt(benefit_a * benefit_b) * 1.6 + 0.3 * (benefit_a + benefit_b)) * acc_mult

        stage2_scored.append(
            (
                s2_score,
                {
                    "partner_roster_id": p_rid,
                    "partner_idx": p_idx,
                    "pkg_a": list(pkg_a),
                    "pkg_b": list(pkg_b),
                    "perceived": perceived,
                    "tx": tx,
                    "d_ppg_a": round(d_ppg_a, 2),
                    "d_ppg_b": round(d_ppg_b, 2),
                    "d_full_ppg_a": round(d_full_ppg_a, 2),
                    "d_full_ppg_b": round(d_full_ppg_b, 2),
                    "d_po_a": round(d_po_a, 1),
                    "d_po_b": round(d_po_b, 1),
                    "s2_wins_a": round(d_wins_a, 3),
                    "s2_wins_b": round(d_wins_b, 3),
                },
            )
        )

    stage2_scored.sort(key=lambda item: item[0], reverse=True)

    # Diversity-preserving selection of top 75 candidates for exact CRN Monte Carlo
    selected_for_mc: List[Dict[str, Any]] = []
    partner_counts: Dict[int, int] = {}
    size_partner_counts: Dict[Tuple[int, int, int], int] = {}
    for _, cand in stage2_scored:
        p_rid = cand["partner_roster_id"]
        sp_key = (p_rid, len(cand["pkg_a"]), len(cand["pkg_b"]))
        if partner_counts.get(p_rid, 0) >= 12 and len(partner_rids) > 1:
            continue
        if size_partner_counts.get(sp_key, 0) >= 4 and len(partner_rids) > 1:
            continue
        selected_for_mc.append(cand)
        partner_counts[p_rid] = partner_counts.get(p_rid, 0) + 1
        size_partner_counts[sp_key] = size_partner_counts.get(sp_key, 0) + 1
        if len(selected_for_mc) >= 75:
            break

    # Backfill up to 75 if partner diversity cap left room
    if len(selected_for_mc) < 75 and len(stage2_scored) > len(selected_for_mc):
        seen_keys = {
            (c["partner_roster_id"], tuple(c["pkg_a"]), tuple(c["pkg_b"]))
            for c in selected_for_mc
        }
        for _, cand in stage2_scored:
            k = (cand["partner_roster_id"], tuple(cand["pkg_a"]), tuple(cand["pkg_b"]))
            if k not in seen_keys:
                selected_for_mc.append(cand)
                seen_keys.add(k)
            if len(selected_for_mc) >= 75:
                break

    t2 = time.perf_counter()

    # Stage 3: Exact CRN Monte Carlo Verification
    base_teams_by_idx = {t["team_idx"]: t for t in baseline_sim["teams"]}
    base_a = base_teams_by_idx[my_idx]
    verified_results: List[Dict[str, Any]] = []
    mc_prof_cache: Dict[Tuple[frozenset, frozenset], Dict[str, Any]] = {}

    def _get_mc_prof(r_ids: List[str], res_ids: Set[str]) -> Dict[str, Any]:
        ckey = (frozenset(r_ids), frozenset(res_ids))
        cached_prof = mc_prof_cache.get(ckey)
        if cached_prof is None:
            cached_prof = compute_team_weekly_profile(r_ids, res_ids, sim_ctx)
            mc_prof_cache[ckey] = cached_prof
        return cached_prof

    for cand in selected_for_mc:
        p_rid = cand["partner_roster_id"]
        p_idx = cand["partner_idx"]
        p_team = teams_by_rid[p_rid]
        base_b = base_teams_by_idx[p_idx]
        tx = cand["tx"]

        prof_a = _get_mc_prof(tx["final_a_ids"], tx["reserve_a"])
        prof_b = _get_mc_prof(tx["final_b_ids"], tx["reserve_b"])

        sim_out = fast_reevaluate_trade(
            baseline_matrix,
            my_idx,
            prof_a["scores"],
            p_idx,
            prof_b["scores"],
            sim_ctx,
            track_weekly_for_both=False,
        )
        post_by_idx = {t["team_idx"]: t for t in sim_out["teams"]}
        post_a = post_by_idx[my_idx]
        post_b = post_by_idx[p_idx]

        d_champ_a = round(post_a["champ_pct"] - base_a["champ_pct"], 2)
        d_champ_b = round(post_b["champ_pct"] - base_b["champ_pct"], 2)
        d_finals_a = round(post_a["finals_pct"] - base_a["finals_pct"], 2)
        d_finals_b = round(post_b["finals_pct"] - base_b["finals_pct"], 2)
        d_semis_a = round(post_a["semis_pct"] - base_a["semis_pct"], 2)
        d_semis_b = round(post_b["semis_pct"] - base_b["semis_pct"], 2)
        d_bye_a = round(post_a["bye_pct"] - base_a["bye_pct"], 2)
        d_bye_b = round(post_b["bye_pct"] - base_b["bye_pct"], 2)
        d_po_pct_a = round(post_a["playoff_pct"] - base_a["playoff_pct"], 2)
        d_po_pct_b = round(post_b["playoff_pct"] - base_b["playoff_pct"], 2)
        d_wins_a = round(post_a["exp_wins"] - base_a["exp_wins"], 2)
        d_wins_b = round(post_b["exp_wins"] - base_b["exp_wins"], 2)

        if require_both_champ_up:
            if d_champ_a <= 0.0 or d_champ_b <= 0.0:
                continue
        else:
            if d_champ_a < -0.05 and d_po_pct_a < 0.2 and d_wins_a < 0.15:
                continue
            if d_champ_b < -0.05 and d_po_pct_b < 0.2 and d_wins_b < 0.15:
                continue
            if (d_champ_a + 0.2 * d_po_pct_a + 0.5 * d_wins_a) <= 0.0:
                continue
            if (d_champ_b + 0.2 * d_po_pct_b + 0.5 * d_wins_b) <= 0.0:
                continue

        perceived = cand["perceived"]
        acc_mult = _acceptance_multiplier(perceived["partner_gain_pct"])
        pos_ca = max(0.0, d_champ_a)
        pos_cb = max(0.0, d_champ_b)

        if mode == "my_champ":
            rank_score = (
                d_champ_a * 2.2
                + 0.30 * d_po_pct_a
                + 0.20 * d_champ_b
                + 0.15 * d_bye_a
            ) * acc_mult
        elif mode == "win_now":
            rank_score = (
                (d_wins_a * 3.0 + d_po_pct_a * 0.50)
                + 0.55 * (d_wins_b * 2.4 + d_po_pct_b * 0.45)
                + 0.40 * (d_champ_a + d_champ_b)
            ) * acc_mult
        else:
            rank_score = (
                math.sqrt(pos_ca * pos_cb) * 2.4
                + 0.35 * (d_champ_a + d_champ_b)
                + 0.12 * (min(d_po_pct_a, 6.0) + min(d_po_pct_b, 6.0))
                + 0.08 * (d_bye_a + d_bye_b)
            ) * acc_mult

        verified_results.append(
            {
                "rank_score": round(rank_score, 3),
                "structure": f"{len(cand['pkg_a'])}-for-{len(cand['pkg_b'])}",
                "my_roster_id": my_roster_id,
                "my_team_name": my_team["team_name"],
                "partner_roster_id": p_rid,
                "partner_team_name": p_team["team_name"],
                "partner_manager_name": p_team["manager_name"],
                "_tx": tx,
                "send_a": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in cand["pkg_a"]
                ],
                "send_b": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in cand["pkg_b"]
                ],
                "drops_a": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in tx["drops_a"]
                ],
                "adds_a": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in tx["adds_a"]
                ],
                "drops_b": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in tx["drops_b"]
                ],
                "adds_b": [
                    _format_player_brief(pid, players_master, fc_values, ros_totals)
                    for pid in tx["adds_b"]
                ],
                "perceived": perceived,
                "impact_a": {
                    "d_champ_pct": d_champ_a,
                    "d_finals_pct": d_finals_a,
                    "d_semis_pct": d_semis_a,
                    "d_bye_pct": d_bye_a,
                    "d_playoff_pct": d_po_pct_a,
                    "d_exp_wins": d_wins_a,
                    "d_starter_ppg": cand["d_ppg_a"],
                    "d_full_ppg": cand["d_full_ppg_a"],
                    "d_playoff_pts": cand["d_po_a"],
                    "base_champ_pct": base_a["champ_pct"],
                    "post_champ_pct": post_a["champ_pct"],
                    "base_finals_pct": base_a["finals_pct"],
                    "post_finals_pct": post_a["finals_pct"],
                    "base_semis_pct": base_a["semis_pct"],
                    "post_semis_pct": post_a["semis_pct"],
                    "base_bye_pct": base_a["bye_pct"],
                    "post_bye_pct": post_a["bye_pct"],
                    "base_playoff_pct": base_a["playoff_pct"],
                    "post_playoff_pct": post_a["playoff_pct"],
                    "base_exp_wins": base_a["exp_wins"],
                    "post_exp_wins": post_a["exp_wins"],
                },
                "impact_b": {
                    "d_champ_pct": d_champ_b,
                    "d_finals_pct": d_finals_b,
                    "d_semis_pct": d_semis_b,
                    "d_bye_pct": d_bye_b,
                    "d_playoff_pct": d_po_pct_b,
                    "d_exp_wins": d_wins_b,
                    "d_starter_ppg": cand["d_ppg_b"],
                    "d_full_ppg": cand["d_full_ppg_b"],
                    "d_playoff_pts": cand["d_po_b"],
                    "base_champ_pct": base_b["champ_pct"],
                    "post_champ_pct": post_b["champ_pct"],
                    "base_finals_pct": base_b["finals_pct"],
                    "post_finals_pct": post_b["finals_pct"],
                    "base_semis_pct": base_b["semis_pct"],
                    "post_semis_pct": post_b["semis_pct"],
                    "base_bye_pct": base_b["bye_pct"],
                    "post_bye_pct": post_b["bye_pct"],
                    "base_playoff_pct": base_b["playoff_pct"],
                    "post_playoff_pct": post_b["playoff_pct"],
                    "base_exp_wins": base_b["exp_wins"],
                    "post_exp_wins": post_b["exp_wins"],
                },
            }
        )

    verified_results.sort(key=lambda item: item["rank_score"], reverse=True)
    top_trades = verified_results[:max_results]
    for tr in top_trades:
        p_rid = tr["partner_roster_id"]
        p_idx = sim_ctx.roster_id_to_idx[p_rid]
        tr["persuasion"] = generate_trade_persuasion(
            my_team=my_team,
            partner_team=teams_by_rid[p_rid],
            my_base=base_a,
            partner_base=base_teams_by_idx[p_idx],
            send_a=tr["send_a"],
            send_b=tr["send_b"],
            tx=tr.pop("_tx"),
            perceived=tr["perceived"],
            impact_a=tr["impact_a"],
            impact_b=tr["impact_b"],
            league_info=league_info,
            ros_totals=ros_totals,
            players_master=players_master,
            starter_slots=sim_ctx.starter_slots,
        )

    t3 = time.perf_counter()

    return {
        "trades": top_trades,
        "telemetry": {
            "stage1_combinations_checked": stage1_total_examined,
            "stage1_value_band_survivors": len(stage1_candidates),
            "stage2_analytical_survivors": len(stage2_scored),
            "stage3_mc_evaluated": len(selected_for_mc),
            "stage3_mutual_verified": len(verified_results),
            "stage1_ms": round((t1 - t0) * 1000.0, 1),
            "stage2_ms": round((t2 - t1) * 1000.0, 1),
            "stage3_ms": round((t3 - t2) * 1000.0, 1),
            "total_ms": round((t3 - t0) * 1000.0, 1),
        },
    }


def evaluate_custom_trade(
    session: Dict[str, Any],
    roster_a_id: int,
    roster_b_id: int,
    send_a_ids: List[str],
    send_b_ids: List[str],
) -> Dict[str, Any]:
    """Evaluate any user-constructed trade with full 5,000-sim CRN weekly matchup breakdown."""
    league_info = session["league_info"]
    players_master = session["players_master"]
    fc_values = session["fc_values"]
    waiver_cal = session["waiver_calibration"]
    ros_totals = session["ros_totals"]
    sim_ctx: SimulationContext = session["sim_ctx"]
    baseline_matrix: np.ndarray = session["baseline_team_matrix"]
    baseline_sim: Dict[str, Any] = session["baseline_sim"]

    teams_by_rid = {int(t["roster_id"]): t for t in league_info["teams"]}
    if roster_a_id not in teams_by_rid or roster_b_id not in teams_by_rid:
        raise ValueError("Invalid roster IDs supplied.")
    if roster_a_id == roster_b_id:
        raise ValueError("Cannot trade with the same team.")

    team_a = teams_by_rid[roster_a_id]
    team_b = teams_by_rid[roster_b_id]
    idx_a = sim_ctx.roster_id_to_idx[roster_a_id]
    idx_b = sim_ctx.roster_id_to_idx[roster_b_id]

    perceived = evaluate_perceived_trade(
        send_a_ids, send_b_ids, fc_values, waiver_cal, ros_totals
    )
    tx = resolve_imbalanced_trade_rosters(
        team_a["players"],
        team_b["players"],
        send_a_ids,
        send_b_ids,
        set(team_a.get("reserve") or []),
        set(team_b.get("reserve") or []),
        players_master,
        fc_values,
        ros_totals,
        waiver_cal["waiver_pool"],
        sim_ctx.starter_slots,
    )

    base_prof_a = session["baseline_profiles"][idx_a]
    base_prof_b = session["baseline_profiles"][idx_b]
    post_prof_a = compute_team_weekly_profile(
        tx["final_a_ids"], tx["reserve_a"], sim_ctx, include_lineups=True
    )
    post_prof_b = compute_team_weekly_profile(
        tx["final_b_ids"], tx["reserve_b"], sim_ctx, include_lineups=True
    )

    post_sim = fast_reevaluate_trade(
        baseline_matrix,
        idx_a,
        post_prof_a["scores"],
        idx_b,
        post_prof_b["scores"],
        sim_ctx,
        track_weekly_for_both=True,
    )

    base_by_idx = {t["team_idx"]: t for t in baseline_sim["teams"]}
    post_by_idx = {t["team_idx"]: t for t in post_sim["teams"]}
    ba, pa = base_by_idx[idx_a], post_by_idx[idx_a]
    bb, pb = base_by_idx[idx_b], post_by_idx[idx_b]

    def _build_weekly_rows(
        t_idx: int,
        base_prof: Dict[str, Any],
        post_prof: Dict[str, Any],
        base_t: Dict[str, Any],
        post_t: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for w_idx, w in enumerate(sim_ctx.sim_weeks):
            is_playoff = w >= sim_ctx.playoff_week_start
            base_pts = round(float(base_prof["weekly_mu"][w_idx]), 2)
            post_pts = round(float(post_prof["weekly_mu"][w_idx]), 2)
            if is_playoff:
                po_round_idx = sim_ctx.playoff_weeks.index(w) if w in sim_ctx.playoff_weeks else 0
                if po_round_idx == 0:
                    opp_name = "Quarterfinals (Rd 1)"
                    prob_label = "Adv to Semis"
                    base_h2h = base_t["semis_pct"]
                    post_h2h = post_t["semis_pct"]
                    prob2_label = "1st-Rd Bye"
                    base_med = base_t["bye_pct"]
                    post_med = post_t["bye_pct"]
                elif po_round_idx == 1:
                    opp_name = "Semifinals (Rd 2)"
                    prob_label = "Adv to Finals"
                    base_h2h = base_t["finals_pct"]
                    post_h2h = post_t["finals_pct"]
                    prob2_label = "In Semis"
                    base_med = base_t["semis_pct"]
                    post_med = post_t["semis_pct"]
                else:
                    opp_name = "Championship Finals"
                    prob_label = "Win Title"
                    base_h2h = base_t["champ_pct"]
                    post_h2h = post_t["champ_pct"]
                    prob2_label = "In Finals"
                    base_med = base_t["finals_pct"]
                    post_med = post_t["finals_pct"]
            else:
                opp_rid = baseline_sim["h2h_opponents"].get(t_idx, {}).get(w)
                opp_name = teams_by_rid[opp_rid]["team_name"] if opp_rid in teams_by_rid else "BYE"
                prob_label = "H2H Win"
                prob2_label = "Median Win"
                base_h2h = baseline_sim["h2h_win_probs"].get(t_idx, {}).get(w)
                post_h2h = post_sim["h2h_win_probs"].get(t_idx, {}).get(w)
                base_med = baseline_sim["med_win_probs"].get(t_idx, {}).get(w)
                post_med = post_sim["med_win_probs"].get(t_idx, {}).get(w)

            rows.append(
                {
                    "week": w,
                    "phase": "Playoff" if is_playoff else "Regular",
                    "opponent_name": opp_name,
                    "prob_label": prob_label,
                    "prob2_label": prob2_label,
                    "base_pts": base_pts,
                    "post_pts": post_pts,
                    "d_pts": round(post_pts - base_pts, 2),
                    "base_h2h_pct": base_h2h,
                    "post_h2h_pct": post_h2h,
                    "d_h2h_pct": round(post_h2h - base_h2h, 2) if base_h2h is not None and post_h2h is not None else None,
                    "base_med_pct": base_med,
                    "post_med_pct": post_med,
                    "d_med_pct": round(post_med - base_med, 2) if base_med is not None and post_med is not None else None,
                    "post_starters": [
                        {
                            **st,
                            "full_name": (players_master.get(st["player_id"]) or {}).get("full_name", st["player_id"]),
                        }
                        for st in (post_prof["lineups_by_week"].get(w, {}).get("starters") or [])
                    ],
                }
            )
        return rows

    base_ppg_a = float(np.mean(base_prof_a["weekly_mu"])) if len(base_prof_a["weekly_mu"]) else 0.0
    post_ppg_a = float(np.mean(post_prof_a["weekly_mu"])) if len(post_prof_a["weekly_mu"]) else 0.0
    base_ppg_b = float(np.mean(base_prof_b["weekly_mu"])) if len(base_prof_b["weekly_mu"]) else 0.0
    post_ppg_b = float(np.mean(post_prof_b["weekly_mu"])) if len(post_prof_b["weekly_mu"]) else 0.0

    base_full_a = solve_full_strength_ppg(
        team_a["players"], ros_totals, players_master, sim_ctx.starter_slots, set(team_a.get("reserve") or [])
    )
    post_full_a = solve_full_strength_ppg(
        tx["final_a_ids"], ros_totals, players_master, sim_ctx.starter_slots, tx["reserve_a"]
    )
    base_full_b = solve_full_strength_ppg(
        team_b["players"], ros_totals, players_master, sim_ctx.starter_slots, set(team_b.get("reserve") or [])
    )
    post_full_b = solve_full_strength_ppg(
        tx["final_b_ids"], ros_totals, players_master, sim_ctx.starter_slots, tx["reserve_b"]
    )

    send_a_brief = [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in send_a_ids]
    send_b_brief = [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in send_b_ids]
    team_a_res = {
        "roster_id": roster_a_id,
        "team_name": team_a["team_name"],
        "base": ba,
        "post": pa,
        "d_champ_pct": round(pa["champ_pct"] - ba["champ_pct"], 2),
        "d_finals_pct": round(pa["finals_pct"] - ba["finals_pct"], 2),
        "d_semis_pct": round(pa["semis_pct"] - ba["semis_pct"], 2),
        "d_bye_pct": round(pa["bye_pct"] - ba["bye_pct"], 2),
        "d_playoff_pct": round(pa["playoff_pct"] - ba["playoff_pct"], 2),
        "d_exp_wins": round(pa["exp_wins"] - ba["exp_wins"], 2),
        "d_starter_ppg": round(post_ppg_a - base_ppg_a, 2),
        "d_full_ppg": round(post_full_a - base_full_a, 2),
        "base_champ_pct": ba["champ_pct"],
        "post_champ_pct": pa["champ_pct"],
        "base_finals_pct": ba["finals_pct"],
        "post_finals_pct": pa["finals_pct"],
        "base_semis_pct": ba["semis_pct"],
        "post_semis_pct": pa["semis_pct"],
        "base_bye_pct": ba["bye_pct"],
        "post_bye_pct": pa["bye_pct"],
        "base_playoff_pct": ba["playoff_pct"],
        "post_playoff_pct": pa["playoff_pct"],
        "base_exp_wins": ba["exp_wins"],
        "post_exp_wins": pa["exp_wins"],
        "weekly": _build_weekly_rows(idx_a, base_prof_a, post_prof_a, ba, pa),
    }
    team_b_res = {
        "roster_id": roster_b_id,
        "team_name": team_b["team_name"],
        "base": bb,
        "post": pb,
        "d_champ_pct": round(pb["champ_pct"] - bb["champ_pct"], 2),
        "d_finals_pct": round(pb["finals_pct"] - bb["finals_pct"], 2),
        "d_semis_pct": round(pb["semis_pct"] - bb["semis_pct"], 2),
        "d_bye_pct": round(pb["bye_pct"] - bb["bye_pct"], 2),
        "d_playoff_pct": round(pb["playoff_pct"] - bb["playoff_pct"], 2),
        "d_exp_wins": round(pb["exp_wins"] - bb["exp_wins"], 2),
        "d_starter_ppg": round(post_ppg_b - base_ppg_b, 2),
        "d_full_ppg": round(post_full_b - base_full_b, 2),
        "base_champ_pct": bb["champ_pct"],
        "post_champ_pct": pb["champ_pct"],
        "base_finals_pct": bb["finals_pct"],
        "post_finals_pct": pb["finals_pct"],
        "base_semis_pct": bb["semis_pct"],
        "post_semis_pct": pb["semis_pct"],
        "base_bye_pct": bb["bye_pct"],
        "post_bye_pct": pb["bye_pct"],
        "base_playoff_pct": bb["playoff_pct"],
        "post_playoff_pct": pb["playoff_pct"],
        "base_exp_wins": bb["exp_wins"],
        "post_exp_wins": pb["exp_wins"],
        "weekly": _build_weekly_rows(idx_b, base_prof_b, post_prof_b, bb, pb),
    }

    persuasion = generate_trade_persuasion(
        my_team=team_a,
        partner_team=team_b,
        my_base=ba,
        partner_base=bb,
        send_a=send_a_brief,
        send_b=send_b_brief,
        tx=tx,
        perceived=perceived,
        impact_a=team_a_res,
        impact_b=team_b_res,
        league_info=league_info,
        ros_totals=ros_totals,
        players_master=players_master,
        starter_slots=sim_ctx.starter_slots,
    )

    return {
        "structure": f"{len(send_a_ids)}-for-{len(send_b_ids)}",
        "invalid_padding": tx["invalid_padding"],
        "team_a": team_a_res,
        "team_b": team_b_res,
        "send_a": send_a_brief,
        "send_b": send_b_brief,
        "drops_a": [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in tx["drops_a"]],
        "adds_a": [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in tx["adds_a"]],
        "drops_b": [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in tx["drops_b"]],
        "adds_b": [_format_player_brief(pid, players_master, fc_values, ros_totals) for pid in tx["adds_b"]],
        "perceived": perceived,
        "persuasion": persuasion,
    }
