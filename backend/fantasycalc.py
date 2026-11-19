"""FantasyCalc redraft perceived trade values & league-calibrated waiver adjustment."""

from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Set

import httpx

from .cache import get_cached, get_cached_timestamp, set_cached

logger = logging.getLogger(__name__)

FANTASYCALC_URL = "https://api.fantasycalc.com/values/current"
TTL_FANTASYCALC = 3 * 3600  # Respect FantasyCalc's 3-hour caching guideline

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_name(name: str) -> str:
    s = name.lower().strip()
    s = _SUFFIX_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


def derive_fantasycalc_params(league_info: Dict[str, Any]) -> Dict[str, str]:
    """Map Sleeper league settings onto FantasyCalc query parameters."""
    roster_positions: List[str] = league_info.get("roster_positions") or []
    scoring_settings: Dict[str, float] = league_info.get("scoring_settings") or {}
    total_rosters = int(league_info.get("total_rosters") or 12)

    is_superflex = ("SUPER_FLEX" in roster_positions) or (roster_positions.count("QB") >= 2)
    num_qbs = "2" if is_superflex else "1"

    supported_sizes = [8, 10, 12, 14]
    closest_teams = min(supported_sizes, key=lambda n: abs(n - total_rosters))

    rec_val = float(scoring_settings.get("rec", 1.0))
    ppr_choices = [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]
    closest_ppr = min(ppr_choices, key=lambda item: abs(item[0] - rec_val))[1]

    te_bonus = float(scoring_settings.get("bonus_rec_te", 0.0))
    if te_bonus >= 0.99:
        tep = "te++"
    elif te_bonus > 0.01:
        tep = "te+"
    else:
        tep = "none"

    return {
        "isDynasty": "false",
        "numQbs": num_qbs,
        "numTeams": str(closest_teams),
        "ppr": closest_ppr,
        "tep": tep,
    }


def fetch_fantasycalc_values(
    league_info: Dict[str, Any],
    players_master: Dict[str, Dict[str, Any]],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch redraft trade values from FantasyCalc (cached >= 3 hours) and map to Sleeper IDs."""
    params = derive_fantasycalc_params(league_info)
    cache_key = (
        f"fantasycalc:values:dyn={params['isDynasty']}:"
        f"qbs={params['numQbs']}:teams={params['numTeams']}:"
        f"ppr={params['ppr']}:tep={params['tep']}"
    )

    # Always respect FantasyCalc's 3-hour refresh limit even on manual sync
    raw = get_cached(cache_key, TTL_FANTASYCALC)
    if raw is None:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(
                FANTASYCALC_URL,
                params=params,
                headers={"User-Agent": "SleeperTradeSim/1.0"},
            )
            resp.raise_for_status()
            raw = resp.json()
        if isinstance(raw, list) and raw:
            set_cached(cache_key, raw)

    fetched_at = get_cached_timestamp(cache_key)

    # Build normalized fallback lookup from players_master
    name_pos_to_pid: Dict[tuple, str] = {}
    name_to_pid: Dict[str, str] = {}
    for pid, pinfo in players_master.items():
        pos = str(pinfo.get("position") or "").upper()
        norm = _normalize_name(str(pinfo.get("full_name") or ""))
        if not norm:
            continue
        name_pos_to_pid.setdefault((norm, pos), pid)
        name_to_pid.setdefault(norm, pid)

    values_by_pid: Dict[str, Dict[str, Any]] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        player_obj = entry.get("player") or {}
        sleeper_id = player_obj.get("sleeperId")
        pid = str(sleeper_id) if sleeper_id not in (None, "", "0") else ""

        if not pid or pid not in players_master:
            fc_name = str(player_obj.get("name") or "")
            fc_pos = str(player_obj.get("position") or "").upper()
            norm = _normalize_name(fc_name)
            pid = name_pos_to_pid.get((norm, fc_pos)) or name_to_pid.get(norm) or pid

        if not pid:
            continue

        values_by_pid[pid] = {
            "player_id": pid,
            "fc_value": float(entry.get("value") or 0.0),
            "overall_rank": int(entry.get("overallRank") or 999),
            "pos_rank": int(entry.get("positionRank") or 999),
            "trend_30d": float(entry.get("trend30Day") or 0.0),
            "redraft_value": float(entry.get("redraftValue") or entry.get("value") or 0.0),
            "fc_name": str(player_obj.get("name") or ""),
            "fc_pos": str(player_obj.get("position") or ""),
        }

    return {
        "params": params,
        "fetched_at": fetched_at,
        "values": values_by_pid,
    }


def compute_waiver_calibration(
    teams: List[Dict[str, Any]],
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
    weekly_projections: Dict[str, Dict[int, float]],
    sim_weeks: List[int],
) -> Dict[str, Any]:
    """Calibrate league-specific waiver wire baselines and escalating roster-spot adjustments."""
    rostered_ids: Set[str] = set()
    for t in teams:
        for pid in t.get("players") or []:
            rostered_ids.add(str(pid))

    skill_positions = {"QB", "RB", "WR", "TE"}
    all_positions = ["QB", "RB", "WR", "TE", "K", "DEF"]

    # Build unrostered skill waiver pool (VORP-adjusted so QBs don't crowd out RB/WR depth)
    pos_baseline_ppg = {"QB": 15.0, "RB": 7.0, "WR": 7.5, "TE": 6.0}
    min_ros_threshold = max(8.0, 2.2 * max(1, len(sim_weeks)))
    by_pos: Dict[str, List[Dict[str, Any]]] = {p: [] for p in skill_positions}
    for pid, pinfo in players_master.items():
        if pid in rostered_ids:
            continue
        pos = str(pinfo.get("position") or "")
        if pos not in skill_positions:
            continue
        ros = ros_totals.get(pid)
        if not ros:
            continue
        total_pts = float(ros.get("total_pts") or 0.0)
        ppg = float(ros.get("ppg") or 0.0)
        fc = float((fc_values.get(pid) or {}).get("fc_value") or 0.0)
        if total_pts < min_ros_threshold and fc < 150.0:
            continue
        vorp_ppg = max(-2.0, ppg - pos_baseline_ppg.get(pos, 7.0))
        composite = vorp_ppg * 140.0 + fc * 0.85 + ppg * 15.0
        by_pos[pos].append(
            {
                "player_id": pid,
                "full_name": pinfo.get("full_name", pid),
                "position": pos,
                "team": pinfo.get("team", "FA"),
                "injury_status": pinfo.get("injury_status", ""),
                "ros_pts": round(total_pts, 1),
                "ppg": round(ppg, 2),
                "fc_value": round(fc, 0),
                "composite": round(composite, 1),
            }
        )

    waiver_candidates: List[Dict[str, Any]] = []
    for pos, items in by_pos.items():
        items.sort(key=lambda item: item["composite"], reverse=True)
        waiver_candidates.extend(items[:25])

    waiver_candidates.sort(key=lambda item: item["composite"], reverse=True)
    waiver_pool = waiver_candidates[:100]

    # Compute weekly streaming baseline per position (best available unrostered player in week w)
    streaming_baselines: Dict[str, Dict[int, float]] = {pos: {} for pos in all_positions}
    unrostered_by_pos: Dict[str, List[str]] = {pos: [] for pos in all_positions}
    for pid, pinfo in players_master.items():
        if pid in rostered_ids:
            continue
        pos = str(pinfo.get("position") or "")
        if pos in unrostered_by_pos and pid in weekly_projections:
            unrostered_by_pos[pos].append(pid)

    for pos in all_positions:
        pids_for_pos = unrostered_by_pos[pos]
        for w in sim_weeks:
            vals: List[float] = []
            for pid in pids_for_pos:
                pts = weekly_projections.get(pid, {}).get(w, 0.0)
                if pts > 0.0:
                    vals.append(pts)
            vals.sort(reverse=True)
            best_w = vals[min(1, len(vals) - 1)] if vals else 0.0
            if best_w <= 0.0:
                # Sensible positional floor if projection data lacks unrostered depth for a week
                best_w = {"QB": 12.0, "RB": 6.5, "WR": 6.5, "TE": 5.5, "K": 7.0, "DEF": 6.5}[pos]
            streaming_baselines[pos][w] = round(best_w, 2)

    # Calibrate FantasyCalc-style escalating waiver adjustment from actual 12-team bench depth
    worst_1_vals: List[float] = []
    worst_2_vals: List[float] = []
    for t in teams:
        reserve_set = set(t.get("reserve") or [])
        droppable_vals: List[float] = []
        for pid in t.get("players") or []:
            if pid in reserve_set:
                continue
            pos = str((players_master.get(pid) or {}).get("position") or "")
            if pos not in skill_positions:
                continue
            fc = float((fc_values.get(pid) or {}).get("fc_value") or 0.0)
            ros_ppg = float((ros_totals.get(pid) or {}).get("ppg") or 0.0)
            # If FC has 0 for a deep sleeper, estimate subtle bench floor from ROS PPG
            eff_fc = fc if fc > 0.0 else round(ros_ppg * 22.0, 1)
            droppable_vals.append(eff_fc)

        droppable_vals.sort()
        if len(droppable_vals) >= 1:
            worst_1_vals.append(droppable_vals[0])
        if len(droppable_vals) >= 2:
            worst_2_vals.append(droppable_vals[1])

    b1 = statistics.median(worst_1_vals) if worst_1_vals else 380.0
    b2 = statistics.median(worst_2_vals) if worst_2_vals else 520.0

    top_waiver_fc = sorted(
        [w["fc_value"] for w in waiver_pool if w["fc_value"] > 0],
        reverse=True,
    )[:5]
    u1 = statistics.mean(top_waiver_fc) if top_waiver_fc else 350.0

    waiver_adj_1 = float(round(max(300.0, min(850.0, 0.55 * b1 + 0.45 * u1))))
    marginal_2 = float(
        round(
            max(
                375.0,
                waiver_adj_1 * 1.08,
                min(1050.0, 1.28 * waiver_adj_1, 0.65 * b2 + 0.35 * u1),
            )
        )
    )
    waiver_adj_2 = float(round(waiver_adj_1 + marginal_2))

    return {
        "waiver_adj_1": waiver_adj_1,
        "waiver_adj_2": waiver_adj_2,
        "marginal_2": marginal_2,
        "median_bench_drop_1": round(b1, 1),
        "median_bench_drop_2": round(b2, 1),
        "top_waiver_avg_fc": round(u1, 1),
        "waiver_pool": waiver_pool,
        "streaming_baselines": streaming_baselines,
    }


def get_player_fc_value(pid: str, fc_values: Dict[str, Dict[str, Any]], ros_totals: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
    """Return FantasyCalc perceived value for a player, with safe projection fallback if unranked."""
    entry = fc_values.get(pid)
    if entry and entry.get("fc_value", 0.0) > 0:
        return float(entry["fc_value"])
    if ros_totals and pid in ros_totals:
        ppg = float(ros_totals[pid].get("ppg") or 0.0)
        return round(max(0.0, ppg * 20.0), 1)
    return 0.0


def evaluate_perceived_trade(
    pkg_a_ids: List[str],
    pkg_b_ids: List[str],
    fc_values: Dict[str, Dict[str, Any]],
    waiver_calibration: Dict[str, Any],
    ros_totals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute FantasyCalc perceived trade balance including imbalanced-package waiver adjustment."""
    raw_a = sum(get_player_fc_value(pid, fc_values, ros_totals) for pid in pkg_a_ids)
    raw_b = sum(get_player_fc_value(pid, fc_values, ros_totals) for pid in pkg_b_ids)

    spot_diff = len(pkg_a_ids) - len(pkg_b_ids)
    abs_diff = abs(spot_diff)
    if abs_diff == 0:
        waiver_adj = 0.0
    elif abs_diff == 1:
        waiver_adj = float(waiver_calibration["waiver_adj_1"])
    else:
        waiver_adj = float(waiver_calibration["waiver_adj_2"])

    # Side with MORE outgoing players opens roster spot(s) and gets waiver pickup value;
    # equivalently, we add waiver_adj to the smaller outgoing package to balance roster spots.
    if spot_diff > 0:
        waiver_beneficiary_side = "A"
        smaller_package_side = "B"
        adjusted_a = raw_a
        adjusted_b = raw_b + waiver_adj
    elif spot_diff < 0:
        waiver_beneficiary_side = "B"
        smaller_package_side = "A"
        adjusted_a = raw_a + waiver_adj
        adjusted_b = raw_b
    else:
        waiver_beneficiary_side = None
        smaller_package_side = None
        adjusted_a = raw_a
        adjusted_b = raw_b

    # Partner (Side B) receives adjusted_a and gives adjusted_b
    partner_net_gain = adjusted_a - adjusted_b
    denom = max(adjusted_b, 500.0)
    partner_gain_pct = 100.0 * partner_net_gain / denom

    if -5.0 <= partner_gain_pct <= 5.0:
        fairness_label = "Fair / Dead Even"
        fairness_tier = "fair"
    elif 5.0 < partner_gain_pct <= 15.0:
        fairness_label = "Slight Partner Edge (High Acceptance)"
        fairness_tier = "high_accept"
    elif partner_gain_pct > 15.0:
        fairness_label = "Strong Partner Overpay (Very Likely)"
        fairness_tier = "overpay"
    elif -12.0 <= partner_gain_pct < -5.0:
        fairness_label = "Slight Underpay (Negotiable)"
        fairness_tier = "negotiable"
    else:
        fairness_label = "Lopsided Against Partner"
        fairness_tier = "lopsided"

    return {
        "raw_a": round(raw_a, 1),
        "raw_b": round(raw_b, 1),
        "spot_diff": spot_diff,
        "extra_spots": abs_diff,
        "waiver_adj_value": round(waiver_adj, 1),
        "waiver_beneficiary_side": waiver_beneficiary_side,
        "smaller_package_side": smaller_package_side,
        "adjusted_a": round(adjusted_a, 1),
        "adjusted_b": round(adjusted_b, 1),
        "partner_net_gain": round(partner_net_gain, 1),
        "partner_gain_pct": round(partner_gain_pct, 1),
        "fairness_label": fairness_label,
        "fairness_tier": fairness_tier,
    }
