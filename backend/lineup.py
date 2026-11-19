"""Fast optimal weekly lineup solver and imbalanced-trade roster transaction resolver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

SLOT_ELIGIBILITY: Dict[str, Set[str]] = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "K": {"K"},
    "DEF": {"DEF"},
    "WRRB_FLEX": {"RB", "WR"},
    "REC_FLEX": {"WR", "TE"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}

# Narrowest slots filled first so flex spots absorb true marginal upside
SLOT_ORDER_PRIORITY: Dict[str, int] = {
    "QB": 0,
    "RB": 1,
    "WR": 2,
    "TE": 3,
    "K": 4,
    "DEF": 5,
    "WRRB_FLEX": 6,
    "REC_FLEX": 7,
    "FLEX": 8,
    "SUPER_FLEX": 9,
}

SKILL_POSITIONS: Set[str] = {"QB", "RB", "WR", "TE"}


def parse_starter_slots(roster_positions: List[str]) -> List[str]:
    """Extract active starter slots from Sleeper roster_positions (ignoring BN/IR)."""
    slots = [str(s) for s in roster_positions if s in SLOT_ELIGIBILITY]
    slots.sort(key=lambda s: SLOT_ORDER_PRIORITY.get(s, 99))
    return slots


def solve_weekly_lineup(
    active_player_ids: List[str],
    week: int,
    weekly_projections: Dict[str, Dict[int, float]],
    players_master: Dict[str, Dict[str, Any]],
    streaming_baselines: Dict[str, Dict[int, float]],
    starter_slots: List[str],
    reserve_ids: Optional[Set[str]] = None,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Solve optimal starting lineup for a single team in a given week."""
    # Allow returning IR players to start in future weeks where they have positive projections
    # (weeks they are out naturally project 0.0 and are excluded by pts > 0.0 below).
    _ = reserve_ids
    pool: List[Tuple[str, str, float]] = []
    for pid in active_player_ids:
        pinfo = players_master.get(pid)
        if not pinfo:
            continue
        pos = str(pinfo.get("position") or "")
        if pos not in {"QB", "RB", "WR", "TE", "K", "DEF"}:
            continue
        pts = float((weekly_projections.get(pid) or {}).get(week, 0.0))
        pool.append((pid, pos, pts))

    # Sort descending by weekly projected points
    pool.sort(key=lambda item: item[2], reverse=True)
    used: Set[str] = set()
    starters: List[Dict[str, Any]] = []
    total_pts = 0.0

    for slot in starter_slots:
        eligible = SLOT_ELIGIBILITY[slot]
        chosen: Optional[Tuple[str, str, float]] = None
        for pid, pos, pts in pool:
            if pid in used:
                continue
            if pos in eligible and pts > 0.0:
                chosen = (pid, pos, pts)
                break

        if chosen is not None:
            pid, pos, pts = chosen
            used.add(pid)
            total_pts += pts
            starters.append(
                {
                    "slot": slot,
                    "player_id": pid,
                    "position": pos,
                    "points": round(pts, 2),
                    "is_streamer": False,
                }
            )
        else:
            # Bye-week / empty-slot waiver streamer fallback (90% of best unrostered player)
            best_pos = "FLEX"
            best_stream = 0.0
            for pos_opt in sorted(eligible):
                val = float((streaming_baselines.get(pos_opt) or {}).get(week, 0.0)) * 0.90
                if val > best_stream:
                    best_stream = val
                    best_pos = pos_opt
            total_pts += best_stream
            starters.append(
                {
                    "slot": slot,
                    "player_id": f"STREAM:{best_pos}",
                    "position": best_pos,
                    "points": round(best_stream, 2),
                    "is_streamer": True,
                }
            )

    bench: List[Dict[str, Any]] = []
    for pid, pos, pts in pool:
        if pid not in used:
            bench.append({"player_id": pid, "position": pos, "points": round(pts, 2)})

    return round(total_pts, 3), starters, bench


def solve_full_strength_ppg(
    active_player_ids: List[str],
    ros_totals: Dict[str, Dict[str, Any]],
    players_master: Dict[str, Dict[str, Any]],
    starter_slots: List[str],
    reserve_ids: Optional[Set[str]] = None,
) -> float:
    """Solve optimal full-strength (bye-neutralized) starting lineup PPG using ROS active-game PPG."""
    synth_proj: Dict[str, Dict[int, float]] = {
        pid: {0: float((ros_totals.get(pid) or {}).get("ppg") or 0.0)}
        for pid in active_player_ids
    }
    synth_stream: Dict[str, Dict[int, float]] = {
        "QB": {0: 15.0},
        "RB": {0: 7.0},
        "WR": {0: 7.5},
        "TE": {0: 6.0},
        "K": {0: 7.8},
        "DEF": {0: 6.8},
    }
    total_pts, _, _ = solve_weekly_lineup(
        active_player_ids,
        0,
        synth_proj,
        players_master,
        synth_stream,
        starter_slots,
        reserve_ids=reserve_ids,
    )
    return round(total_pts, 3)


def _player_retention_score(
    pid: str,
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
) -> float:
    ppg = float((ros_totals.get(pid) or {}).get("ppg") or 0.0)
    fc = float((fc_values.get(pid) or {}).get("fc_value") or 0.0)
    return 0.70 * (ppg / 20.0) + 0.30 * (fc / 6000.0)


def _choose_drops(
    roster_ids: Set[str],
    reserve_ids: Set[str],
    protected_incoming: Set[str],
    count: int,
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Select `count` lowest-retention droppable bench players, protecting IR, incoming, and sole K/DEF."""
    remaining = [pid for pid in roster_ids if pid not in reserve_ids]
    pos_counts: Dict[str, int] = {}
    for pid in remaining:
        pos = str((players_master.get(pid) or {}).get("position") or "")
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    drops: List[str] = []
    for _ in range(count):
        best_pid: Optional[str] = None
        best_key: Optional[Tuple[int, float]] = None
        for pid in remaining:
            if pid in protected_incoming:
                continue
            pos = str((players_master.get(pid) or {}).get("position") or "")
            score = _player_retention_score(pid, players_master, fc_values, ros_totals)
            is_sole_k_def = pos in {"K", "DEF"} and pos_counts.get(pos, 0) <= 1
            if pos in SKILL_POSITIONS or not is_sole_k_def:
                adj_score = score - (0.35 if pos in {"K", "DEF"} else 0.0)
                cand_key = (0, adj_score)
            else:
                cand_key = (1, score + 5.0)
            if best_key is None or cand_key < best_key:
                best_key = cand_key
                best_pid = pid

        if best_pid is None:
            break
        drops.append(best_pid)
        remaining.remove(best_pid)
        dropped_pos = str((players_master.get(best_pid) or {}).get("position") or "")
        pos_counts[dropped_pos] = max(0, pos_counts.get(dropped_pos, 0) - 1)

    return drops


def _choose_waiver_adds(
    roster_ids: Set[str],
    other_roster_ids: Set[str],
    count: int,
    waiver_pool: List[Dict[str, Any]],
    players_master: Dict[str, Dict[str, Any]],
    starter_slots: List[str],
) -> List[str]:
    """Pick the best `count` available waiver players tailored to the team's positional needs."""
    if count <= 0:
        return []
    excluded = roster_ids | other_roster_ids

    pos_counts: Dict[str, int] = {}
    for pid in roster_ids:
        pos = str((players_master.get(pid) or {}).get("position") or "")
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    qb_slots = starter_slots.count("QB") + starter_slots.count("SUPER_FLEX")
    te_slots = starter_slots.count("TE")
    need_bonus: Dict[str, float] = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0}
    if pos_counts.get("QB", 0) < max(1, starter_slots.count("QB")):
        need_bonus["QB"] = 600.0
    elif pos_counts.get("QB", 0) >= qb_slots + 1:
        need_bonus["QB"] = -700.0
    if pos_counts.get("TE", 0) < max(1, te_slots):
        need_bonus["TE"] = 450.0
    elif pos_counts.get("TE", 0) >= te_slots + 1:
        need_bonus["TE"] = -450.0
    if pos_counts.get("RB", 0) < starter_slots.count("RB") + 1:
        need_bonus["RB"] = 250.0
    if pos_counts.get("WR", 0) < starter_slots.count("WR") + 1:
        need_bonus["WR"] = 250.0

    scored: List[Tuple[float, str]] = []
    for w in waiver_pool:
        pid = w["player_id"]
        if pid in excluded:
            continue
        pos = w["position"]
        score = float(w["composite"]) + need_bonus.get(pos, 0.0)
        scored.append((score, pid))

    scored.sort(key=lambda item: item[0], reverse=True)
    adds: List[str] = []
    for _, pid in scored:
        adds.append(pid)
        if len(adds) >= count:
            break
    return adds


def resolve_imbalanced_trade_rosters(
    roster_a_ids: List[str],
    roster_b_ids: List[str],
    pkg_a_ids: List[str],
    pkg_b_ids: List[str],
    reserve_a: Set[str],
    reserve_b: Set[str],
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
    waiver_pool: List[Dict[str, Any]],
    starter_slots: List[str],
) -> Dict[str, Any]:
    """Apply a trade to both rosters, resolving bench drops and waiver additions for imbalanced deals."""
    set_a = set(roster_a_ids)
    set_b = set(roster_b_ids)
    send_a = set(pkg_a_ids)
    send_b = set(pkg_b_ids)

    new_a = (set_a - send_a) | send_b
    new_b = (set_b - send_b) | send_a

    # Keep reserve sets consistent if an IR player were ever moved (they become active on new roster)
    eff_res_a = reserve_a & (set_a - send_a)
    eff_res_b = reserve_b & (set_b - send_b)

    drops_a: List[str] = []
    adds_a: List[str] = []
    drops_b: List[str] = []
    adds_b: List[str] = []
    invalid_padding = False

    spot_diff = len(pkg_a_ids) - len(pkg_b_ids)
    if spot_diff > 0:
        # Team A sent more players -> Team B drops spot_diff, Team A adds spot_diff
        drops_b = _choose_drops(
            new_b, eff_res_b, send_a, spot_diff, players_master, fc_values, ros_totals
        )
        new_b -= set(drops_b)

        # Reject junk padding where an incoming skill player is worse than Team B's worst skill drop
        skill_drops_b = [
            pid for pid in drops_b
            if str((players_master.get(pid) or {}).get("position") or "") in SKILL_POSITIONS
        ]
        skill_in_a = [
            pid for pid in pkg_a_ids
            if str((players_master.get(pid) or {}).get("position") or "") in SKILL_POSITIONS
        ]
        if skill_drops_b and skill_in_a:
            max_drop_score = max(
                _player_retention_score(pid, players_master, fc_values, ros_totals)
                for pid in skill_drops_b
            )
            min_in_score = min(
                _player_retention_score(pid, players_master, fc_values, ros_totals)
                for pid in skill_in_a
            )
            if min_in_score < max_drop_score - 0.02:
                invalid_padding = True

        adds_a = _choose_waiver_adds(
            new_a, new_b, spot_diff, waiver_pool, players_master, starter_slots
        )
        new_a |= set(adds_a)

    elif spot_diff < 0:
        need = abs(spot_diff)
        # Team B sent more players -> Team A drops `need`, Team B adds `need`
        drops_a = _choose_drops(
            new_a, eff_res_a, send_b, need, players_master, fc_values, ros_totals
        )
        new_a -= set(drops_a)

        skill_drops_a = [
            pid for pid in drops_a
            if str((players_master.get(pid) or {}).get("position") or "") in SKILL_POSITIONS
        ]
        skill_in_b = [
            pid for pid in pkg_b_ids
            if str((players_master.get(pid) or {}).get("position") or "") in SKILL_POSITIONS
        ]
        if skill_drops_a and skill_in_b:
            max_drop_score = max(
                _player_retention_score(pid, players_master, fc_values, ros_totals)
                for pid in skill_drops_a
            )
            min_in_score = min(
                _player_retention_score(pid, players_master, fc_values, ros_totals)
                for pid in skill_in_b
            )
            if min_in_score < max_drop_score - 0.02:
                invalid_padding = True

        adds_b = _choose_waiver_adds(
            new_b, new_a, need, waiver_pool, players_master, starter_slots
        )
        new_b |= set(adds_b)

    return {
        "final_a_ids": sorted(new_a),
        "final_b_ids": sorted(new_b),
        "reserve_a": eff_res_a,
        "reserve_b": eff_res_b,
        "drops_a": drops_a,
        "adds_a": adds_a,
        "drops_b": drops_b,
        "adds_b": adds_b,
        "invalid_padding": invalid_padding,
    }
