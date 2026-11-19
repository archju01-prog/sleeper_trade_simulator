"""Multi-user Tornado HTTP server for the Public Sleeper Redraft Trade Simulator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tornado.ioloop
import tornado.web

from .fantasycalc import (
    compute_waiver_calibration,
    fetch_fantasycalc_values,
    get_player_fc_value,
)
from .simulator import (
    SimulationContext,
    compute_team_weekly_profile,
    run_league_simulation,
)
from .sleeper import (
    fetch_custom_projections,
    fetch_league_data,
    fetch_players_master,
    lookup_user_leagues,
)
from .trade_finder import (
    _analytical_team_metrics,
    _compute_deterministic_profile,
    evaluate_custom_trade,
    find_viable_trades,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(os.path.abspath(__file__)).parent.parent / "frontend"

MAX_CACHED_SESSIONS = 24
_SESSION_LOCK = threading.Lock()
_SESSION_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_DEFAULT_SESSION_KEY: Optional[str] = None
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _compute_team_positional_strength(
    lineups_by_week: Dict[int, Dict[str, Any]],
    sim_weeks: List[int],
) -> Dict[str, float]:
    """Average projected starter points per position slot across remaining weeks."""
    buckets: Dict[str, List[float]] = {
        "QB": [],
        "RB": [],
        "WR": [],
        "TE": [],
        "FLEX": [],
        "K": [],
        "DEF": [],
    }
    for w in sim_weeks:
        wk_data = lineups_by_week.get(w) or {}
        starters = wk_data.get("starters") or []
        wk_sums: Dict[str, float] = {k: 0.0 for k in buckets}
        wk_counts: Dict[str, int] = {k: 0 for k in buckets}
        for st in starters:
            slot = st["slot"]
            pts = float(st["points"])
            if slot in ("FLEX", "SUPER_FLEX", "WRRB_FLEX", "REC_FLEX"):
                wk_sums["FLEX"] += pts
                wk_counts["FLEX"] += 1
            elif slot in wk_sums:
                wk_sums[slot] += pts
                wk_counts[slot] += 1
        for k in buckets:
            if wk_counts[k] > 0:
                buckets[k].append(wk_sums[k] / wk_counts[k])

    return {
        k: round(sum(v) / len(v), 2) if v else 0.0
        for k, v in buckets.items()
    }


def _build_arbitrage_table(
    league_info: Dict[str, Any],
    players_master: Dict[str, Dict[str, Any]],
    fc_values: Dict[str, Dict[str, Any]],
    ros_totals: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Sleeper custom-projection VORP vs FantasyCalc market value arbitrage rankings."""
    owner_by_pid: Dict[str, Dict[str, Any]] = {}
    for t in league_info["teams"]:
        for pid in t.get("players") or []:
            owner_by_pid[pid] = {
                "roster_id": t["roster_id"],
                "team_name": t["team_name"],
                "is_my_team": t["roster_id"] == league_info["my_roster_id"],
            }

    replacement_ppg = {"QB": 15.2, "RB": 7.6, "WR": 8.0, "TE": 6.2}
    rows: List[Dict[str, Any]] = []

    for pid, pinfo in players_master.items():
        pos = str(pinfo.get("position") or "")
        if pos not in replacement_ppg:
            continue
        ros = ros_totals.get(pid) or {}
        fc_entry = fc_values.get(pid) or {}
        fc_val = float(fc_entry.get("fc_value") or 0.0)
        ros_pts = float(ros.get("total_pts") or 0.0)
        ppg = float(ros.get("ppg") or 0.0)
        active_games = int(ros.get("active_games") or 0)

        if fc_val < 150.0 and ros_pts < 30.0:
            continue

        vorp_ppg = ppg - replacement_ppg[pos]
        vorp_total = max(0.0, vorp_ppg * max(1, active_games))
        rows.append(
            {
                "player_id": pid,
                "full_name": pinfo.get("full_name", pid),
                "position": pos,
                "team": pinfo.get("team", "FA"),
                "injury_status": pinfo.get("injury_status", ""),
                "ppg": round(ppg, 2),
                "ros_pts": round(ros_pts, 1),
                "playoff_pts": round(float(ros.get("playoff_pts") or 0.0), 1),
                "vorp_total": round(vorp_total, 1),
                "fc_value": round(fc_val, 0),
                "fc_rank": int(fc_entry.get("overall_rank") or 999),
                "owner": owner_by_pid.get(pid),
            }
        )

    rows.sort(key=lambda r: r["vorp_total"], reverse=True)
    for idx, r in enumerate(rows, start=1):
        r["sleeper_vorp_rank"] = idx

    fc_sorted = sorted(rows, key=lambda r: (r["fc_value"], -r["fc_rank"]), reverse=True)
    for idx, r in enumerate(fc_sorted, start=1):
        r["fc_subset_rank"] = idx
        r["rank_gap"] = idx - r["sleeper_vorp_rank"]

    rows.sort(key=lambda r: max(r["fc_value"], r["vorp_total"] * 45.0), reverse=True)
    return rows[:220]


def _make_session_key(
    league_id: str,
    user_key: str,
    start_week: Optional[int],
    use_median: bool,
    neutralize_byes: bool,
) -> str:
    wk_tag = str(start_week) if start_week is not None else "auto"
    return f"{league_id}:{user_key or 'anon'}:w{wk_tag}:m{1 if use_median else 0}:b{1 if neutralize_byes else 0}"


def _find_cached_session(
    league_id: str,
    user_key: str,
    start_week: Optional[int],
    use_median: Optional[bool],
    neutralize_byes: bool,
) -> Optional[Dict[str, Any]]:
    wk_tag = str(start_week) if start_week is not None else "auto"
    uk = (user_key or "").strip().lower()
    with _SESSION_LOCK:
        for key in reversed(list(_SESSION_CACHE.keys())):
            sess = _SESSION_CACHE[key]
            lg = sess["league_info"]
            if str(lg["league_id"]) != str(league_id):
                continue
            if uk and sess.get("user_key", "") != uk:
                continue
            if not uk and sess.get("user_key", "") not in ("", "anon"):
                continue
            if sess.get("start_week_tag", "auto") != wk_tag:
                continue
            if bool(lg.get("neutralize_byes", True)) != bool(neutralize_byes):
                continue
            if use_median is None:
                if sess.get("was_median_override", False):
                    continue
            elif bool(lg["use_median_scoring"]) != bool(use_median):
                continue
            _SESSION_CACHE.move_to_end(key)
            return sess
    return None


def _store_cached_session(session: Dict[str, Any]) -> None:
    skey = session["session_key"]
    with _SESSION_LOCK:
        _SESSION_CACHE[skey] = session
        _SESSION_CACHE.move_to_end(skey)
        while len(_SESSION_CACHE) > MAX_CACHED_SESSIONS:
            _SESSION_CACHE.popitem(last=False)


def build_league_session(
    league_id: Optional[str] = None,
    username: str = "",
    user_id: Optional[str] = None,
    target_team_name: str = "",
    force_refresh: bool = False,
    start_week_override: Optional[int] = None,
    use_median_override: Optional[bool] = None,
    neutralize_byes: bool = True,
) -> Dict[str, Any]:
    """Sync Sleeper league + FantasyCalc, run 5,000-sim CRN baseline, and cache session."""
    t0 = time.perf_counter()
    uname_clean = str(username or "").strip()
    lid_clean = str(league_id or "").strip() or None
    uid_clean = str(user_id or "").strip() or None
    available_leagues: List[Dict[str, Any]] = []

    if not lid_clean:
        if not uname_clean:
            raise ValueError("Please provide a Sleeper username or League ID.")
        lookup = lookup_user_leagues(uname_clean, force_refresh=force_refresh)
        available_leagues = lookup["leagues"]
        lid_clean = available_leagues[0]["league_id"]
        uid_clean = uid_clean or lookup.get("user_id")
        uname_clean = str(lookup.get("username") or uname_clean)
    elif uname_clean and not uname_clean.isdigit():
        try:
            lookup = lookup_user_leagues(uname_clean, force_refresh=False)
            available_leagues = lookup["leagues"]
            uid_clean = uid_clean or lookup.get("user_id")
            uname_clean = str(lookup.get("username") or uname_clean)
        except Exception:
            pass

    players_master = fetch_players_master(force_refresh=force_refresh)
    league_info = fetch_league_data(
        lid_clean,
        force_refresh=force_refresh,
        target_team_name=target_team_name,
        start_week_override=start_week_override,
        use_median_override=use_median_override,
        user_id=uid_clean,
        username=uname_clean,
    )
    league_info["platform"] = "sleeper"
    league_info["neutralize_byes"] = bool(neutralize_byes)

    proj = fetch_custom_projections(
        league_info["season"],
        league_info["current_week"],
        17,
        league_info["scoring_settings"],
        players_master,
        force_refresh=force_refresh,
    )

    fc = fetch_fantasycalc_values(league_info, players_master, force_refresh=force_refresh)
    fc_values = fc["values"]

    waiver_cal = compute_waiver_calibration(
        league_info["teams"],
        players_master,
        fc_values,
        proj["ros_totals"],
        proj["weekly_projections"],
        proj["weeks"],
    )

    sim_ctx = SimulationContext(
        league_info,
        players_master,
        proj["weekly_projections"],
        waiver_cal["streaming_baselines"],
        waiver_cal["waiver_pool"],
        num_sims=5000,
        seed=42,
        ros_totals=proj["ros_totals"],
    )

    num_teams = sim_ctx.num_teams
    num_weeks = len(sim_ctx.sim_weeks)
    baseline_team_matrix = np.zeros((sim_ctx.num_sims, num_weeks, num_teams), dtype=np.float32)
    baseline_mu = np.zeros((num_teams, num_weeks), dtype=np.float64)
    baseline_var = np.zeros((num_teams, num_weeks), dtype=np.float64)
    baseline_profiles: Dict[int, Dict[str, Any]] = {}
    baseline_det_profiles: Dict[int, Dict[str, Any]] = {}

    for idx, team in enumerate(league_info["teams"]):
        res_set = set(team.get("reserve") or [])
        prof = compute_team_weekly_profile(
            team["players"], res_set, sim_ctx, include_lineups=True
        )
        baseline_team_matrix[:, :, idx] = prof["scores"]
        baseline_mu[idx, :] = prof["weekly_mu"]
        baseline_var[idx, :] = prof["weekly_var"]
        baseline_profiles[idx] = prof
        baseline_det_profiles[idx] = _compute_deterministic_profile(
            team["players"], res_set, sim_ctx
        )

    baseline_med_mu = np.median(baseline_mu, axis=0)
    baseline_med_var = np.median(baseline_var, axis=0)

    baseline_det_metrics: Dict[int, Dict[str, float]] = {}
    for idx in range(num_teams):
        baseline_det_metrics[idx] = _analytical_team_metrics(
            idx,
            baseline_det_profiles[idx],
            baseline_mu,
            baseline_var,
            baseline_med_mu,
            baseline_med_var,
            sim_ctx,
        )

    baseline_sim = run_league_simulation(
        baseline_team_matrix,
        sim_ctx,
        track_weekly_for_indices=list(range(num_teams)),
    )

    pos_strength_by_idx: Dict[int, Dict[str, float]] = {}
    for idx in range(num_teams):
        pos_strength_by_idx[idx] = _compute_team_positional_strength(
            baseline_profiles[idx]["lineups_by_week"], sim_ctx.sim_weeks
        )

    pos_ranks_by_idx: Dict[int, Dict[str, int]] = {idx: {} for idx in range(num_teams)}
    for pos_key in ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]:
        ordered_indices = sorted(
            range(num_teams),
            key=lambda i: pos_strength_by_idx[i].get(pos_key, 0.0),
            reverse=True,
        )
        for rank, idx in enumerate(ordered_indices, start=1):
            pos_ranks_by_idx[idx][pos_key] = rank

    arbitrage_rows = _build_arbitrage_table(
        league_info, players_master, fc_values, proj["ros_totals"]
    )

    user_key = (uid_clean or uname_clean or "").strip().lower()
    session_key = _make_session_key(
        str(lid_clean),
        user_key,
        start_week_override,
        bool(league_info["use_median_scoring"]),
        bool(neutralize_byes),
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    logger.info(
        "Synced Sleeper league '%s' (%s) user='%s' week %s-17 in %.1f ms",
        league_info["league_name"],
        lid_clean,
        uname_clean or uid_clean or "direct",
        league_info["current_week"],
        elapsed_ms,
    )

    session = {
        "session_key": session_key,
        "user_key": user_key,
        "user_id": uid_clean,
        "username": uname_clean,
        "start_week_tag": str(start_week_override) if start_week_override is not None else "auto",
        "was_median_override": use_median_override is not None,
        "available_leagues": available_leagues,
        "league_info": league_info,
        "players_master": players_master,
        "weekly_projections": proj["weekly_projections"],
        "ros_totals": proj["ros_totals"],
        "fc_params": fc["params"],
        "fc_fetched_at": fc["fetched_at"],
        "fc_values": fc_values,
        "waiver_calibration": waiver_cal,
        "sim_ctx": sim_ctx,
        "baseline_team_matrix": baseline_team_matrix,
        "baseline_mu": baseline_mu,
        "baseline_var": baseline_var,
        "baseline_med_mu": baseline_med_mu,
        "baseline_med_var": baseline_med_var,
        "baseline_profiles": baseline_profiles,
        "baseline_det_metrics": baseline_det_metrics,
        "baseline_sim": baseline_sim,
        "pos_strength_by_idx": pos_strength_by_idx,
        "pos_ranks_by_idx": pos_ranks_by_idx,
        "arbitrage_rows": arbitrage_rows,
        "sync_ms": elapsed_ms,
    }
    _store_cached_session(session)
    return session


def _serialize_session_state(session: Dict[str, Any]) -> Dict[str, Any]:
    lg = session["league_info"]
    pm = session["players_master"]
    fc_values = session["fc_values"]
    ros_totals = session["ros_totals"]
    waiver_cal = session["waiver_calibration"]
    sim_ctx: SimulationContext = session["sim_ctx"]
    base_sim = session["baseline_sim"]
    base_by_idx = {t["team_idx"]: t for t in base_sim["teams"]}

    teams_out: List[Dict[str, Any]] = []
    for idx, t in enumerate(lg["teams"]):
        sim_t = base_by_idx[idx]
        res_set = set(t.get("reserve") or [])
        roster_details: List[Dict[str, Any]] = []
        for pid in t["players"]:
            pinfo = pm.get(pid) or {}
            ros = ros_totals.get(pid) or {}
            fc_entry = fc_values.get(pid) or {}
            roster_details.append(
                {
                    "player_id": pid,
                    "full_name": pinfo.get("full_name", pid),
                    "position": pinfo.get("position", "?"),
                    "team": pinfo.get("team", "FA"),
                    "injury_status": pinfo.get("injury_status", ""),
                    "is_ir": pid in res_set,
                    "ppg": round(float(ros.get("ppg") or 0.0), 2),
                    "ros_pts": round(float(ros.get("total_pts") or 0.0), 1),
                    "playoff_pts": round(float(ros.get("playoff_pts") or 0.0), 1),
                    "next_bye": ros.get("next_bye"),
                    "fc_value": round(get_player_fc_value(pid, fc_values, ros_totals), 0),
                    "overall_rank": int(fc_entry.get("overall_rank") or 999),
                }
            )
        roster_details.sort(key=lambda p: (p["is_ir"], -p["fc_value"], -p["ppg"]))

        prof = session["baseline_profiles"][idx]
        avg_wk_proj = round(float(np.mean(prof["weekly_mu"])), 1) if len(prof["weekly_mu"]) else 0.0

        weekly_sched: List[Dict[str, Any]] = []
        teams_by_rid = {int(x["roster_id"]): x for x in lg["teams"]}
        for w_idx, w in enumerate(sim_ctx.sim_weeks):
            opp_rid = base_sim["h2h_opponents"].get(idx, {}).get(w)
            opp_name = (
                teams_by_rid[opp_rid]["team_name"]
                if opp_rid in teams_by_rid
                else ("Playoffs" if w >= sim_ctx.playoff_week_start else "BYE")
            )
            weekly_sched.append(
                {
                    "week": w,
                    "opponent_name": opp_name,
                    "proj_pts": round(float(prof["weekly_mu"][w_idx]), 1),
                    "h2h_pct": base_sim["h2h_win_probs"].get(idx, {}).get(w),
                    "med_pct": base_sim["med_win_probs"].get(idx, {}).get(w),
                }
            )

        teams_out.append(
            {
                "roster_id": t["roster_id"],
                "team_name": t["team_name"],
                "manager_name": t["manager_name"],
                "is_my_team": t["roster_id"] == lg["my_roster_id"],
                "wins": t["wins"],
                "losses": t["losses"],
                "ties": t["ties"],
                "points_for": t["points_for"],
                "points_against": t["points_against"],
                "exp_wins": sim_t["exp_wins"],
                "exp_losses": sim_t["exp_losses"],
                "exp_pf": sim_t["exp_pf"],
                "playoff_pct": sim_t["playoff_pct"],
                "bye_pct": sim_t["bye_pct"],
                "semis_pct": sim_t["semis_pct"],
                "finals_pct": sim_t["finals_pct"],
                "champ_pct": sim_t["champ_pct"],
                "avg_weekly_proj": avg_wk_proj,
                "pos_strength": session["pos_strength_by_idx"][idx],
                "pos_ranks": session["pos_ranks_by_idx"][idx],
                "roster": roster_details,
                "weekly_schedule": weekly_sched,
            }
        )

    scoring = lg["scoring_settings"]
    scoring_summary = {
        "ppr": scoring.get("rec", 0.0),
        "pass_td": scoring.get("pass_td", 4.0),
        "pass_int": scoring.get("pass_int", -1.0),
        "pass_yd_per_pt": round(1.0 / scoring["pass_yd"]) if scoring.get("pass_yd") else 25,
        "rush_td": scoring.get("rush_td", 6.0),
        "rec_td": scoring.get("rec_td", 6.0),
        "te_premium": scoring.get("bonus_rec_te", 0.0),
        "fum_lost": scoring.get("fum_lost", -2.0),
    }

    return {
        "ready": True,
        "session_key": session["session_key"],
        "user": {
            "user_id": session.get("user_id"),
            "username": session.get("username"),
        },
        "league": {
            "platform": "sleeper",
            "league_id": lg["league_id"],
            "league_name": lg["league_name"],
            "season": lg["season"],
            "status": lg["status"],
            "current_week": lg["current_week"],
            "sim_weeks": sim_ctx.sim_weeks,
            "reg_weeks": sim_ctx.reg_weeks,
            "playoff_weeks": sim_ctx.playoff_weeks,
            "playoff_week_start": lg["playoff_week_start"],
            "playoff_teams": lg["playoff_teams"],
            "total_rosters": lg["total_rosters"],
            "use_median_scoring": lg["use_median_scoring"],
            "neutralize_byes": bool(lg.get("neutralize_byes", True)),
            "trade_deadline": lg["trade_deadline"],
            "reserve_slots": lg["reserve_slots"],
            "roster_positions": lg["roster_positions"],
            "starter_slots": sim_ctx.starter_slots,
            "my_roster_id": lg["my_roster_id"],
            "scoring_summary": scoring_summary,
            "num_sims": sim_ctx.num_sims,
        },
        "available_leagues": session["available_leagues"],
        "fc_params": session["fc_params"],
        "fc_fetched_at": session["fc_fetched_at"],
        "waiver_calibration": {
            "waiver_adj_1": waiver_cal["waiver_adj_1"],
            "waiver_adj_2": waiver_cal["waiver_adj_2"],
            "marginal_2": waiver_cal["marginal_2"],
            "median_bench_drop_1": waiver_cal["median_bench_drop_1"],
            "median_bench_drop_2": waiver_cal["median_bench_drop_2"],
            "top_waiver_avg_fc": waiver_cal["top_waiver_avg_fc"],
            "top_waivers": waiver_cal["waiver_pool"][:25],
        },
        "teams": teams_out,
        "arbitrage": session["arbitrage_rows"],
        "sync_ms": session["sync_ms"],
    }


def _resolve_request_session(body: Dict[str, Any]) -> Dict[str, Any]:
    skey = str(body.get("session_key") or "").strip()
    lid = str(body.get("league_id") or "").strip()
    username = str(body.get("username") or "").strip()
    user_id = str(body.get("user_id") or "").strip() or None

    with _SESSION_LOCK:
        if skey and skey in _SESSION_CACHE:
            _SESSION_CACHE.move_to_end(skey)
            return _SESSION_CACHE[skey]
        if lid:
            for key in reversed(list(_SESSION_CACHE.keys())):
                sess = _SESSION_CACHE[key]
                if str(sess["league_info"]["league_id"]) == lid:
                    _SESSION_CACHE.move_to_end(key)
                    return sess
        if _DEFAULT_SESSION_KEY and _DEFAULT_SESSION_KEY in _SESSION_CACHE and not lid:
            return _SESSION_CACHE[_DEFAULT_SESSION_KEY]

    if lid or username:
        return build_league_session(
            league_id=lid or None,
            username=username,
            user_id=user_id,
        )

    raise ValueError("No active league session. Enter your Sleeper username and select a league.")


class BaseAPIHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store")

    def write_json(self, data: Any, status: int = 200) -> None:
        self.set_status(status)
        self.finish(json.dumps(data))

    def parse_body(self) -> Dict[str, Any]:
        if not self.request.body:
            return {}
        try:
            parsed = json.loads(self.request.body.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


class IndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        index_path = FRONTEND_DIR / "index.html"
        with open(index_path, "r", encoding="utf-8") as f:
            self.finish(f.read())


class UserLeaguesHandler(BaseAPIHandler):
    async def post(self) -> None:
        body = self.parse_body()
        username = str(body.get("username") or "").strip()
        force_refresh = bool(body.get("force_refresh", False))
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _EXECUTOR,
                lookup_user_leagues,
                username,
                force_refresh,
            )
            self.write_json(result)
        except Exception as exc:
            logger.warning("League lookup failed for '%s': %s", username, exc)
            self.write_json({"error": str(exc)}, status=400)


class StateHandler(BaseAPIHandler):
    async def get(self) -> None:
        username = self.get_argument("username", "").strip()
        league_id = self.get_argument("league_id", "").strip()
        try:
            loop = asyncio.get_event_loop()
            if not username and not league_id:
                with _SESSION_LOCK:
                    default_sess = (
                        _SESSION_CACHE.get(_DEFAULT_SESSION_KEY)
                        if _DEFAULT_SESSION_KEY
                        else None
                    )
                if default_sess is None:
                    self.write_json(
                        {
                            "ready": False,
                            "message": "Enter your Sleeper username to load your leagues.",
                        }
                    )
                    return
                payload = await loop.run_in_executor(
                    _EXECUTOR, _serialize_session_state, default_sess
                )
                self.write_json(payload)
                return

            sess = _find_cached_session(
                league_id=league_id,
                user_key=username,
                start_week=None,
                use_median=None,
                neutralize_byes=True,
            ) if league_id else None

            if sess is None:
                sess = await loop.run_in_executor(
                    _EXECUTOR,
                    build_league_session,
                    league_id or None,
                    username,
                )
            payload = await loop.run_in_executor(_EXECUTOR, _serialize_session_state, sess)
            self.write_json(payload)
        except Exception as exc:
            logger.exception("Error in GET /api/state")
            self.write_json({"error": str(exc)}, status=400)


class SyncHandler(BaseAPIHandler):
    async def post(self) -> None:
        body = self.parse_body()
        username = str(body.get("username") or "").strip()
        user_id = str(body.get("user_id") or "").strip() or None
        league_id = str(body.get("league_id") or "").strip() or None
        team_name = str(body.get("team_name") or body.get("target_team_name") or "").strip()
        force_refresh = bool(body.get("force_refresh", False))
        start_week = body.get("start_week")
        start_week_int = int(start_week) if start_week not in (None, "", 0) else None
        use_median = body.get("use_median")
        use_median_opt = None if use_median is None else bool(use_median)
        neutralize_byes = bool(body.get("neutralize_byes", True))

        try:
            loop = asyncio.get_event_loop()
            user_key = (user_id or username or "").strip().lower()
            cached_sess = None
            if not force_refresh and league_id and use_median_opt is not None:
                cached_sess = _find_cached_session(
                    league_id=league_id,
                    user_key=user_key,
                    start_week=start_week_int,
                    use_median=use_median_opt,
                    neutralize_byes=neutralize_byes,
                )

            if cached_sess is not None:
                session = cached_sess
            else:
                session = await loop.run_in_executor(
                    _EXECUTOR,
                    build_league_session,
                    league_id,
                    username,
                    user_id,
                    team_name,
                    force_refresh,
                    start_week_int,
                    use_median_opt,
                    neutralize_byes,
                )

            payload = await loop.run_in_executor(_EXECUTOR, _serialize_session_state, session)
            self.write_json(payload)
        except Exception as exc:
            logger.exception("Error in POST /api/sync")
            self.write_json({"error": str(exc)}, status=400)


class FindTradesHandler(BaseAPIHandler):
    async def post(self) -> None:
        body = self.parse_body()
        try:
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(_EXECUTOR, _resolve_request_session, body)
            my_rid = int(body.get("my_roster_id") or session["league_info"]["my_roster_id"])
            partner_raw = body.get("partner_roster_id")
            partner_rid = int(partner_raw) if partner_raw not in (None, "", 0, "all") else None
            package_sizes = body.get("package_sizes")
            min_pct = float(body.get("min_perceived_pct", -12.0))
            max_pct = float(body.get("max_perceived_pct", 20.0))
            mode = str(body.get("mode") or "mutual_champ")
            must_include = body.get("must_include") or []
            exclude = body.get("exclude") or []
            require_both = bool(body.get("require_both_champ_up", True))
            allow_same_pos = bool(body.get("allow_same_pos_1for1", False))

            result = await loop.run_in_executor(
                _EXECUTOR,
                find_viable_trades,
                session,
                my_rid,
                partner_rid,
                package_sizes,
                min_pct,
                max_pct,
                mode,
                must_include,
                exclude,
                require_both,
                allow_same_pos,
            )
            self.write_json(result)
        except Exception as exc:
            logger.exception("Error in POST /api/trades/find")
            self.write_json({"error": str(exc)}, status=400)


class EvaluateTradeHandler(BaseAPIHandler):
    async def post(self) -> None:
        body = self.parse_body()
        try:
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(_EXECUTOR, _resolve_request_session, body)
            roster_a = int(body["roster_a_id"])
            roster_b = int(body["roster_b_id"])
            send_a = [str(p) for p in (body.get("send_a") or [])]
            send_b = [str(p) for p in (body.get("send_b") or [])]

            result = await loop.run_in_executor(
                _EXECUTOR,
                evaluate_custom_trade,
                session,
                roster_a,
                roster_b,
                send_a,
                send_b,
            )
            self.write_json(result)
        except Exception as exc:
            logger.exception("Error in POST /api/trades/evaluate")
            self.write_json({"error": str(exc)}, status=400)


class NoCacheStaticFileHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path: str) -> None:
        self.set_header("Cache-Control", "no-store")


def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/api/user/leagues", UserLeaguesHandler),
            (r"/api/state", StateHandler),
            (r"/api/sync", SyncHandler),
            (r"/api/trades/find", FindTradesHandler),
            (r"/api/trades/evaluate", EvaluateTradeHandler),
            (r"/static/(.*)", NoCacheStaticFileHandler, {"path": str(FRONTEND_DIR)}),
        ]
    )


def main() -> None:
    global _DEFAULT_SESSION_KEY
    parser = argparse.ArgumentParser(description="Public Sleeper Redraft Trade Simulator Server")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8060")))
    parser.add_argument("--preload-user", type=str, default="", help="Optional Sleeper username to preload on startup")
    args = parser.parse_args()

    if args.preload_user.strip():
        sess = build_league_session(username=args.preload_user.strip())
        _DEFAULT_SESSION_KEY = sess["session_key"]

    app = make_app()
    app.listen(args.port, address=args.host)
    logger.info("Public Sleeper Trade Simulator listening on http://%s:%s", args.host, args.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
