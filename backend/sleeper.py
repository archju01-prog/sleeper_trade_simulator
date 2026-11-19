"""Sleeper API client, league resolver, and custom-scoring projection engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from .cache import get_cached, set_cached

logger = logging.getLogger(__name__)

SLEEPER_APP_BASE = "https://api.sleeper.app/v1"
SLEEPER_PROJECTIONS_BASE = "https://api.sleeper.com/projections/nfl"

VALID_POSITIONS: Set[str] = {"QB", "RB", "WR", "TE", "K", "DEF"}
NON_SCORING_STAT_PREFIXES = ("adp_", "rank_", "pos_rank_")
NON_SCORING_STAT_KEYS = {"gp", "gs", "gms_active", "pts_ppr", "pts_half_ppr", "pts_std", "pts_idp"}

TTL_PLAYERS = 24 * 3600
TTL_PROJECTIONS = 6 * 3600
TTL_LEAGUE = 1800
TTL_STATE = 1800

MAX_SINGLE_WEEK_PROJ: Dict[str, float] = {
    "QB": 32.0,
    "RB": 30.0,
    "WR": 30.0,
    "TE": 26.0,
    "K": 12.5,
    "DEF": 13.5,
}


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, params=params, headers={"User-Agent": "SleeperTradeSim/1.0"})
        resp.raise_for_status()
        return resp.json()


def _cached_get_json(
    cache_key: str,
    url: str,
    max_age_seconds: float,
    params: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> Any:
    if not force_refresh:
        cached = get_cached(cache_key, max_age_seconds)
        if cached is not None:
            return cached
    try:
        data = _http_get_json(url, params=params, timeout=timeout)
        if data is not None:
            set_cached(cache_key, data)
        return data
    except httpx.HTTPError as exc:
        stale = get_cached(cache_key, max_age_seconds=-1)
        if stale is not None:
            logger.warning("HTTP fetch failed (%s); falling back to cached %s", exc, cache_key)
            return stale
        raise


def get_nfl_state(force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch current NFL season/week metadata from Sleeper."""
    raw = _cached_get_json(
        "sleeper:state:nfl",
        f"{SLEEPER_APP_BASE}/state/nfl",
        TTL_STATE,
        force_refresh=force_refresh,
    )
    return {
        "season": str(raw.get("season") or raw.get("league_season") or "2026"),
        "week": int(raw.get("week") or 1),
        "display_week": int(raw.get("display_week") or raw.get("week") or 1),
        "season_type": str(raw.get("season_type") or "regular"),
        "leg": int(raw.get("leg") or raw.get("week") or 1),
        "raw": raw,
    }


def _match_user_team_in_league(
    league: Dict[str, Any],
    users: List[Dict[str, Any]],
    rosters: List[Dict[str, Any]],
    user_id: Optional[str],
    target_team_name: str,
) -> Tuple[bool, str]:
    """Check whether a league matches target_team_name or user_id; return (matched, team_name)."""
    target_lower = target_team_name.strip().lower()
    user_map = {str(u.get("user_id")): u for u in users if u.get("user_id")}

    # First check explicit team name match across users/rosters
    for r in rosters:
        owner_id = str(r.get("owner_id") or "")
        u = user_map.get(owner_id, {})
        meta_r = r.get("metadata") or {}
        meta_u = u.get("metadata") or {}
        candidates = [
            str(meta_r.get("team_name") or ""),
            str(meta_u.get("team_name") or ""),
            str(u.get("display_name") or ""),
        ]
        for c in candidates:
            if c and target_lower and target_lower in c.lower():
                return True, c

    # Fallback check user_id ownership
    if user_id:
        for r in rosters:
            if str(r.get("owner_id") or "") == str(user_id):
                u = user_map.get(str(user_id), {})
                meta_u = u.get("metadata") or {}
                meta_r = r.get("metadata") or {}
                tname = (
                    meta_r.get("team_name")
                    or meta_u.get("team_name")
                    or u.get("display_name")
                    or league.get("name")
                    or ""
                )
                return False, str(tname)

    return False, ""


def _summarize_league_obj(lg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract lightweight league metadata from a Sleeper league object without extra HTTP calls."""
    league_id = str(lg.get("league_id") or "")
    name = str(lg.get("name") or f"League {league_id}")
    season = str(lg.get("season") or "")
    total_rosters = int(lg.get("total_rosters") or 12)
    status = str(lg.get("status") or "")
    settings = lg.get("settings") or {}
    scoring = lg.get("scoring_settings") or {}
    roster_positions = [str(p) for p in (lg.get("roster_positions") or [])]

    rec = float(scoring.get("rec") or 0.0)
    ppr_label = "Full PPR" if rec >= 0.9 else ("Half-PPR" if rec >= 0.4 else "Std")
    qb_count = roster_positions.count("QB") + roster_positions.count("SUPER_FLEX")
    qb_label = "Superflex" if ("SUPER_FLEX" in roster_positions or qb_count >= 2) else "1QB"
    is_redraft = int(settings.get("type") or 0) == 0
    use_median_scoring = int(settings.get("league_average_match") or 0) == 1

    return {
        "league_id": league_id,
        "name": name,
        "season": season,
        "total_rosters": total_rosters,
        "status": status,
        "ppr_label": ppr_label,
        "qb_label": qb_label,
        "is_redraft": is_redraft,
        "use_median_scoring": use_median_scoring,
    }


def lookup_user_leagues(identifier: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Resolve a Sleeper username or League ID into lightweight league summaries sorted by relevance."""
    ident = str(identifier or "").strip()
    if not ident:
        raise ValueError("Please enter a Sleeper username or League ID.")

    if ident.isdigit():
        try:
            league = _cached_get_json(
                f"sleeper:league:{ident}",
                f"{SLEEPER_APP_BASE}/league/{ident}",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
        except httpx.HTTPError as exc:
            raise ValueError(f"Sleeper league '{ident}' not found.") from exc
        if not isinstance(league, dict) or not league.get("league_id"):
            raise ValueError(f"Sleeper league '{ident}' not found.")
        return {
            "user_id": None,
            "username": ident,
            "display_name": ident,
            "leagues": [_summarize_league_obj(league)],
        }

    user_info = _cached_get_json(
        f"sleeper:user:{ident.lower()}",
        f"{SLEEPER_APP_BASE}/user/{ident}",
        TTL_LEAGUE,
        force_refresh=force_refresh,
    )
    if not user_info or not user_info.get("user_id"):
        raise ValueError(f"Sleeper user '{ident}' not found.")

    user_id = str(user_info["user_id"])
    nfl_state = get_nfl_state(force_refresh=force_refresh)
    seasons_to_check: List[str] = []
    for s in [nfl_state["season"], "2026", "2025", "2024"]:
        if s and s not in seasons_to_check:
            seasons_to_check.append(s)

    all_leagues: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for season in seasons_to_check:
        try:
            leagues = _cached_get_json(
                f"sleeper:user_leagues:{user_id}:{season}",
                f"{SLEEPER_APP_BASE}/user/{user_id}/leagues/nfl/{season}",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
        except httpx.HTTPError:
            leagues = []
        if not isinstance(leagues, list):
            continue
        for lg in leagues:
            lid = str(lg.get("league_id") or "")
            if not lid or lid in seen_ids:
                continue
            seen_ids.add(lid)
            all_leagues.append(lg)

    if not all_leagues:
        raise ValueError(f"No NFL leagues found for Sleeper user '{ident}'.")

    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    cur_season = str(nfl_state["season"])
    for lg in all_leagues:
        summary = _summarize_league_obj(lg)
        score = 0
        if summary["season"] == cur_season:
            score += 200
        st = summary["status"]
        if st == "in_season":
            score += 150
        elif st in ("post_season", "complete"):
            score += 80
        elif st == "drafting":
            score += 50
        if summary["is_redraft"]:
            score += 40
        scored.append((score, summary["season"], summary))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    sorted_summaries = [item[2] for item in scored]

    return {
        "user_id": user_id,
        "username": str(user_info.get("username") or ident),
        "display_name": str(user_info.get("display_name") or user_info.get("username") or ident),
        "leagues": sorted_summaries,
    }


def resolve_league_id(
    identifier: str = "archju01",
    target_team_name: str = "Pollard McCaff Muscle",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Resolve a Sleeper league_id or username into a concrete league_id plus candidate leagues."""
    ident = str(identifier or "").strip()
    if not ident:
        raise ValueError("Sleeper username or league ID is required.")

    if ident.isdigit():
        league = _cached_get_json(
            f"sleeper:league:{ident}",
            f"{SLEEPER_APP_BASE}/league/{ident}",
            TTL_LEAGUE,
            force_refresh=force_refresh,
        )
        return {
            "league_id": ident,
            "user_id": None,
            "username": None,
            "leagues": [
                {
                    "league_id": ident,
                    "name": league.get("name", f"League {ident}"),
                    "season": str(league.get("season", "")),
                    "total_rosters": int(league.get("total_rosters") or 12),
                    "status": league.get("status", ""),
                }
            ],
        }

    user_info = _cached_get_json(
        f"sleeper:user:{ident.lower()}",
        f"{SLEEPER_APP_BASE}/user/{ident}",
        TTL_LEAGUE,
        force_refresh=force_refresh,
    )
    if not user_info or not user_info.get("user_id"):
        raise ValueError(f"Sleeper user '{ident}' not found.")

    user_id = str(user_info["user_id"])
    nfl_state = get_nfl_state(force_refresh=force_refresh)
    seasons_to_check: List[str] = []
    for s in [nfl_state["season"], "2026", "2025", "2024"]:
        if s and s not in seasons_to_check:
            seasons_to_check.append(s)

    all_leagues: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for season in seasons_to_check:
        try:
            leagues = _cached_get_json(
                f"sleeper:user_leagues:{user_id}:{season}",
                f"{SLEEPER_APP_BASE}/user/{user_id}/leagues/nfl/{season}",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
        except httpx.HTTPError:
            leagues = []
        if not isinstance(leagues, list):
            continue
        for lg in leagues:
            lid = str(lg.get("league_id") or "")
            if not lid or lid in seen_ids:
                continue
            seen_ids.add(lid)
            all_leagues.append(lg)

    if not all_leagues:
        raise ValueError(f"No NFL leagues found for Sleeper user '{ident}'.")

    scored_candidates: List[Tuple[int, Dict[str, Any]]] = []
    available_summaries: List[Dict[str, Any]] = []

    for lg in all_leagues:
        lid = str(lg["league_id"])
        try:
            users = _cached_get_json(
                f"sleeper:league:{lid}:users",
                f"{SLEEPER_APP_BASE}/league/{lid}/users",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
            rosters = _cached_get_json(
                f"sleeper:league:{lid}:rosters",
                f"{SLEEPER_APP_BASE}/league/{lid}/rosters",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
        except httpx.HTTPError:
            users, rosters = [], []

        matched_team, team_name = _match_user_team_in_league(
            lg, users or [], rosters or [], user_id, target_team_name
        )
        has_players = any(bool(r.get("players")) for r in (rosters or []))
        settings = lg.get("settings") or {}
        is_redraft = int(settings.get("type") or 0) == 0
        status = str(lg.get("status") or "")

        score = 0
        if matched_team:
            score += 1000
        if str(lg.get("season")) == nfl_state["season"]:
            score += 200
        if status == "in_season":
            score += 150
        elif status == "drafting":
            score += 50
        if has_players:
            score += 100
        if is_redraft:
            score += 40
        if int(lg.get("total_rosters") or 0) == 12:
            score += 20

        summary = {
            "league_id": lid,
            "name": lg.get("name", f"League {lid}"),
            "season": str(lg.get("season", "")),
            "total_rosters": int(lg.get("total_rosters") or 12),
            "status": status,
            "team_name": team_name,
            "has_rosters": has_players,
            "is_redraft": is_redraft,
        }
        available_summaries.append(summary)
        scored_candidates.append((score, summary))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best = scored_candidates[0][1]

    return {
        "league_id": best["league_id"],
        "user_id": user_id,
        "username": user_info.get("username") or ident,
        "leagues": available_summaries,
    }


def fetch_players_master(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch Sleeper NFL player catalog and return a compact map for fantasy positions."""
    compact_key = "sleeper:players_master:compact:v1"
    if not force_refresh:
        cached_compact = get_cached(compact_key, TTL_PLAYERS)
        if cached_compact is not None:
            return cached_compact

    raw = _http_get_json(f"{SLEEPER_APP_BASE}/players/nfl", timeout=60.0)
    compact: Dict[str, Dict[str, Any]] = {}
    for pid, info in raw.items():
        if not isinstance(info, dict):
            continue
        pos = info.get("position")
        if pos not in VALID_POSITIONS:
            continue
        full_name = info.get("full_name")
        if not full_name:
            first = str(info.get("first_name") or "").strip()
            last = str(info.get("last_name") or "").strip()
            full_name = f"{first} {last}".strip() or str(pid)
        team = info.get("team") or (pid if pos == "DEF" else "FA")
        compact[str(pid)] = {
            "player_id": str(pid),
            "full_name": full_name,
            "position": pos,
            "team": team,
            "injury_status": info.get("injury_status") or "",
            "status": info.get("status") or "",
            "years_exp": info.get("years_exp"),
        }

    set_cached(compact_key, compact)
    return compact


def _generate_round_robin_schedule(roster_ids: List[int], weeks: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """Generate a deterministic balanced round-robin schedule for unscheduled weeks."""
    teams = sorted(roster_ids)
    if len(teams) % 2 == 1:
        teams.append(-1)
    n = len(teams)
    schedule: Dict[int, List[Tuple[int, int]]] = {}
    rot = list(teams)
    rounds: List[List[Tuple[int, int]]] = []
    for _ in range(n - 1):
        pairs: List[Tuple[int, int]] = []
        for i in range(n // 2):
            t1, t2 = rot[i], rot[n - 1 - i]
            if t1 != -1 and t2 != -1:
                pairs.append((t1, t2))
        rounds.append(pairs)
        rot = [rot[0]] + [rot[-1]] + rot[1:-1]

    for idx, week in enumerate(weeks):
        schedule[week] = rounds[idx % len(rounds)]
    return schedule


def _resolve_team_display_name(
    roster: Dict[str, Any],
    user_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    """Return (team_name, manager_name) for a Sleeper roster."""
    owner_id = str(roster.get("owner_id") or "")
    u = user_map.get(owner_id, {})
    meta_r = roster.get("metadata") or {}
    meta_u = u.get("metadata") or {}
    manager_name = str(u.get("display_name") or u.get("username") or f"Manager {roster.get('roster_id')}")
    team_name = str(
        meta_r.get("team_name")
        or meta_u.get("team_name")
        or manager_name
        or f"Team {roster.get('roster_id')}"
    )
    return team_name, manager_name


def fetch_league_data(
    league_id: str,
    force_refresh: bool = False,
    target_team_name: str = "",
    start_week_override: Optional[int] = None,
    use_median_override: Optional[bool] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch Sleeper league settings, rosters, standings, and remaining schedule."""
    nfl_state = get_nfl_state(force_refresh=force_refresh)

    league = _cached_get_json(
        f"sleeper:league:{league_id}",
        f"{SLEEPER_APP_BASE}/league/{league_id}",
        TTL_LEAGUE,
        force_refresh=force_refresh,
    )
    users = _cached_get_json(
        f"sleeper:league:{league_id}:users",
        f"{SLEEPER_APP_BASE}/league/{league_id}/users",
        TTL_LEAGUE,
        force_refresh=force_refresh,
    )
    rosters_raw = _cached_get_json(
        f"sleeper:league:{league_id}:rosters",
        f"{SLEEPER_APP_BASE}/league/{league_id}/rosters",
        TTL_LEAGUE,
        force_refresh=force_refresh,
    )

    settings = league.get("settings") or {}
    scoring_settings = {
        str(k): float(v)
        for k, v in (league.get("scoring_settings") or {}).items()
        if isinstance(v, (int, float))
    }
    roster_positions = [str(p) for p in (league.get("roster_positions") or [])]

    total_rosters = int(league.get("total_rosters") or len(rosters_raw) or 12)
    playoff_week_start = int(settings.get("playoff_week_start") or 15)
    if playoff_week_start <= 1:
        playoff_week_start = 15
    playoff_teams = int(settings.get("playoff_teams") or 6)
    league_average_match = int(settings.get("league_average_match") or 0) == 1
    trade_deadline = int(settings.get("trade_deadline") or 11)
    reserve_slots = int(settings.get("reserve_slots") or 2)
    playoff_round_type = int(settings.get("playoff_round_type") or 0)
    playoff_seed_type = int(settings.get("playoff_seed_type") or 0)

    user_map = {str(u.get("user_id")): u for u in (users or []) if u.get("user_id")}

    # Fetch regular-season matchups up to playoff_week_start - 1
    reg_weeks = list(range(1, playoff_week_start))
    matchups_raw_by_week: Dict[int, List[Dict[str, Any]]] = {}
    schedule_by_week: Dict[int, List[Tuple[int, int]]] = {}
    completed_weeks: List[int] = []

    for w in reg_weeks:
        try:
            wk_matchups = _cached_get_json(
                f"sleeper:league:{league_id}:matchups:{w}",
                f"{SLEEPER_APP_BASE}/league/{league_id}/matchups/{w}",
                TTL_LEAGUE,
                force_refresh=force_refresh,
            )
        except httpx.HTTPError:
            wk_matchups = []
        if not isinstance(wk_matchups, list):
            wk_matchups = []
        matchups_raw_by_week[w] = wk_matchups

        has_points = any(float(m.get("points") or 0.0) > 0.0 for m in wk_matchups)
        if has_points:
            completed_weeks.append(w)

        by_matchup_id: Dict[Any, List[int]] = {}
        for m in wk_matchups:
            mid = m.get("matchup_id")
            rid = m.get("roster_id")
            if mid is not None and rid is not None:
                by_matchup_id.setdefault(mid, []).append(int(rid))

        pairs: List[Tuple[int, int]] = []
        for mid, rids in sorted(by_matchup_id.items()):
            if len(rids) == 2:
                pairs.append((rids[0], rids[1]))
        if len(pairs) == total_rosters // 2:
            schedule_by_week[w] = pairs

    # Determine current week to begin remaining simulation
    league_season = str(league.get("season") or nfl_state["season"])
    state_week = int(nfl_state.get("display_week") or nfl_state.get("week") or 1)
    if start_week_override is not None:
        current_week = max(1, min(17, int(start_week_override)))
    else:
        # Infer from completed weeks if state_week is ahead/behind or league is from an earlier season
        if league_season == str(nfl_state["season"]) and nfl_state["season_type"] == "regular":
            last_done = max(completed_weeks) if completed_weeks else 0
            # If state_week already has full points recorded, start at next week; else state_week
            if state_week in completed_weeks and state_week < 17:
                # Check if matchup points look final (most teams scored > 40)
                wk_pts = [float(m.get("points") or 0.0) for m in matchups_raw_by_week.get(state_week, [])]
                if len(wk_pts) >= total_rosters and sum(1 for p in wk_pts if p > 40.0) >= total_rosters // 2:
                    current_week = state_week + 1
                else:
                    current_week = state_week
            else:
                current_week = max(state_week, last_done + 1)
        elif completed_weeks:
            current_week = min(17, max(completed_weeks) + 1)
        else:
            current_week = 1

    current_week = max(1, min(17, current_week))

    roster_ids = [int(r["roster_id"]) for r in (rosters_raw or [])]
    unscheduled_weeks = [w for w in reg_weeks if w not in schedule_by_week]
    if unscheduled_weeks and roster_ids:
        fallback_sched = _generate_round_robin_schedule(roster_ids, unscheduled_weeks)
        schedule_by_week.update(fallback_sched)

    # Parse teams & standings
    teams: List[Dict[str, Any]] = []
    owner_matched_rid: Optional[int] = None
    name_matched_rid: Optional[int] = None
    uname_matched_rid: Optional[int] = None
    target_lower = target_team_name.strip().lower()
    uname_lower = (username or "").strip().lower()

    for r in sorted((rosters_raw or []), key=lambda x: int(x["roster_id"])):
        rid = int(r["roster_id"])
        rset = r.get("settings") or {}
        team_name, manager_name = _resolve_team_display_name(r, user_map)
        wins = int(rset.get("wins") or 0)
        losses = int(rset.get("losses") or 0)
        ties = int(rset.get("ties") or 0)
        fpts = float(rset.get("fpts") or 0) + float(rset.get("fpts_decimal") or 0) / 100.0
        fpts_against = float(rset.get("fpts_against") or 0) + float(rset.get("fpts_against_decimal") or 0) / 100.0

        players = [str(p) for p in (r.get("players") or []) if p]
        reserve = [str(p) for p in (r.get("reserve") or []) if p]
        starters = [str(p) for p in (r.get("starters") or []) if p and p != "0"]
        taxi = [str(p) for p in (r.get("taxi") or []) if p]

        owner_str = str(r.get("owner_id") or "")
        co_owners = [str(x) for x in (r.get("co_owners") or [])]
        owner_match = bool(user_id and (owner_str == str(user_id) or str(user_id) in co_owners))
        name_match = bool(target_lower and (target_lower in team_name.lower() or target_lower in manager_name.lower()))
        owner_username = str((user_map.get(owner_str) or {}).get("username") or "").lower()
        username_match = bool(uname_lower and (uname_lower == manager_name.lower() or uname_lower == owner_username))

        if owner_match and owner_matched_rid is None:
            owner_matched_rid = rid
        if name_match and name_matched_rid is None:
            name_matched_rid = rid
        if username_match and uname_matched_rid is None:
            uname_matched_rid = rid

        teams.append(
            {
                "roster_id": rid,
                "owner_id": owner_str,
                "team_name": team_name,
                "manager_name": manager_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "points_for": round(fpts, 2),
                "points_against": round(fpts_against, 2),
                "players": players,
                "reserve": reserve,
                "starters": starters,
                "taxi": taxi,
            }
        )

    my_roster_id: Optional[int] = next(
        (x for x in (owner_matched_rid, name_matched_rid, uname_matched_rid) if x is not None),
        None,
    )
    if my_roster_id is None and teams:
        my_roster_id = teams[0]["roster_id"]

    # Determine median scoring flag
    use_median = league_average_match if use_median_override is None else bool(use_median_override)
    if use_median_override is None and not use_median:
        # Check if total recorded wins across teams exceeds total H2H games played (~2x), indicating median scoring
        total_recorded_games = sum(t["wins"] + t["losses"] + t["ties"] for t in teams)
        completed_reg = len([w for w in completed_weeks if w < current_week])
        if completed_reg > 0 and total_recorded_games >= int(1.6 * total_rosters * completed_reg):
            use_median = True

    return {
        "league_id": str(league_id),
        "league_name": str(league.get("name") or f"League {league_id}"),
        "season": league_season,
        "status": str(league.get("status") or ""),
        "nfl_state": nfl_state,
        "current_week": current_week,
        "completed_weeks": completed_weeks,
        "total_rosters": total_rosters,
        "playoff_week_start": playoff_week_start,
        "playoff_teams": playoff_teams,
        "playoff_round_type": playoff_round_type,
        "playoff_seed_type": playoff_seed_type,
        "league_average_match": league_average_match,
        "use_median_scoring": use_median,
        "trade_deadline": trade_deadline,
        "reserve_slots": reserve_slots,
        "roster_positions": roster_positions,
        "scoring_settings": scoring_settings,
        "schedule_by_week": schedule_by_week,
        "teams": teams,
        "my_roster_id": my_roster_id,
    }


def _compute_custom_points(stats: Dict[str, Any], scoring_settings: Dict[str, float]) -> float:
    """Compute league-specific projected fantasy points from raw Sleeper projected stats."""
    pts = 0.0
    matched_any = False
    for stat_key, stat_val in stats.items():
        if not isinstance(stat_val, (int, float)):
            continue
        if stat_key in NON_SCORING_STAT_KEYS or stat_key.startswith(NON_SCORING_STAT_PREFIXES):
            continue
        weight = scoring_settings.get(stat_key)
        if weight is not None and weight != 0.0:
            pts += float(stat_val) * float(weight)
            matched_any = True

    if (not matched_any or pts == 0.0) and (
        "pts_ppr" in stats or "pts_half_ppr" in stats or "pts_std" in stats
    ):
        rec_weight = float(scoring_settings.get("rec", 1.0))
        pts_std = float(stats.get("pts_std") or 0.0)
        pts_half = float(stats.get("pts_half_ppr") or pts_std)
        pts_ppr = float(stats.get("pts_ppr") or pts_half)
        if rec_weight >= 1.0:
            pts = pts_ppr
        elif rec_weight >= 0.5:
            pts = pts_half + (rec_weight - 0.5) * 2.0 * (pts_ppr - pts_half)
        else:
            pts = pts_std + rec_weight * 2.0 * (pts_half - pts_std)

    return max(0.0, round(pts, 3))


def fetch_custom_projections(
    season: str,
    start_week: int,
    end_week: int,
    scoring_settings: Dict[str, float],
    players_master: Dict[str, Dict[str, Any]],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch Sleeper weekly raw projections and apply custom league scoring_settings."""
    weekly_projections: Dict[str, Dict[int, float]] = {}
    weeks = list(range(start_week, end_week + 1))

    params = [
        ("season_type", "regular"),
        ("position[]", "QB"),
        ("position[]", "RB"),
        ("position[]", "WR"),
        ("position[]", "TE"),
        ("position[]", "K"),
        ("position[]", "DEF"),
        ("order_by", "pts_ppr"),
    ]

    for w in weeks:
        cache_key = f"sleeper:projections:{season}:{w}"
        url = f"{SLEEPER_PROJECTIONS_BASE}/{season}/{w}"
        raw_list: Optional[List[Dict[str, Any]]] = None
        if not force_refresh:
            raw_list = get_cached(cache_key, TTL_PROJECTIONS)
        if raw_list is None:
            try:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    resp = client.get(url, params=params, headers={"User-Agent": "SleeperTradeSim/1.0"})
                    resp.raise_for_status()
                    raw_list = resp.json()
                if isinstance(raw_list, list) and len(raw_list) > 0:
                    set_cached(cache_key, raw_list)
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch projections for %s week %s: %s", season, w, exc)
                raw_list = []

        if not isinstance(raw_list, list):
            continue

        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            pid = str(entry.get("player_id") or "")
            if not pid:
                continue
            if pid not in players_master:
                pinfo = entry.get("player") or {}
                pos = pinfo.get("position")
                if pos in VALID_POSITIONS:
                    first = str(pinfo.get("first_name") or "").strip()
                    last = str(pinfo.get("last_name") or "").strip()
                    players_master[pid] = {
                        "player_id": pid,
                        "full_name": f"{first} {last}".strip() or pid,
                        "position": pos,
                        "team": pinfo.get("team") or "FA",
                        "injury_status": pinfo.get("injury_status") or "",
                        "status": pinfo.get("status") or "",
                        "years_exp": pinfo.get("years_exp"),
                    }
                else:
                    continue

            stats = entry.get("stats") or {}
            pts = _compute_custom_points(stats, scoring_settings)
            pos = str((players_master.get(pid) or {}).get("position") or "")
            if pos == "K" and pts > 15.0:
                pts = round(pts / max(2, round(pts / 9.0)), 3)
            else:
                cap = MAX_SINGLE_WEEK_PROJ.get(pos)
                if cap is not None and pts > cap:
                    pts = cap
            if pts > 0.0:
                weekly_projections.setdefault(pid, {})[w] = pts

    # Identify which weeks each NFL team actually plays so injury absences aren't mistaken for byes
    team_active_weeks: Dict[str, Set[int]] = {}
    for pid, wk_map in weekly_projections.items():
        nfl_team = str((players_master.get(pid) or {}).get("team") or "")
        if not nfl_team or nfl_team == "FA":
            continue
        t_wks = team_active_weeks.setdefault(nfl_team, set())
        for w, pts in wk_map.items():
            if pts > 1.0:
                t_wks.add(w)

    # Compute ROS totals
    ros_totals: Dict[str, Dict[str, Any]] = {}
    for pid, wk_map in weekly_projections.items():
        reg_pts = sum(v for w, v in wk_map.items() if w < 15)
        playoff_pts = sum(v for w, v in wk_map.items() if 15 <= w <= 17)
        total_pts = reg_pts + playoff_pts
        active_games = sum(1 for w in weeks if wk_map.get(w, 0.0) > 0.5)
        reg_games = sum(1 for w in weeks if w < 15 and wk_map.get(w, 0.0) > 0.5)
        po_games = sum(1 for w in weeks if 15 <= w <= 17 and wk_map.get(w, 0.0) > 0.5)
        nfl_team = str((players_master.get(pid) or {}).get("team") or "")
        team_wks = team_active_weeks.get(nfl_team, set())
        bye_weeks: List[int] = []
        if active_games >= 3 and playoff_pts > 0.0 and nfl_team and nfl_team != "FA":
            bye_weeks = [
                w for w in weeks
                if w < 15 and wk_map.get(w, 0.0) <= 0.05 and w not in team_wks
            ]
        return_week = min((w for w in weeks if wk_map.get(w, 0.0) > 0.5), default=None)
        ros_totals[pid] = {
            "reg_pts": round(reg_pts, 2),
            "playoff_pts": round(playoff_pts, 2),
            "total_pts": round(total_pts, 2),
            "active_games": active_games,
            "ppg": round(total_pts / active_games, 2) if active_games > 0 else 0.0,
            "reg_ppg": round(reg_pts / reg_games, 2) if reg_games > 0 else 0.0,
            "playoff_ppg": round(playoff_pts / po_games, 2) if po_games > 0 else 0.0,
            "bye_weeks": bye_weeks,
            "next_bye": bye_weeks[0] if bye_weeks else None,
            "return_week": return_week,
        }

    return {
        "weeks": weeks,
        "weekly_projections": weekly_projections,
        "ros_totals": ros_totals,
    }
