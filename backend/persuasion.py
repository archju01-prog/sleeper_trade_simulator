"""Behavioral-psychology-driven trade persuasion and authentic Sleeper DM generator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from .lineup import solve_weekly_lineup

SYNTH_STREAMING_BASELINES: Dict[str, Dict[int, float]] = {
    "QB": {0: 15.0},
    "RB": {0: 7.0},
    "WR": {0: 7.5},
    "TE": {0: 6.0},
    "K": {0: 7.8},
    "DEF": {0: 6.8},
}

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.IGNORECASE)

# Recognizable fantasy shorthand / nicknames used naturally in league chats
_NICKNAME_MAP: Dict[str, str] = {
    "christian mccaffrey": "CMC",
    "jaxon smith-njigba": "JSN",
    "bijan robinson": "Bijan",
    "javonte williams": "Javonte",
    "ceedee lamb": "CeeDee",
    "saquon barkley": "Saquon",
    "chuba hubbard": "Chuba",
    "puka nacua": "Puka",
    "breece hall": "Breece",
    "amon-ra st. brown": "Amon-Ra",
    "de'von achane": "Achane",
    "ja'marr chase": "Chase",
    "brock bowers": "Bowers",
    "tony pollard": "Pollard",
    "tee higgins": "Higgins",
    "drake london": "London",
    "isaiah likely": "Likely",
    "rashee rice": "Rice",
    "alvin kamara": "Kamara",
    "justin jefferson": "Jefferson",
    "george kittle": "Kittle",
    "travis kelce": "Kelce",
    "trey mcbride": "McBride",
    "patrick mahomes": "Mahomes",
    "joe burrow": "Burrow",
    "jahmyr gibbs": "Gibbs",
    "tyreek hill": "Tyreek",
    "kyren williams": "Kyren",
    "derrick henry": "Henry",
    "lamar jackson": "Lamar",
    "jalen hurts": "Hurts",
    "malik nabers": "Nabers",
    "nicolas collins": "Nico",
    "nico collins": "Nico",
}

# Common surnames or ambiguous names where real managers keep "First Last"
_KEEP_FIRST_LAST_SURNAMES: Set[str] = {
    "allen",
    "brown",
    "clark",
    "davis",
    "evans",
    "gordon",
    "hall",
    "harris",
    "hill",
    "jackson",
    "johnson",
    "jones",
    "lewis",
    "martin",
    "metcalf",
    "miller",
    "moore",
    "murray",
    "robinson",
    "smith",
    "taylor",
    "thomas",
    "tucker",
    "walker",
    "warren",
    "watson",
    "white",
    "williams",
    "wilson",
    "young",
}


def _solve_full_strength_detail(
    roster_ids: List[str],
    reserve_ids: Set[str],
    ros_totals: Dict[str, Dict[str, Any]],
    players_master: Dict[str, Dict[str, Any]],
    starter_slots: List[str],
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Solve full-strength optimal lineup and enrich starter/bench dicts with full_name and ppg."""
    synth_proj: Dict[str, Dict[int, float]] = {
        pid: {0: float((ros_totals.get(pid) or {}).get("ppg") or 0.0)}
        for pid in roster_ids
    }
    total_pts, starters, bench = solve_weekly_lineup(
        roster_ids,
        0,
        synth_proj,
        players_master,
        SYNTH_STREAMING_BASELINES,
        starter_slots,
        reserve_ids=reserve_ids,
    )
    for item in starters + bench:
        pid = item["player_id"]
        pinfo = players_master.get(pid) or {}
        item["full_name"] = pinfo.get("full_name", pid)
        item["ppg"] = float(item["points"])
    return total_pts, starters, bench


def _short_name(full_name: str) -> str:
    """Strip suffixes for internal coaching notes."""
    return _SUFFIX_RE.sub("", str(full_name or "").strip())


def _casual_name(full_name: str) -> str:
    """Format player name the way a real manager types it in Sleeper chat."""
    clean = _SUFFIX_RE.sub("", str(full_name or "").strip())
    low = clean.lower()
    if low in _NICKNAME_MAP:
        return _NICKNAME_MAP[low]

    parts = clean.split()
    if len(parts) <= 1:
        return clean
    first, last = parts[0], parts[-1]
    if len(first) <= 2 and first.isupper():
        return f"{first} {last}"
    if last.lower() in _KEEP_FIRST_LAST_SURNAMES:
        return f"{first} {last}"
    return last


def _casual_pkg(items: List[Dict[str, Any]], sep: str = " + ") -> str:
    return sep.join(_casual_name(x["full_name"]) for x in items)


def _join_notes(items: List[Dict[str, Any]]) -> str:
    names = [_short_name(x["full_name"]) for x in items]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _partner_archetype(
    partner_base: Dict[str, Any],
    partner_team: Dict[str, Any],
) -> Tuple[str, str]:
    po_pct = float(partner_base.get("playoff_pct") or 0.0)
    bye_pct = float(partner_base.get("bye_pct") or 0.0)
    wins = int(partner_team.get("wins") or 0)
    losses = int(partner_team.get("losses") or 0)

    if po_pct >= 80.0:
        if bye_pct < 60.0:
            return "contender_bye", f"Playoff Lock Hunting 1st-Rd Bye ({wins}-{losses}, {po_pct:.0f}% PO / {bye_pct:.0f}% Bye)"
        return "contender_title", f"Championship Contender Optimizing Wks 15–17 ({wins}-{losses}, {po_pct:.0f}% PO)"
    if po_pct >= 20.0:
        return "bubble", f"Playoff Bubble Contender ({wins}-{losses}, {po_pct:.1f}% Playoff Odds)"
    return "underdog", f"Must-Win-Out Stretch-Run Roster ({wins}-{losses})"


def generate_trade_persuasion(
    my_team: Dict[str, Any],
    partner_team: Dict[str, Any],
    my_base: Dict[str, Any],
    partner_base: Dict[str, Any],
    send_a: List[Dict[str, Any]],
    send_b: List[Dict[str, Any]],
    tx: Dict[str, Any],
    perceived: Dict[str, Any],
    impact_a: Dict[str, Any],
    impact_b: Dict[str, Any],
    league_info: Dict[str, Any],
    ros_totals: Dict[str, Dict[str, Any]],
    players_master: Dict[str, Dict[str, Any]],
    starter_slots: List[str],
) -> Dict[str, Any]:
    """Generate internal behavioral notes and brief, natural Sleeper DM pitches."""
    my_res = set(my_team.get("reserve") or [])
    p_res = set(partner_team.get("reserve") or [])

    _, _, pre_a_bn = _solve_full_strength_detail(
        my_team["players"], my_res, ros_totals, players_master, starter_slots
    )
    pre_b_tot, pre_b_st, _ = _solve_full_strength_detail(
        partner_team["players"], p_res, ros_totals, players_master, starter_slots
    )
    post_b_tot, post_b_st, _ = _solve_full_strength_detail(
        tx["final_b_ids"], tx["reserve_b"], ros_totals, players_master, starter_slots
    )

    send_a_pids = {p["player_id"] for p in send_a}
    send_b_pids = {p["player_id"] for p in send_b}

    pre_a_bn_pids = {x["player_id"] for x in pre_a_bn}
    bench_sent_by_me = [p for p in send_a if p["player_id"] in pre_a_bn_pids]

    post_b_st_pids = {x["player_id"] for x in post_b_st if not x["is_streamer"]}
    incoming_starters_b = [x for x in post_b_st if x["player_id"] in send_a_pids]
    traded_away_starters_b = [x for x in pre_b_st if x["player_id"] in send_b_pids]
    displaced_weak_b = [
        x for x in pre_b_st
        if not x["is_streamer"]
        and x["player_id"] not in send_b_pids
        and x["player_id"] not in post_b_st_pids
    ]

    incoming_starters_b.sort(key=lambda x: x["ppg"], reverse=True)
    traded_away_starters_b.sort(key=lambda x: x["ppg"], reverse=True)
    displaced_weak_b.sort(key=lambda x: x["ppg"])

    d_full_b = float(impact_b.get("d_full_ppg") or (post_b_tot - pre_b_tot))
    partner_gain_pct = float(perceived.get("partner_gain_pct") or 0.0)

    old_slots_b = traded_away_starters_b + displaced_weak_b
    new_slots_b = incoming_starters_b

    if old_slots_b and new_slots_b and len(old_slots_b) == len(new_slots_b):
        old_sum = sum(x["ppg"] for x in old_slots_b)
        new_sum = sum(x["ppg"] for x in new_slots_b)
        old_desc = " + ".join(f"{_short_name(x['full_name'])} ({x['ppg']:.1f})" for x in old_slots_b)
        new_desc = " + ".join(f"{_short_name(x['full_name'])} ({x['ppg']:.1f})" for x in new_slots_b)
        replaced_note = (
            f"Replaces {old_desc} [{old_sum:.1f} PPG] with {new_desc} [{new_sum:.1f} PPG] "
            f"({new_sum - old_sum:+.1f} PPG/wk)"
        )
    elif displaced_weak_b and new_slots_b:
        weakest = displaced_weak_b[0]
        replaced_note = (
            f"Moves {_short_name(weakest['full_name'])} ({weakest['ppg']:.1f} PPG) out of starting "
            f"{weakest['slot']} slot; net Full-Strength Lineup change: {d_full_b:+.2f} PPG/wk"
        )
    else:
        replaced_note = (
            f"Full-Strength Starting Lineup moves {pre_b_tot:.1f} -> {post_b_tot:.1f} PPG "
            f"({d_full_b:+.2f} PPG/wk)"
        )

    arch_key, arch_label = _partner_archetype(partner_base, partner_team)
    use_median = bool(league_info.get("use_median_scoring", True))
    trade_deadline = int(league_info.get("trade_deadline") or 11)
    current_week = int(league_info.get("current_week") or 11)

    sorted_b_sent = sorted(send_b, key=lambda x: x["ppg"], reverse=True)
    sorted_a_sent = sorted(send_a, key=lambda x: x["ppg"], reverse=True)
    their_stud = sorted_b_sent[0]
    your_lead = sorted_a_sent[0]

    byes_b_shed = [p for p in send_b if p.get("next_bye")]
    byes_a_sent = [p for p in send_a if p.get("next_bye")]
    shed_bye = byes_b_shed[0] if byes_b_shed and len(byes_b_shed) > len(byes_a_sent) else None

    # Internal strategy bullets (analytical context for the user only)
    hooks: List[str] = []
    if d_full_b < -0.15 or partner_gain_pct < -10.0:
        hooks.append(
            f"Caution: This deal drops {partner_team['team_name']}'s full-strength lineup "
            f"by {d_full_b:+.2f} PPG ({partner_gain_pct:+.1f}% FC). Consider sweetening before sending."
        )
    if displaced_weak_b:
        w0 = displaced_weak_b[0]
        hooks.append(
            f"Weak-Link Relief: Gets {_short_name(w0['full_name'])} ({w0['ppg']:.1f} PPG in {w0['slot']}) out of their starting lineup."
        )
    elif d_full_b > 0.2:
        hooks.append(
            f"Starting Lineup Upgrade: Adds {d_full_b:+.2f} PPG to their full-strength lineup."
        )
    if bench_sent_by_me:
        hooks.append(
            f"Cover Story: {_join_notes(bench_sent_by_me[:2])} sit on your bench/surplus, making a consolidation overpay believable."
        )
    if use_median and arch_key in ("bubble", "contender_bye"):
        hooks.append("League Median Angle: Raising their weekly floor protects against median losses down the stretch.")
    if partner_gain_pct >= 2.5:
        hooks.append(f"Market Value Edge: +{partner_gain_pct:.1f}% in partner's favor on FantasyCalc (waiver-adjusted).")
    if shed_bye:
        hooks.append(f"Bye Bonus: Partner avoids {_short_name(shed_bye['full_name'])}'s Week {shed_bye['next_bye']} bye.")

    if len(send_a) > len(send_b):
        if displaced_weak_b:
            w0 = displaced_weak_b[0]
            objection_handler = (
                f"If they say they don't want to move {_casual_name(their_stud['full_name'])}: agree he's the best single player, "
                f"and point out it's really {_casual_name(their_stud['full_name'])} + {_casual_name(w0['full_name'])} vs {_casual_pkg(send_a)}."
            )
        else:
            objection_handler = (
                f"If they hesitate on {_casual_name(their_stud['full_name'])}: validate his value, "
                f"and note their total starting lineup comes out ahead ({d_full_b:+.1f} PPG/wk)."
            )
    elif len(send_a) < len(send_b):
        objection_handler = (
            f"If they feel they're giving up too much depth: remind them they get the best player ({_casual_name(your_lead['full_name'])}) "
            "plus an open roster spot."
        )
    else:
        objection_handler = (
            f"If they hesitate on the swap: keep it focused on filling both starting lineups ({d_full_b:+.1f} PPG/wk for them)."
        )

    # --- Natural, succinct Sleeper messages (no em-dashes, no decimals, no sales essays) ---
    pkg_a_cas = _casual_pkg(send_a)
    pkg_b_cas = _casual_pkg(send_b)
    stud_cas = _casual_name(their_stud["full_name"])
    lead_cas = _casual_name(your_lead["full_name"])

    # Determine concise position / consolidation motive
    pos_counts_a: Dict[str, int] = {}
    for p in send_a:
        pos_counts_a[p["position"]] = pos_counts_a.get(p["position"], 0) + 1
    main_pos = max(pos_counts_a.items(), key=lambda kv: kv[1])[0] if pos_counts_a else "RB"

    deadline_phrase = f"before the wk {trade_deadline} deadline" if current_week <= trade_deadline else "for the stretch run"

    # 1) DM Probe (Ask First - before sending an official trade in Sleeper)
    if len(send_a) > len(send_b):
        if displaced_weak_b:
            weak_cas = _casual_name(displaced_weak_b[0]["full_name"])
            weak_pos = displaced_weak_b[0]["slot"]
            probe_text = (
                f"hey any chance you'd move {pkg_b_cas} {deadline_phrase}? "
                f"trying to consolidate my {main_pos}s, could do {pkg_a_cas} if you want an upgrade over {weak_cas} at {weak_pos}"
            )
        else:
            probe_text = (
                f"hey any interest in moving {pkg_b_cas} {deadline_phrase}? "
                f"trying to consolidate some bench depth, could put {pkg_a_cas} on the table if you need extra starters"
            )
    elif len(send_a) < len(send_b):
        probe_text = (
            f"hey any chance you're looking to consolidate for a stud? "
            f"could do {lead_cas} if you'd want to work something around {pkg_b_cas}"
        )
    else:
        if displaced_weak_b:
            weak_cas = _casual_name(displaced_weak_b[0]["full_name"])
            probe_text = (
                f"yo any interest in swapping {pkg_a_cas} for {pkg_b_cas}? "
                f"looks like it helps your {displaced_weak_b[0]['slot']} spot and fits what I need too"
            )
        else:
            probe_text = (
                f"yo any interest in {pkg_a_cas} for {pkg_b_cas}? "
                "looks like our rosters match up pretty well positional wise"
            )

    # 2) Trade Offer Note (1-liner in Sleeper trade chat when submitting)
    if len(send_a) > len(send_b):
        if bench_sent_by_me:
            note_text = (
                f"sent an offer over - got a logjam on my bench so willing to move {pkg_a_cas} for {pkg_b_cas}. "
                "lmk if we're close"
            )
        else:
            note_text = (
                f"sent an offer over - trying to consolidate {pkg_a_cas} into {pkg_b_cas} so you get multiple weekly starters. "
                "no worries if not"
            )
    else:
        note_text = (
            f"sent an offer over with {pkg_a_cas} for {pkg_b_cas}. "
            "lmk what you think or feel free to counter"
        )

    # 3) Follow-Up (Only if they push back on moving their top piece)
    if old_slots_b and new_slots_b and len(old_slots_b) == len(new_slots_b) and len(send_a) > len(send_b):
        old_cas = _casual_pkg(old_slots_b)
        new_cas = _casual_pkg(new_slots_b)
        bye_add = f" (plus you skip {stud_cas}'s wk {shed_bye['next_bye']} bye)" if shed_bye else ""
        counter_text = (
            f"totally get it on {stud_cas}, mostly figured {new_cas} outscores {old_cas} for your starting lineup down the stretch{bye_add}. "
            "down to tweak if someone else on my squad fits better"
        )
    elif displaced_weak_b:
        weak_cas = _casual_name(displaced_weak_b[0]["full_name"])
        bye_add = f" and you dodge {stud_cas}'s wk {shed_bye['next_bye']} bye" if shed_bye else ""
        counter_text = (
            f"fair enough on {stud_cas}, just thought getting {weak_cas} out of your starting lineup helped your weekly floor{bye_add}. "
            "lmk who else you'd prefer from my roster"
        )
    else:
        bye_add = f" plus you avoid his wk {shed_bye['next_bye']} bye" if shed_bye else ""
        counter_text = (
            f"totally fair on {stud_cas}{bye_add}. "
            "if there's a different combo on my bench you like better lmk"
        )

    platform_label = "ESPN" if league_info.get("platform") == "espn" else "Sleeper"

    return {
        "partner_archetype": arch_label,
        "replaced_starters_note": replaced_note,
        "psychological_hooks": hooks,
        "objection_handler": objection_handler,
        "messages": [
            {
                "id": "probe",
                "label": "1. DM Probe (Ask First)",
                "tone": f"Best opener before sending the offer in {platform_label} - low pressure, asks a question",
                "text": probe_text,
            },
            {
                "id": "note",
                "label": "2. Offer Chat Note",
                "tone": f"Quick 1-liner to drop with the official {platform_label} offer",
                "text": note_text,
            },
            {
                "id": "counter",
                "label": "3. If They Hesitate",
                "tone": "Casual follow-up only if they push back on trading their top player",
                "text": counter_text,
            },
        ],
    }
