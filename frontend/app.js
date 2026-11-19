(() => {
  "use strict";

  let appState = null;
  let activeSessionKey = null;
  let currentUser = { user_id: null, username: "", display_name: "", leagues: [] };
  let lastFoundTrades = [];
  const mustIncludeSet = new Set();
  const excludeSet = new Set();

  // Sandbox state
  let sandboxSendA = new Set();
  let sandboxSendB = new Set();
  let sandboxLastEval = null;
  let sandboxActiveWeekSide = "A";

  function esc(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtSigned(num, digits = 2, suffix = "") {
    if (num == null || Number.isNaN(Number(num))) return "—";
    const val = Number(num);
    const sign = val > 0 ? "+" : "";
    return `${sign}${val.toFixed(digits)}${suffix}`;
  }

  function deltaClass(num, threshold = 0.005) {
    const val = Number(num || 0);
    if (val > threshold) return "pos-delta";
    if (val < -threshold) return "neg-delta";
    return "neu-delta";
  }

  async function apiJson(url, options = {}) {
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `Request failed (${resp.status})`);
    }
    return data;
  }

  function switchTab(tabName) {
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${tabName}`);
    });
  }

  function getTeamById(rid) {
    if (!appState) return null;
    return appState.teams.find((t) => Number(t.roster_id) === Number(rid)) || null;
  }

  function populateLeagueSelect(leagues, selectedLeagueId = null) {
    const sel = document.getElementById("leagueSelect");
    if (!leagues || !leagues.length) {
      sel.innerHTML = `<option value="">Enter username &amp; click Find Leagues...</option>`;
      sel.disabled = true;
      document.getElementById("syncBtn").disabled = true;
      return;
    }
    sel.innerHTML = "";
    leagues.forEach((lg) => {
      const opt = document.createElement("option");
      opt.value = lg.league_id;
      const medTag = lg.use_median_scoring ? " · Median" : "";
      opt.textContent = `${lg.name} (${lg.season} · ${lg.total_rosters}T ${lg.ppr_label} · ${lg.qb_label}${medTag})`;
      sel.appendChild(opt);
    });
    sel.disabled = false;
    document.getElementById("syncBtn").disabled = false;

    if (selectedLeagueId && leagues.some((l) => String(l.league_id) === String(selectedLeagueId))) {
      sel.value = String(selectedLeagueId);
    } else {
      sel.value = String(leagues[0].league_id);
    }
  }

  function persistUserUrlState(username, leagueId) {
    try {
      if (username) localStorage.setItem("sleeper_public_username", username);
      if (leagueId) localStorage.setItem("sleeper_public_league_id", leagueId);
      const url = new URL(window.location.href);
      if (username) url.searchParams.set("user", username);
      if (leagueId) url.searchParams.set("league", leagueId);
      window.history.replaceState(null, "", url.toString());
    } catch (_) {
      // Ignore storage or history restrictions
    }
  }

  function renderWelcomeOnboarding(errorMsg = "") {
    document.getElementById("finderTelemetry").style.display = "none";
    document.getElementById("runFinderBtn").disabled = true;
    const errHtml = errorMsg
      ? `<div style="margin-bottom:14px;padding:10px 14px;border-radius:8px;background:var(--red-bg);border:1px solid rgba(244,63,94,0.4);color:#fda4af;font-size:13px;">${esc(errorMsg)}</div>`
      : "";
    document.getElementById("finderResults").innerHTML = `
      <div class="onboarding-card">
        <div class="hero-team-label" style="margin-bottom:6px;">Public Sleeper Trade Engine</div>
        <h2 class="onboarding-title">Find Mutually Winning Trades in Any Sleeper League</h2>
        <p class="onboarding-sub">
          Enter your public Sleeper username (or numeric League ID) to load your leagues, run 5,000 Monte Carlo schedule + playoff simulations under custom scoring, adjust for imbalanced waiver spots via FantasyCalc, and generate natural trade pitches.
        </p>
        ${errHtml}
        <form class="onboarding-form" id="onboardingQuickForm">
          <input type="text" id="onboardingUsernameInput" placeholder="Enter your Sleeper username..." value="${esc(document.getElementById("sleeperUsername").value)}" />
          <button type="submit" class="btn btn-primary">Find My Leagues</button>
          <button type="button" class="btn btn-secondary" id="tryDemoBtn">Try Demo League (archju01)</button>
        </form>
        <div style="font-size:12px;color:var(--text-dim);">No login or password required — Sleeper league data is read-only and public.</div>
      </div>
    `;
    const quickForm = document.getElementById("onboardingQuickForm");
    if (quickForm) {
      quickForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const u = document.getElementById("onboardingUsernameInput").value.trim();
        document.getElementById("sleeperUsername").value = u;
        await lookupUserLeagues(u, null, false);
      });
    }
    const demoBtn = document.getElementById("tryDemoBtn");
    if (demoBtn) {
      demoBtn.addEventListener("click", async () => {
        document.getElementById("sleeperUsername").value = "archju01";
        await lookupUserLeagues("archju01", "1385431008031014912", false);
      });
    }
  }

  function renderHeaderAndSelectors() {
    const lg = appState.league;
    const fc = appState.fc_params || {};

    if (appState.available_leagues && appState.available_leagues.length) {
      currentUser.leagues = appState.available_leagues;
      populateLeagueSelect(appState.available_leagues, lg.league_id);
    } else if (!document.getElementById("leagueSelect").value && lg.league_id) {
      populateLeagueSelect(
        [
          {
            league_id: lg.league_id,
            name: lg.league_name,
            season: lg.season,
            total_rosters: lg.total_rosters,
            ppr_label: fc.ppr === "1" ? "Full PPR" : fc.ppr === "0.5" ? "Half-PPR" : "Std",
            qb_label: fc.numQbs === "2" ? "Superflex" : "1QB",
            use_median_scoring: Boolean(lg.use_median_scoring),
          },
        ],
        lg.league_id
      );
    }

    const pprLabel = fc.ppr === "1" ? "Full PPR" : fc.ppr === "0.5" ? "Half-PPR" : "Std";
    const qbLabel = fc.numQbs === "2" ? "Superflex/2QB" : "1QB";
    const medPill = lg.use_median_scoring ? `<span class="pill-tag">Median Scoring ON</span>` : "";
    const byeLabel = lg.neutralize_byes !== false ? "Bye Protection 85%" : "Raw Byes";

    document.getElementById("leagueTitle").textContent = `[Sleeper] ${lg.league_name} · ${lg.season} Redraft`;
    document.getElementById("leagueMeta").innerHTML = `
      <span class="pill-tag">${lg.total_rosters} Teams · ${qbLabel} · ${pprLabel}</span>
      ${medPill}
      <span class="pill-tag">${byeLabel}</span>
      <span class="pill-tag">Sim Weeks ${lg.sim_weeks[0]}–${lg.sim_weeks[lg.sim_weeks.length - 1]} (${lg.num_sims.toLocaleString()} CRN Sims)</span>
      <span class="pill-tag">Sync ${appState.sync_ms} ms</span>
    `;

    document.getElementById("syncMedianToggle").checked = Boolean(lg.use_median_scoring);
    document.getElementById("syncByeToggle").checked = lg.neutralize_byes !== false;
    document.getElementById("syncBtn").disabled = false;
    document.getElementById("runFinderBtn").disabled = false;

    const winNowOpt = document.querySelector('#finderMode option[value="win_now"]');
    if (winNowOpt) {
      winNowOpt.textContent = lg.use_median_scoring
        ? "Win Now (Reg Season H2H + Median)"
        : "Win Now (Remaining Regular Season)";
    }

    const wkSel = document.getElementById("syncStartWeek");
    const prevWk = wkSel.value;
    wkSel.innerHTML = `<option value="">Auto (Wk ${lg.current_week})</option>`;
    for (let w = 1; w <= 17; w++) {
      const opt = document.createElement("option");
      opt.value = String(w);
      opt.textContent = `From Week ${w}`;
      wkSel.appendChild(opt);
    }
    if (prevWk) wkSel.value = prevWk;

    const sortedTeams = [...appState.teams].sort((a, b) => b.champ_pct - a.champ_pct);
    const mySel = document.getElementById("finderMyTeam");
    const partnerSel = document.getElementById("finderPartner");
    const sbASel = document.getElementById("sandboxTeamA");
    const sbBSel = document.getElementById("sandboxTeamB");

    const leagueKey = `sleeper:${lg.league_id}:${lg.my_roster_id}`;
    const leagueChanged = mySel.dataset.leagueKey !== leagueKey;
    const curMy = leagueChanged ? lg.my_roster_id : (Number(mySel.value) || lg.my_roster_id);
    const curPartner = leagueChanged ? "" : partnerSel.value;

    mySel.innerHTML = "";
    partnerSel.innerHTML = `<option value="">All ${lg.total_rosters - 1} League Partners</option>`;
    sbASel.innerHTML = "";
    sbBSel.innerHTML = "";

    sortedTeams.forEach((t) => {
      const label = `${t.team_name} (${t.wins}-${t.losses}, Champ ${t.champ_pct.toFixed(1)}%)`;
      [mySel, sbASel, sbBSel].forEach((sel) => {
        const opt = document.createElement("option");
        opt.value = String(t.roster_id);
        opt.textContent = label;
        sel.appendChild(opt);
      });
      const pOpt = document.createElement("option");
      pOpt.value = String(t.roster_id);
      pOpt.textContent = label;
      partnerSel.appendChild(pOpt);
    });

    mySel.dataset.leagueKey = leagueKey;
    mySel.value = String(curMy);
    if (curPartner && Number(curPartner) !== Number(curMy)) {
      partnerSel.value = curPartner;
    }

    if (leagueChanged || !sbASel.dataset.initialized) {
      sbASel.value = String(lg.my_roster_id);
      const other = sortedTeams.find((t) => Number(t.roster_id) !== Number(lg.my_roster_id));
      if (other) sbBSel.value = String(other.roster_id);
      sbASel.dataset.initialized = "1";
      const myT = getTeamById(curMy);
      const isLongshot = Boolean(myT && myT.champ_pct < 1.5);
      document.getElementById("finderMode").value = isLongshot ? "win_now" : "mutual_champ";
      document.getElementById("finderRequireBothChamp").checked = !isLongshot;
    }

    renderHeroBanner();
    populateConstraintPickers();
  }

  function renderHeroBanner() {
    const myRid = Number(document.getElementById("finderMyTeam").value || appState.league.my_roster_id);
    const t = getTeamById(myRid);
    if (!t) return;

    document.getElementById("heroTeamName").textContent = t.team_name;
    document.getElementById("heroRecord").textContent =
      `Manager: ${t.manager_name} · Record: ${t.wins}-${t.losses}${t.ties ? `-${t.ties}` : ""} · ` +
      `PF: ${t.points_for.toFixed(1)} · Avg Optimal Lineup: ${t.avg_weekly_proj.toFixed(1)} pts/wk`;
    document.getElementById("heroExpWins").textContent = `${t.exp_wins.toFixed(1)}-${t.exp_losses.toFixed(1)}`;
    document.getElementById("heroPlayoffPct").textContent = `${t.playoff_pct.toFixed(1)}%`;
    document.getElementById("heroByePct").textContent = `${t.bye_pct.toFixed(1)}%`;
    document.getElementById("heroChampPct").textContent = `${t.champ_pct.toFixed(1)}%`;

    const posKeys = ["QB", "RB", "WR", "TE", "FLEX"];
    const ranksHtml = posKeys
      .map((pos) => {
        const rank = t.pos_ranks[pos] || 12;
        const ppg = t.pos_strength[pos] || 0;
        const cls = rank <= 4 ? "rank-top" : rank <= 8 ? "rank-mid" : "rank-low";
        return `
          <div class="pos-rank-chip">
            <div class="pr-pos">${pos}</div>
            <div class="pr-val ${cls}">#${rank}</div>
            <div class="pr-ppg">${ppg.toFixed(1)} ppg</div>
          </div>
        `;
      })
      .join("");
    document.getElementById("heroPosRanks").innerHTML = ranksHtml;
  }

  function populateConstraintPickers() {
    const myRid = Number(document.getElementById("finderMyTeam").value || appState.league.my_roster_id);
    const partnerVal = document.getElementById("finderPartner").value;
    const mustSel = document.getElementById("finderMustInclude");
    const exclSel = document.getElementById("finderExclude");

    mustSel.innerHTML = `<option value="">+ Lock a player into trade...</option>`;
    exclSel.innerHTML = `<option value="">+ Exclude / mark untouchable...</option>`;

    const teamsToList = appState.teams.filter((t) => {
      if (Number(t.roster_id) === myRid) return true;
      if (!partnerVal) return true;
      return Number(t.roster_id) === Number(partnerVal);
    });

    teamsToList.sort((a, b) => (Number(a.roster_id) === myRid ? -1 : Number(b.roster_id) === myRid ? 1 : a.team_name.localeCompare(b.team_name)));

    teamsToList.forEach((t) => {
      const grpMust = document.createElement("optgroup");
      grpMust.label = t.team_name;
      const grpExcl = document.createElement("optgroup");
      grpExcl.label = t.team_name;

      t.roster
        .filter((p) => ["QB", "RB", "WR", "TE"].includes(p.position))
        .forEach((p) => {
          const text = `${p.full_name} (${p.position}, FC ${Math.round(p.fc_value)}, ${p.ppg.toFixed(1)} ppg)`;
          const o1 = document.createElement("option");
          o1.value = p.player_id;
          o1.textContent = text;
          grpMust.appendChild(o1);

          const o2 = document.createElement("option");
          o2.value = p.player_id;
          o2.textContent = text;
          grpExcl.appendChild(o2);
        });

      mustSel.appendChild(grpMust);
      exclSel.appendChild(grpExcl);
    });
  }

  function findPlayerName(pid) {
    if (!appState) return pid;
    for (const t of appState.teams) {
      const p = t.roster.find((x) => x.player_id === pid);
      if (p) return `${p.full_name} (${p.position})`;
    }
    return pid;
  }

  function renderConstraintTags() {
    const el = document.getElementById("constraintTags");
    const chips = [];
    mustIncludeSet.forEach((pid) => {
      chips.push(`<span class="chip active" data-remove-must="${esc(pid)}">MUST: ${esc(findPlayerName(pid))} &times;</span>`);
    });
    excludeSet.forEach((pid) => {
      chips.push(`<span class="chip" style="border-color:var(--red);color:#fda4af;background:var(--red-bg);" data-remove-excl="${esc(pid)}">EXCL: ${esc(findPlayerName(pid))} &times;</span>`);
    });
    if (!chips.length) {
      el.innerHTML = `<span style="font-size:12px;color:var(--text-dim);">No player locks or exclusions active</span>`;
    } else {
      el.innerHTML = chips.join("");
    }
  }

  function renderPlayerRowHtml(p) {
    const inj = p.injury_status ? `<span class="injury-tag">${esc(p.injury_status)}</span>` : "";
    const byeTag = p.next_bye ? `<span class="pill-tag" style="color:var(--amber);border-color:rgba(245,158,11,0.45);margin-left:4px;">BYE W${p.next_bye}</span>` : "";
    return `
      <div class="player-row">
        <div class="player-left">
          <span class="pos-pill pos-${esc(p.position)}">${esc(p.position)}</span>
          <div style="min-width:0;">
            <div class="player-name">${esc(p.full_name)} ${inj}${byeTag}</div>
            <div class="player-sub">${esc(p.team)} · ROS ${p.ros_pts.toFixed(0)} pts · PO ${p.playoff_pts.toFixed(0)}</div>
          </div>
        </div>
        <div class="player-metrics">
          <div class="metric-chip">
            <span class="mc-val">${p.ppg.toFixed(1)}</span>
            <span class="mc-lbl">PPG</span>
          </div>
          <div class="metric-chip">
            <span class="mc-val" style="color:var(--accent);">${Math.round(p.fc_value).toLocaleString()}</span>
            <span class="mc-lbl">FC Val</span>
          </div>
        </div>
      </div>
    `;
  }

  function renderTradeScoreboardHtml(tr, isSandbox = false) {
    const perc = tr.perceived;
    const ia = isSandbox ? tr.team_a : tr.impact_a;
    const ib = isSandbox ? tr.team_b : tr.impact_b;
    const nameA = isSandbox ? tr.team_a.team_name : tr.my_team_name;
    const nameB = isSandbox ? tr.team_b.team_name : tr.partner_team_name;

    const totalBar = Math.max(1, perc.adjusted_a + perc.adjusted_b);
    const pctA = Math.max(12, Math.min(88, (perc.adjusted_a / totalBar) * 100));
    const pctB = 100 - pctA;

    const txNotes = [];
    if (tr.adds_a && tr.adds_a.length) {
      txNotes.push(`<span><strong>${esc(nameA)}</strong> adds from waivers: ${tr.adds_a.map((p) => `${esc(p.full_name)} (${p.ppg.toFixed(1)} ppg)`).join(", ")}</span>`);
    }
    if (tr.drops_a && tr.drops_a.length) {
      txNotes.push(`<span><strong>${esc(nameA)}</strong> drops: ${tr.drops_a.map((p) => `${esc(p.full_name)}`).join(", ")}</span>`);
    }
    if (tr.adds_b && tr.adds_b.length) {
      txNotes.push(`<span><strong>${esc(nameB)}</strong> adds from waivers: ${tr.adds_b.map((p) => `${esc(p.full_name)} (${p.ppg.toFixed(1)} ppg)`).join(", ")}</span>`);
    }
    if (tr.drops_b && tr.drops_b.length) {
      txNotes.push(`<span><strong>${esc(nameB)}</strong> drops: ${tr.drops_b.map((p) => `${esc(p.full_name)}`).join(", ")}</span>`);
    }

    const txStrip = txNotes.length
      ? `<div class="roster-tx-strip"><strong>Simulated Roster Transactions:</strong> ${txNotes.join(" · ")}</div>`
      : "";

    function fmtProgCell(delta, baseVal, postVal, digits = 2, suffix = "%", threshold = 0.01) {
      const dStr = fmtSigned(delta, digits, suffix);
      const cls = deltaClass(delta, threshold);
      const sub =
        baseVal != null && postVal != null
          ? `<span class="sub-prog">${Number(baseVal).toFixed(digits === 2 && suffix === "%" ? 1 : digits)}${suffix} &rarr; ${Number(postVal).toFixed(digits === 2 && suffix === "%" ? 1 : digits)}${suffix}</span>`
          : "";
      return `<td class="${cls}">${dStr}${sub}</td>`;
    }

    return `
      ${txStrip}
      <div class="scoreboard-grid">
        <div class="score-panel">
          <div class="score-panel-title">Playoff Progression &amp; Simulation Impact (Sleeper Projections · 5,000 CRN Sims)</div>
          <table class="impact-table">
            <thead>
              <tr>
                <th>Team</th>
                <th>Playoffs %</th>
                <th>1st-Rd Bye %</th>
                <th>Adv. Semis %</th>
                <th>Adv. Finals %</th>
                <th>Win Champ %</th>
                <th>Exp Wins</th>
                <th>Full-Str PPG</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>${esc(nameA)}</td>
                ${fmtProgCell(ia.d_playoff_pct, ia.base_playoff_pct, ia.post_playoff_pct, 2, "%", 0.01)}
                ${fmtProgCell(ia.d_bye_pct, ia.base_bye_pct, ia.post_bye_pct, 2, "%", 0.01)}
                ${fmtProgCell(ia.d_semis_pct, ia.base_semis_pct, ia.post_semis_pct, 2, "%", 0.01)}
                ${fmtProgCell(ia.d_finals_pct, ia.base_finals_pct, ia.post_finals_pct, 2, "%", 0.01)}
                ${fmtProgCell(ia.d_champ_pct, ia.base_champ_pct, ia.post_champ_pct, 2, "%", 0.001)}
                ${fmtProgCell(ia.d_exp_wins, ia.base_exp_wins, ia.post_exp_wins, 2, "", 0.005)}
                <td class="${deltaClass(ia.d_full_ppg, 0.01)}">${fmtSigned(ia.d_full_ppg, 2)}<span class="sub-prog">Sched ${fmtSigned(ia.d_starter_ppg, 2)}</span></td>
              </tr>
              <tr>
                <td>${esc(nameB)}</td>
                ${fmtProgCell(ib.d_playoff_pct, ib.base_playoff_pct, ib.post_playoff_pct, 2, "%", 0.01)}
                ${fmtProgCell(ib.d_bye_pct, ib.base_bye_pct, ib.post_bye_pct, 2, "%", 0.01)}
                ${fmtProgCell(ib.d_semis_pct, ib.base_semis_pct, ib.post_semis_pct, 2, "%", 0.01)}
                ${fmtProgCell(ib.d_finals_pct, ib.base_finals_pct, ib.post_finals_pct, 2, "%", 0.01)}
                ${fmtProgCell(ib.d_champ_pct, ib.base_champ_pct, ib.post_champ_pct, 2, "%", 0.001)}
                ${fmtProgCell(ib.d_exp_wins, ib.base_exp_wins, ib.post_exp_wins, 2, "", 0.005)}
                <td class="${deltaClass(ib.d_full_ppg, 0.01)}">${fmtSigned(ib.d_full_ppg, 2)}<span class="sub-prog">Sched ${fmtSigned(ib.d_starter_ppg, 2)}</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="score-panel">
          <div class="score-panel-title">Perceived Value &amp; Acceptance (FantasyCalc + Waiver Adj)</div>
          <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:700;">
            <span>${esc(nameA)} Sends: ${Math.round(perc.adjusted_a).toLocaleString()}</span>
            <span>${esc(nameB)} Sends: ${Math.round(perc.adjusted_b).toLocaleString()}</span>
          </div>
          <div class="perceived-bar-wrap">
            <div class="perceived-bar-track">
              <div class="perceived-bar-a" style="width:${pctA.toFixed(1)}%;"></div>
              <div class="perceived-bar-b" style="width:${pctB.toFixed(1)}%;"></div>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--text-muted);gap:8px;flex-wrap:wrap;">
            <span>Partner Perceived Edge: <strong class="${deltaClass(perc.partner_gain_pct, 0.1)}">${fmtSigned(perc.partner_gain_pct, 1, "%")}</strong> (${fmtSigned(perc.partner_net_gain, 0)} FC)</span>
            <span class="fairness-badge tier-${esc(perc.fairness_tier)}">${esc(perc.fairness_label)}</span>
          </div>
        </div>
      </div>
    `;
  }

  function renderPersuasionPlaybookHtml(persuasion, scopeId) {
    if (!persuasion || !persuasion.messages || !persuasion.messages.length) return "";
    const msgs = persuasion.messages;
    const activeMsg = msgs[0];
    const hooksHtml = (persuasion.psychological_hooks || [])
      .map((h) => `<li>${esc(h)}</li>`)
      .join("");
    const tabsHtml = msgs
      .map(
        (m, idx) =>
          `<button type="button" class="pitch-tab ${idx === 0 ? "active" : ""}" data-scope="${esc(scopeId)}" data-msg-id="${esc(m.id)}">${esc(m.label)}</button>`
      )
      .join("");
    const msgMapJson = esc(JSON.stringify(msgs));

    return `
      <div class="persuasion-playbook" data-playbook-scope="${esc(scopeId)}" data-messages="${msgMapJson}">
        <div class="persuasion-header">
          <strong style="font-size:13.5px;color:var(--accent);">Negotiation &amp; Persuasion Playbook</strong>
          <span class="pill-tag">${esc(persuasion.partner_archetype || "")}</span>
        </div>
        ${persuasion.replaced_starters_note ? `<div class="persuasion-swap-note"><strong>Partner Lineup Upgrade:</strong> ${esc(persuasion.replaced_starters_note)}</div>` : ""}
        ${hooksHtml ? `<ul class="psych-hooks-list">${hooksHtml}</ul>` : ""}
        ${persuasion.objection_handler ? `<div class="objection-box"><strong>Objection Handler:</strong> ${esc(persuasion.objection_handler)}</div>` : ""}
        <div class="pitch-tabs">${tabsHtml}</div>
        <div class="pitch-box">
          <div class="pitch-tone">${esc(activeMsg.tone)}</div>
          <div class="pitch-text">${esc(activeMsg.text)}</div>
          <div class="pitch-actions">
            <button type="button" class="btn btn-sm btn-primary copy-pitch-btn" data-scope="${esc(scopeId)}">Copy Message</button>
          </div>
        </div>
      </div>
    `;
  }

  async function copyTextToClipboard(text, btn) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      const prev = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = prev;
      }, 1400);
    } catch (_) {
      btn.textContent = "Copy failed";
    }
  }

  function handlePlaybookClick(e) {
    const tabBtn = e.target.closest(".pitch-tab");
    if (tabBtn) {
      const scope = tabBtn.dataset.scope;
      const msgId = tabBtn.dataset.msgId;
      const box = document.querySelector(`[data-playbook-scope="${CSS.escape(scope)}"]`);
      if (!box) return;
      const msgs = JSON.parse(box.dataset.messages || "[]");
      const chosen = msgs.find((m) => m.id === msgId) || msgs[0];
      if (!chosen) return;
      box.querySelectorAll(".pitch-tab").forEach((b) => b.classList.toggle("active", b.dataset.msgId === msgId));
      box.querySelector(".pitch-tone").textContent = chosen.tone;
      box.querySelector(".pitch-text").textContent = chosen.text;
      return true;
    }
    const copyBtn = e.target.closest(".copy-pitch-btn");
    if (copyBtn) {
      const scope = copyBtn.dataset.scope;
      const box = document.querySelector(`[data-playbook-scope="${CSS.escape(scope)}"]`);
      if (!box) return;
      const text = box.querySelector(".pitch-text").textContent;
      copyTextToClipboard(text, copyBtn);
      return true;
    }
    return false;
  }

  function renderTradeCards(data) {
    const telEl = document.getElementById("finderTelemetry");
    const listEl = document.getElementById("finderResults");
    const tel = data.telemetry;
    lastFoundTrades = data.trades || [];

    telEl.style.display = "flex";
    telEl.innerHTML = `
      <span>Funnel: <strong>${tel.stage1_combinations_checked.toLocaleString()}</strong> combos &rarr; <strong>${tel.stage1_value_band_survivors.toLocaleString()}</strong> in perceived window &rarr; <strong>${tel.stage2_analytical_survivors}</strong> analytically mutual &rarr; <strong>${tel.stage3_mc_evaluated}</strong> Monte Carlo sims &rarr; <strong style="color:var(--green);">${tel.stage3_mutual_verified}</strong> verified mutual trades</span>
      <span>Completed in <strong>${tel.total_ms.toFixed(0)} ms</strong></span>
    `;

    if (!lastFoundTrades.length) {
      listEl.innerHTML = `
        <div class="empty-state">
          <div style="font-size:15px;font-weight:700;color:var(--text);margin-bottom:6px;">No mutually beneficial trades matched these exact filters</div>
          <div>Try widening the Perceived Edge window (e.g. -15% to +25%), enabling all package sizes, or unchecking "Require Both Teams Champ % &uarr;" for Win-Now bubble partners.</div>
        </div>
      `;
      return;
    }

    listEl.innerHTML = lastFoundTrades
      .map((tr, idx) => {
        const perc = tr.perceived;
        const partnerTeam = getTeamById(tr.partner_roster_id);
        const partnerMeta = partnerTeam
          ? `${partnerTeam.wins}-${partnerTeam.losses} · Playoff ${partnerTeam.playoff_pct.toFixed(1)}% · Champ ${partnerTeam.champ_pct.toFixed(1)}%`
          : "";

        const waiverLineA =
          perc.smaller_package_side === "A" && perc.waiver_adj_value > 0
            ? `<div class="waiver-adj-line"><span>+ League Waiver Spot Adjustment (${perc.extra_spots} ${perc.extra_spots === 1 ? "spot" : "spots"})</span><strong>+${Math.round(perc.waiver_adj_value).toLocaleString()} FC</strong></div>`
            : "";

        const waiverLineB =
          perc.smaller_package_side === "B" && perc.waiver_adj_value > 0
            ? `<div class="waiver-adj-line"><span>+ League Waiver Spot Adjustment (${perc.extra_spots} ${perc.extra_spots === 1 ? "spot" : "spots"})</span><strong>+${Math.round(perc.waiver_adj_value).toLocaleString()} FC</strong></div>`
            : "";

        return `
          <article class="trade-card">
            <div class="trade-card-header">
              <div class="trade-title-group">
                <span class="trade-rank-badge">#${idx + 1}</span>
                <span class="struct-badge">${esc(tr.structure)}</span>
                <div>
                  <strong style="font-size:15px;">vs. ${esc(tr.partner_team_name)}</strong>
                  <span style="font-size:12px;color:var(--text-muted);margin-left:8px;">${esc(partnerMeta)}</span>
                </div>
              </div>
              <span class="fairness-badge tier-${esc(perc.fairness_tier)}">${fmtSigned(perc.partner_gain_pct, 1, "%")} Partner · ${esc(perc.fairness_label)}</span>
            </div>

            <div class="trade-sides-grid">
              <div class="trade-side-box">
                <div class="side-header">
                  <span>${esc(tr.my_team_name)} Sends</span>
                  <span>Raw ${Math.round(perc.raw_a).toLocaleString()} &rarr; Adj ${Math.round(perc.adjusted_a).toLocaleString()} FC</span>
                </div>
                ${tr.send_a.map(renderPlayerRowHtml).join("")}
                ${waiverLineA}
              </div>

              <div class="trade-side-box">
                <div class="side-header">
                  <span>${esc(tr.partner_team_name)} Sends</span>
                  <span>Raw ${Math.round(perc.raw_b).toLocaleString()} &rarr; Adj ${Math.round(perc.adjusted_b).toLocaleString()} FC</span>
                </div>
                ${tr.send_b.map(renderPlayerRowHtml).join("")}
                ${waiverLineB}
              </div>
            </div>

            ${renderTradeScoreboardHtml(tr, false)}

            <div class="trade-card-footer">
              <div style="font-size:12px;color:var(--text-dim);">
                Post-Trade Champ Odds: You <strong>${tr.impact_a.post_champ_pct.toFixed(2)}%</strong> · Partner <strong>${tr.impact_b.post_champ_pct.toFixed(2)}%</strong>
              </div>
              <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <button type="button" class="btn btn-sm persuasion-toggle-btn" data-trade-idx="${idx}">
                  Show Persuasion Pitch &amp; Copy DM
                </button>
                <button type="button" class="btn btn-sm btn-primary load-sandbox-btn" data-trade-idx="${idx}">
                  Inspect Week-by-Week &amp; Load in Sandbox &rarr;
                </button>
              </div>
            </div>
            <div class="persuasion-drawer" id="finder-persuasion-${idx}" style="display:none;">
              ${renderPersuasionPlaybookHtml(tr.persuasion, `finder-${idx}`)}
            </div>
          </article>
        `;
      })
      .join("");
  }

  async function runTradeFinder() {
    if (!appState) return;
    const btn = document.getElementById("runFinderBtn");
    const listEl = document.getElementById("finderResults");
    btn.disabled = true;
    listEl.innerHTML = `<div class="empty-state"><div class="loading-pulse"><span class="spinner"></span>Scanning multi-player packages with FantasyCalc waiver adjustments &amp; 5,000 CRN simulations...</div></div>`;

    const selectedSizes = Array.from(document.querySelectorAll("#packageSizeChips .chip.active")).map((c) => c.dataset.size);
    const body = {
      session_key: activeSessionKey,
      league_id: appState.league.league_id,
      username: currentUser.username,
      user_id: currentUser.user_id,
      my_roster_id: Number(document.getElementById("finderMyTeam").value || appState.league.my_roster_id),
      partner_roster_id: document.getElementById("finderPartner").value || null,
      package_sizes: selectedSizes.length ? selectedSizes : ["1-1", "2-1", "1-2", "2-2"],
      min_perceived_pct: Number(document.getElementById("finderMinPct").value || -12),
      max_perceived_pct: Number(document.getElementById("finderMaxPct").value || 20),
      mode: document.getElementById("finderMode").value,
      must_include: Array.from(mustIncludeSet),
      exclude: Array.from(excludeSet),
      require_both_champ_up: document.getElementById("finderRequireBothChamp").checked,
      allow_same_pos_1for1: document.getElementById("finderAllowSamePos").checked,
    };

    try {
      const data = await apiJson("/api/trades/find", {
        method: "POST",
        body: JSON.stringify(body),
      });
      renderTradeCards(data);
    } catch (err) {
      listEl.innerHTML = `<div class="empty-state" style="color:#fda4af;">Trade scan failed: ${esc(err.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  // SANDBOX
  function renderSandboxRosters() {
    const ridA = Number(document.getElementById("sandboxTeamA").value);
    const ridB = Number(document.getElementById("sandboxTeamB").value);
    const teamA = getTeamById(ridA);
    const teamB = getTeamById(ridB);

    function renderSideList(team, selectedSet, containerId, infoId) {
      const container = document.getElementById(containerId);
      const infoEl = document.getElementById(infoId);
      if (!team) {
        container.innerHTML = "";
        return;
      }
      const chosen = team.roster.filter((p) => selectedSet.has(p.player_id));
      infoEl.innerHTML = chosen.length
        ? `Selected (${chosen.length}): <strong style="color:var(--accent);">${chosen.map((p) => esc(p.full_name)).join(", ")}</strong>`
        : `Click players below to include in ${esc(team.team_name)}'s outgoing package.`;

      container.innerHTML = team.roster
        .map((p) => {
          const isSel = selectedSet.has(p.player_id);
          const irTag = p.is_ir ? `<span class="pill-tag" style="margin-left:4px;">IR</span>` : "";
          const byeTag = p.next_bye ? `<span class="pill-tag" style="color:var(--amber);border-color:rgba(245,158,11,0.45);margin-left:4px;">BYE W${p.next_bye}</span>` : "";
          return `
            <div class="roster-pick-item ${isSel ? "selected" : ""}" data-pid="${esc(p.player_id)}">
              <div class="player-left">
                <span class="pos-pill pos-${esc(p.position)}">${esc(p.position)}</span>
                <div>
                  <div class="player-name">${esc(p.full_name)} ${irTag}${byeTag}</div>
                  <div class="player-sub">${esc(p.team)} · ROS ${p.ros_pts.toFixed(0)} pts · PO ${p.playoff_pts.toFixed(0)}</div>
                </div>
              </div>
              <div class="player-metrics">
                <div class="metric-chip"><span class="mc-val">${p.ppg.toFixed(1)}</span><span class="mc-lbl">PPG</span></div>
                <div class="metric-chip"><span class="mc-val" style="color:var(--accent);">${Math.round(p.fc_value).toLocaleString()}</span><span class="mc-lbl">FC</span></div>
              </div>
            </div>
          `;
        })
        .join("");
    }

    renderSideList(teamA, sandboxSendA, "sandboxRosterA", "sandboxSelectedAInfo");
    renderSideList(teamB, sandboxSendB, "sandboxRosterB", "sandboxSelectedBInfo");

    const sumEl = document.getElementById("sandboxSummaryLine");
    if (sandboxSendA.size > 0 && sandboxSendB.size > 0) {
      sumEl.textContent = `Package ready: ${sandboxSendA.size}-for-${sandboxSendB.size}`;
    } else {
      sumEl.textContent = `Select at least one player on each side to simulate.`;
    }
  }

  function renderSandboxEvaluation(ev) {
    sandboxLastEval = ev;
    const el = document.getElementById("sandboxResults");
    const perc = ev.perceived;
    const activeTeam = sandboxActiveWeekSide === "A" ? ev.team_a : ev.team_b;

    const waiverBanner =
      perc.waiver_adj_value > 0
        ? `<div class="waiver-adj-line" style="margin-bottom:14px;"><span>Imbalanced ${esc(ev.structure)} Waiver Spot Adjustment applied to Side ${esc(perc.smaller_package_side)} (${perc.extra_spots} ${perc.extra_spots === 1 ? "spot" : "spots"})</span><strong>+${Math.round(perc.waiver_adj_value).toLocaleString()} FC Value</strong></div>`
        : "";

    const useMed = Boolean(appState && appState.league && appState.league.use_median_scoring);

    const weekRows = activeTeam.weekly
      .map((wk) => {
        const lbl1 = wk.phase === "Playoff" && wk.prob_label ? `<span class="sub-prog">${esc(wk.prob_label)}</span>` : "";
        const lbl2 = wk.phase === "Playoff" && wk.prob2_label ? `<span class="sub-prog">${esc(wk.prob2_label)}</span>` : "";
        const h2hStr =
          wk.base_h2h_pct != null
            ? `${wk.base_h2h_pct.toFixed(1)}% &rarr; <strong>${wk.post_h2h_pct.toFixed(1)}%</strong> <span class="${deltaClass(wk.d_h2h_pct, 0.05)}">(${fmtSigned(wk.d_h2h_pct, 2, "%")})</span>${lbl1}`
            : "—";
        const showCol5 = (useMed || wk.phase === "Playoff") && wk.base_med_pct != null;
        const medStr = showCol5
          ? `${wk.base_med_pct.toFixed(1)}% &rarr; <strong>${wk.post_med_pct.toFixed(1)}%</strong> <span class="${deltaClass(wk.d_med_pct, 0.05)}">(${fmtSigned(wk.d_med_pct, 2, "%")})</span>${lbl2}`
          : "—";
        const slotDisplayOrder = {
          QB: 0,
          RB: 1,
          WR: 2,
          TE: 3,
          WRRB_FLEX: 4,
          REC_FLEX: 5,
          FLEX: 6,
          SUPER_FLEX: 7,
          K: 8,
          DEF: 9,
        };
        const orderedStarters = [...(wk.post_starters || [])].sort(
          (a, b) => (slotDisplayOrder[a.slot] ?? 99) - (slotDisplayOrder[b.slot] ?? 99)
        );
        const topStarters = orderedStarters
          .map((s) => `${esc(s.slot)}:${esc(s.full_name.split(" ").slice(-1)[0])}(${s.points.toFixed(1)})`)
          .join(" · ");
        return `
          <tr>
            <td class="left"><strong>Wk ${wk.week}</strong> <span class="pill-tag">${esc(wk.phase)}</span></td>
            <td class="left">${esc(wk.opponent_name)}</td>
            <td>${wk.base_pts.toFixed(1)} &rarr; <strong>${wk.post_pts.toFixed(1)}</strong> <span class="${deltaClass(wk.d_pts, 0.05)}">(${fmtSigned(wk.d_pts, 1)})</span></td>
            <td>${h2hStr}</td>
            <td>${medStr}</td>
            <td class="left" style="font-size:11.5px;color:var(--text-muted);white-space:normal;">${topStarters}</td>
          </tr>
        `;
      })
      .join("");

    el.innerHTML = `
      <div class="card" style="margin-bottom:18px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
          <div>
            <span class="struct-badge">${esc(ev.structure)}</span>
            <strong style="margin-left:8px;font-size:16px;">${esc(ev.team_a.team_name)} &harr; ${esc(ev.team_b.team_name)}</strong>
          </div>
          <span class="fairness-badge tier-${esc(perc.fairness_tier)}">${fmtSigned(perc.partner_gain_pct, 1, "%")} Partner · ${esc(perc.fairness_label)}</span>
        </div>
        ${waiverBanner}
        ${renderTradeScoreboardHtml(ev, true)}
      </div>

      ${ev.persuasion ? `<div class="card" style="margin-bottom:18px;">${renderPersuasionPlaybookHtml(ev.persuasion, "sandbox")}</div>` : ""}

      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
          <div>
            <h3 style="margin:0;font-size:15.5px;">Week-by-Week Optimal Lineup &amp; Win Probability Impact</h3>
            <div style="font-size:12px;color:var(--text-muted);">Before vs. After simulation across remaining regular-season (${useMed ? "H2H + Median" : "H2H"}) and playoff weeks.</div>
          </div>
          <div class="chip-group">
            <span class="chip ${sandboxActiveWeekSide === "A" ? "active" : ""}" data-sb-side="A">${esc(ev.team_a.team_name)}</span>
            <span class="chip ${sandboxActiveWeekSide === "B" ? "active" : ""}" data-sb-side="B">${esc(ev.team_b.team_name)}</span>
          </div>
        </div>
        <div class="data-table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th class="left">Week</th>
                <th class="left">Opponent</th>
                <th>Optimal Proj Pts</th>
                <th>H2H Win / Round Adv %</th>
                <th>${useMed ? "Median Win / Stage %" : "Playoff Stage %"}</th>
                <th class="left">Post-Trade Optimal Starters</th>
              </tr>
            </thead>
            <tbody>${weekRows}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  async function evaluateSandbox() {
    if (!appState) return;
    const ridA = Number(document.getElementById("sandboxTeamA").value);
    const ridB = Number(document.getElementById("sandboxTeamB").value);
    const resultsEl = document.getElementById("sandboxResults");

    if (!ridA || !ridB || ridA === ridB) {
      resultsEl.innerHTML = `<div class="empty-state">Choose two different teams to evaluate a trade.</div>`;
      return;
    }
    if (sandboxSendA.size === 0 || sandboxSendB.size === 0) {
      resultsEl.innerHTML = `<div class="empty-state">Select at least one player from each team.</div>`;
      return;
    }

    resultsEl.innerHTML = `<div class="empty-state"><div class="loading-pulse"><span class="spinner"></span>Running 5,000 CRN simulations &amp; week-by-week matchup deltas...</div></div>`;
    try {
      const ev = await apiJson("/api/trades/evaluate", {
        method: "POST",
        body: JSON.stringify({
          session_key: activeSessionKey,
          league_id: appState.league.league_id,
          username: currentUser.username,
          user_id: currentUser.user_id,
          roster_a_id: ridA,
          roster_b_id: ridB,
          send_a: Array.from(sandboxSendA),
          send_b: Array.from(sandboxSendB),
        }),
      });
      renderSandboxEvaluation(ev);
    } catch (err) {
      resultsEl.innerHTML = `<div class="empty-state" style="color:#fda4af;">Evaluation error: ${esc(err.message)}</div>`;
    }
  }

  // STANDINGS & HEATMAP
  function renderStandingsAndHeatmap() {
    const teams = [...appState.teams].sort((a, b) => b.champ_pct - a.champ_pct || b.playoff_pct - a.playoff_pct || b.exp_wins - a.exp_wins);
    document.getElementById("standingsSimBadge").textContent = `${appState.league.num_sims.toLocaleString()} CRN Sims · ${appState.league.use_median_scoring ? "H2H + Median" : "H2H Only"}`;

    const stHead = `
      <thead>
        <tr>
          <th class="left">#</th>
          <th class="left">Team / Manager</th>
          <th>Current</th>
          <th>Points For</th>
          <th>Exp Record</th>
          <th>Avg Proj/Wk</th>
          <th>Playoff %</th>
          <th>Bye %</th>
          <th>Semis %</th>
          <th>Finals %</th>
          <th>Champ %</th>
        </tr>
      </thead>
    `;
    const stRows = teams
      .map((t, idx) => `
        <tr class="${t.is_my_team ? "my-team-row" : ""}" data-inspect-rid="${t.roster_id}" style="cursor:pointer;">
          <td class="left">${idx + 1}</td>
          <td class="left">
            <strong>${esc(t.team_name)}</strong> ${t.is_my_team ? `<span class="pill-tag" style="color:var(--accent);border-color:var(--accent);">YOU</span>` : ""}
            <div style="font-size:11px;color:var(--text-dim);">${esc(t.manager_name)}</div>
          </td>
          <td>${t.wins}-${t.losses}${t.ties ? `-${t.ties}` : ""}</td>
          <td>${t.points_for.toFixed(1)}</td>
          <td>${t.exp_wins.toFixed(1)}-${t.exp_losses.toFixed(1)}</td>
          <td>${t.avg_weekly_proj.toFixed(1)}</td>
          <td><strong>${t.playoff_pct.toFixed(1)}%</strong></td>
          <td>${t.bye_pct.toFixed(1)}%</td>
          <td>${(t.semis_pct || 0).toFixed(1)}%</td>
          <td>${t.finals_pct.toFixed(1)}%</td>
          <td style="color:${t.champ_pct >= 10 ? "var(--green)" : "var(--text)"};font-weight:800;">${t.champ_pct.toFixed(2)}%</td>
        </tr>
      `)
      .join("");
    document.getElementById("standingsTable").innerHTML = stHead + `<tbody>${stRows}</tbody>`;

    const posCols = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"];
    const hmHead = `
      <thead>
        <tr>
          <th class="left">Team</th>
          ${posCols.map((p) => `<th style="text-align:center;">${p}</th>`).join("")}
        </tr>
      </thead>
    `;
    const hmRows = teams
      .map((t) => {
        const cells = posCols
          .map((pos) => {
            const ppg = t.pos_strength[pos] || 0;
            const rk = t.pos_ranks[pos] || 12;
            const cls = rk <= 4 ? "heat-good" : rk <= 8 ? "heat-mid" : "heat-bad";
            return `<td style="text-align:center;"><span class="heat-cell ${cls}">${ppg.toFixed(1)} (#${rk})</span></td>`;
          })
          .join("");
        return `
          <tr class="${t.is_my_team ? "my-team-row" : ""}" data-inspect-rid="${t.roster_id}" style="cursor:pointer;">
            <td class="left"><strong>${esc(t.team_name)}</strong></td>
            ${cells}
          </tr>
        `;
      })
      .join("");
    document.getElementById("heatmapTable").innerHTML = hmHead + `<tbody>${hmRows}</tbody>`;
  }

  function openTeamModal(rid) {
    const t = getTeamById(rid);
    if (!t) return;
    document.getElementById("teamModalTitle").textContent = `${t.team_name} (${t.wins}-${t.losses}, Champ ${t.champ_pct.toFixed(1)}%)`;
    const rosterRows = t.roster
      .map(
        (p) => `
          <tr>
            <td class="left"><span class="pos-pill pos-${esc(p.position)}">${esc(p.position)}</span></td>
            <td class="left"><strong>${esc(p.full_name)}</strong> ${p.is_ir ? `<span class="pill-tag">IR</span>` : ""} ${p.injury_status ? `<span class="injury-tag">${esc(p.injury_status)}</span>` : ""}</td>
            <td>${esc(p.team)}</td>
            <td>${p.ppg.toFixed(1)}</td>
            <td>${p.ros_pts.toFixed(1)}</td>
            <td>${p.playoff_pts.toFixed(1)}</td>
            <td style="color:var(--accent);font-weight:700;">${Math.round(p.fc_value).toLocaleString()}</td>
          </tr>
        `
      )
      .join("");
    document.getElementById("teamModalBody").innerHTML = `
      <div class="data-table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th class="left">Pos</th>
              <th class="left">Player</th>
              <th>NFL</th>
              <th>ROS PPG</th>
              <th>ROS Pts</th>
              <th>PO Pts</th>
              <th>FC Value</th>
            </tr>
          </thead>
          <tbody>${rosterRows}</tbody>
        </table>
      </div>
    `;
    document.getElementById("teamModal").classList.add("open");
  }

  // VALUES, WAIVERS & ARBITRAGE
  function renderValuesAndArbitrage() {
    const wc = appState.waiver_calibration;
    const fc = appState.fc_params || {};
    document.getElementById("waiverCalibrationBody").innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
        <div class="kpi-box">
          <div class="kpi-label">1-Spot Waiver Adj (2v1 / 3v2)</div>
          <div class="kpi-val" style="color:#d8b4fe;">+${Math.round(wc.waiver_adj_1).toLocaleString()}</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-label">2-Spot Waiver Adj (3v1)</div>
          <div class="kpi-val" style="color:#d8b4fe;">+${Math.round(wc.waiver_adj_2).toLocaleString()}</div>
        </div>
      </div>
      <div style="line-height:1.6;">
        <div><strong>FantasyCalc Sync Settings:</strong> Redraft · ${fc.numQbs === "2" ? "2QB/SF" : "1QB"} · ${fc.numTeams}-Team · ${fc.ppr} PPR · TEP: ${esc(fc.tep)}</div>
        <div><strong>League Bench Drop Calibration:</strong> Median 1st-drop bench FC = <strong>${wc.median_bench_drop_1}</strong> · 2nd-drop = <strong>${wc.median_bench_drop_2}</strong></div>
        <div><strong>Top Unrostered Waiver Avg FC:</strong> <strong>${wc.top_waiver_avg_fc}</strong> · 2nd Spot Marginal Cost: <strong>+${Math.round(wc.marginal_2)}</strong></div>
      </div>
    `;

    const wRows = (wc.top_waivers || [])
      .map(
        (w) => `
          <tr>
            <td class="left"><span class="pos-pill pos-${esc(w.position)}">${esc(w.position)}</span> <strong>${esc(w.full_name)}</strong></td>
            <td>${esc(w.team)}</td>
            <td>${w.ppg.toFixed(1)}</td>
            <td>${w.ros_pts.toFixed(0)}</td>
            <td style="color:var(--accent);font-weight:700;">${Math.round(w.fc_value).toLocaleString()}</td>
          </tr>
        `
      )
      .join("");
    document.getElementById("topWaiversTable").innerHTML = `
      <thead><tr><th class="left">Player</th><th>NFL</th><th>ROS PPG</th><th>ROS Pts</th><th>FC Val</th></tr></thead>
      <tbody>${wRows}</tbody>
    `;

    renderArbitrageTable();
  }

  function renderArbitrageTable() {
    if (!appState) return;
    const q = document.getElementById("arbSearch").value.trim().toLowerCase();
    const posF = document.getElementById("arbPosFilter").value;
    const ownF = document.getElementById("arbOwnerFilter").value;

    const filtered = (appState.arbitrage || []).filter((r) => {
      if (posF !== "ALL" && r.position !== posF) return false;
      if (q && !r.full_name.toLowerCase().includes(q)) return false;
      if (ownF === "MINE" && (!r.owner || !r.owner.is_my_team)) return false;
      if (ownF === "OTHER" && (!r.owner || r.owner.is_my_team)) return false;
      if (ownF === "WAIVER" && r.owner) return false;
      return true;
    });

    const rowsHtml = filtered
      .slice(0, 120)
      .map((r) => {
        const ownerTag = r.owner
          ? r.owner.is_my_team
            ? `<span class="pill-tag" style="color:var(--accent);border-color:var(--accent);">YOU</span>`
            : esc(r.owner.team_name)
          : `<span class="pill-tag">Waivers</span>`;
        const gap = r.rank_gap;
        const gapBadge =
          gap >= 10
            ? `<span class="pos-delta">+${gap} Buy Low</span>`
            : gap <= -10
            ? `<span class="neg-delta">${gap} Sell High</span>`
            : `<span class="neu-delta">${fmtSigned(gap, 0)}</span>`;
        return `
          <tr>
            <td class="left"><span class="pos-pill pos-${esc(r.position)}">${esc(r.position)}</span> <strong>${esc(r.full_name)}</strong> <span style="font-size:11px;color:var(--text-dim);">${esc(r.team)}</span></td>
            <td class="left">${ownerTag}</td>
            <td>${r.ppg.toFixed(1)}</td>
            <td>${r.ros_pts.toFixed(0)}</td>
            <td>${r.playoff_pts.toFixed(0)}</td>
            <td>#${r.sleeper_vorp_rank}</td>
            <td style="color:var(--accent);font-weight:700;">${Math.round(r.fc_value).toLocaleString()}</td>
            <td>#${r.fc_subset_rank}</td>
            <td>${gapBadge}</td>
          </tr>
        `;
      })
      .join("");

    document.getElementById("arbitrageTable").innerHTML = `
      <thead>
        <tr>
          <th class="left">Player</th>
          <th class="left">Owner</th>
          <th>Sleeper PPG</th>
          <th>ROS Pts</th>
          <th>PO Pts</th>
          <th>VORP Rk</th>
          <th>FC Value</th>
          <th>FC Rk</th>
          <th>Arbitrage Signal</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    `;
  }

  function applyState(state) {
    appState = state;
    activeSessionKey = state.session_key || null;
    if (state.user) {
      currentUser.user_id = state.user.user_id || currentUser.user_id;
      currentUser.username = state.user.username || currentUser.username;
    }
    if (currentUser.username && !document.getElementById("sleeperUsername").value) {
      document.getElementById("sleeperUsername").value = currentUser.username;
    }
    renderHeaderAndSelectors();
    renderSandboxRosters();
    renderStandingsAndHeatmap();
    renderValuesAndArbitrage();
  }

  async function lookupUserLeagues(usernameInput, preferredLeagueId = null, forceRefresh = false) {
    const uname = (usernameInput || document.getElementById("sleeperUsername").value || "").trim();
    if (!uname) {
      renderWelcomeOnboarding("Please enter a Sleeper username or League ID.");
      return;
    }
    const findBtn = document.getElementById("findLeaguesBtn");
    findBtn.disabled = true;
    findBtn.textContent = "Finding...";
    if (!appState) {
      document.getElementById("finderResults").innerHTML = `<div class="empty-state"><div class="loading-pulse"><span class="spinner"></span>Looking up Sleeper leagues for <strong>${esc(uname)}</strong>...</div></div>`;
    }
    try {
      const data = await apiJson("/api/user/leagues", {
        method: "POST",
        body: JSON.stringify({ username: uname, force_refresh: forceRefresh }),
      });
      currentUser = {
        user_id: data.user_id || null,
        username: data.username || uname,
        display_name: data.display_name || uname,
        leagues: data.leagues || [],
      };
      populateLeagueSelect(currentUser.leagues, preferredLeagueId);
      await runSync(false, null);
    } catch (err) {
      if (!appState) {
        renderWelcomeOnboarding(err.message);
      } else {
        alert(`Could not find leagues: ${err.message}`);
      }
    } finally {
      findBtn.disabled = false;
      findBtn.textContent = "Find Leagues";
    }
  }

  async function runSync(forceRefresh = true, overrideUseMedian = undefined) {
    const btn = document.getElementById("syncBtn");
    const usernameVal = document.getElementById("sleeperUsername").value.trim() || currentUser.username;
    const leagueId = document.getElementById("leagueSelect").value;

    if (!leagueId) {
      await lookupUserLeagues(usernameVal, null, forceRefresh);
      return;
    }

    btn.disabled = true;
    btn.textContent = "Syncing...";
    const startWkVal = document.getElementById("syncStartWeek").value;
    const useMed =
      overrideUseMedian === undefined
        ? document.getElementById("syncMedianToggle").checked
        : overrideUseMedian;

    if (!appState) {
      document.getElementById("finderResults").innerHTML = `<div class="empty-state"><div class="loading-pulse"><span class="spinner"></span>Syncing league &amp; running 5,000 CRN simulations...</div></div>`;
    }

    try {
      const state = await apiJson("/api/sync", {
        method: "POST",
        body: JSON.stringify({
          username: usernameVal,
          user_id: currentUser.user_id,
          league_id: leagueId,
          start_week: startWkVal ? Number(startWkVal) : null,
          use_median: useMed,
          neutralize_byes: document.getElementById("syncByeToggle").checked,
          force_refresh: forceRefresh,
        }),
      });
      mustIncludeSet.clear();
      excludeSet.clear();
      sandboxSendA.clear();
      sandboxSendB.clear();
      document.getElementById("sandboxResults").innerHTML = "";
      document.getElementById("sandboxTeamA").dataset.initialized = "";
      applyState(state);
      renderConstraintTags();
      persistUserUrlState(usernameVal || currentUser.username, state.league.league_id);
      await runTradeFinder();
    } catch (err) {
      if (!appState) {
        renderWelcomeOnboarding(err.message);
      } else {
        alert(`Sync failed: ${err.message}`);
      }
    } finally {
      btn.disabled = false;
      btn.textContent = "Sync League";
    }
  }

  async function initialLoad() {
    try {
      const params = new URLSearchParams(window.location.search);
      const urlUser = (params.get("user") || "").trim();
      const urlLeague = (params.get("league") || "").trim();
      let savedUser = "";
      let savedLeague = "";
      try {
        savedUser = (localStorage.getItem("sleeper_public_username") || "").trim();
        savedLeague = (localStorage.getItem("sleeper_public_league_id") || "").trim();
      } catch (_) {}

      const initialUser = urlUser || savedUser;
      const initialLeague = urlLeague || (urlUser ? "" : savedLeague);

      if (initialUser) {
        document.getElementById("sleeperUsername").value = initialUser;
        await lookupUserLeagues(initialUser, initialLeague || null, false);
        return;
      }

      const state = await apiJson("/api/state");
      if (state && state.ready) {
        applyState(state);
        await runTradeFinder();
      } else {
        renderWelcomeOnboarding();
      }
    } catch (err) {
      renderWelcomeOnboarding(err.message);
    }
  }

  // Event Listeners
  document.getElementById("mainTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-tab");
    if (btn && btn.dataset.tab) switchTab(btn.dataset.tab);
  });

  document.getElementById("findLeaguesBtn").addEventListener("click", async () => {
    await lookupUserLeagues(document.getElementById("sleeperUsername").value.trim(), null, false);
  });

  document.getElementById("sleeperUsername").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      await lookupUserLeagues(e.target.value.trim(), null, false);
    }
  });

  document.getElementById("leagueSelect").addEventListener("change", async () => {
    await runSync(false, null);
  });

  document.getElementById("syncForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const selVal = document.getElementById("leagueSelect").value;
    if (!selVal) {
      await lookupUserLeagues(document.getElementById("sleeperUsername").value.trim(), null, true);
    } else {
      await runSync(true);
    }
  });

  document.getElementById("finderMyTeam").addEventListener("change", () => {
    renderHeroBanner();
    populateConstraintPickers();
  });

  document.getElementById("finderPartner").addEventListener("change", () => {
    populateConstraintPickers();
  });

  function updateRangeReadout() {
    const mn = document.getElementById("finderMinPct").value;
    const mx = document.getElementById("finderMaxPct").value;
    document.getElementById("perceivedRangeReadout").textContent = `${mn}% to +${mx}%`.replace("+-", "-");
  }
  document.getElementById("finderMinPct").addEventListener("input", updateRangeReadout);
  document.getElementById("finderMaxPct").addEventListener("input", updateRangeReadout);

  document.getElementById("packageSizeChips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) chip.classList.toggle("active");
  });

  document.getElementById("pkgSelectAll").addEventListener("click", () => {
    document.querySelectorAll("#packageSizeChips .chip").forEach((c) => c.classList.add("active"));
  });

  document.getElementById("pkgSelectImbalanced").addEventListener("click", () => {
    document.querySelectorAll("#packageSizeChips .chip").forEach((c) => {
      const [a, b] = c.dataset.size.split("-");
      c.classList.toggle("active", a !== b);
    });
  });

  document.getElementById("finderMustInclude").addEventListener("change", (e) => {
    if (e.target.value) {
      mustIncludeSet.add(e.target.value);
      excludeSet.delete(e.target.value);
      e.target.value = "";
      renderConstraintTags();
    }
  });

  document.getElementById("finderExclude").addEventListener("change", (e) => {
    if (e.target.value) {
      excludeSet.add(e.target.value);
      mustIncludeSet.delete(e.target.value);
      e.target.value = "";
      renderConstraintTags();
    }
  });

  document.getElementById("constraintTags").addEventListener("click", (e) => {
    const mustPid = e.target.dataset.removeMust;
    const exclPid = e.target.dataset.removeExcl;
    if (mustPid) mustIncludeSet.delete(mustPid);
    if (exclPid) excludeSet.delete(exclPid);
    if (mustPid || exclPid) renderConstraintTags();
  });

  document.getElementById("runFinderBtn").addEventListener("click", runTradeFinder);

  document.getElementById("finderResults").addEventListener("click", async (e) => {
    if (handlePlaybookClick(e)) return;

    const toggleBtn = e.target.closest(".persuasion-toggle-btn");
    if (toggleBtn) {
      const idx = toggleBtn.dataset.tradeIdx;
      const drawer = document.getElementById(`finder-persuasion-${idx}`);
      if (drawer) {
        const isHidden = drawer.style.display === "none";
        drawer.style.display = isHidden ? "block" : "none";
        toggleBtn.textContent = isHidden ? "Hide Persuasion Pitch" : "Show Persuasion Pitch & Copy DM";
      }
      return;
    }

    const btn = e.target.closest(".load-sandbox-btn");
    if (!btn) return;
    const tr = lastFoundTrades[Number(btn.dataset.tradeIdx)];
    if (!tr) return;

    document.getElementById("sandboxTeamA").value = String(tr.my_roster_id);
    document.getElementById("sandboxTeamB").value = String(tr.partner_roster_id);
    sandboxSendA = new Set(tr.send_a.map((p) => p.player_id));
    sandboxSendB = new Set(tr.send_b.map((p) => p.player_id));
    sandboxActiveWeekSide = "A";
    renderSandboxRosters();
    switchTab("sandbox");
    await evaluateSandbox();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Sandbox events
  document.getElementById("sandboxTeamA").addEventListener("change", () => {
    sandboxSendA.clear();
    renderSandboxRosters();
  });
  document.getElementById("sandboxTeamB").addEventListener("change", () => {
    sandboxSendB.clear();
    renderSandboxRosters();
  });

  document.getElementById("sandboxRosterA").addEventListener("click", (e) => {
    const item = e.target.closest(".roster-pick-item");
    if (!item) return;
    const pid = item.dataset.pid;
    if (sandboxSendA.has(pid)) sandboxSendA.delete(pid);
    else sandboxSendA.add(pid);
    renderSandboxRosters();
    if (sandboxSendA.size > 0 && sandboxSendB.size > 0) evaluateSandbox();
  });

  document.getElementById("sandboxRosterB").addEventListener("click", (e) => {
    const item = e.target.closest(".roster-pick-item");
    if (!item) return;
    const pid = item.dataset.pid;
    if (sandboxSendB.has(pid)) sandboxSendB.delete(pid);
    else sandboxSendB.add(pid);
    renderSandboxRosters();
    if (sandboxSendA.size > 0 && sandboxSendB.size > 0) evaluateSandbox();
  });

  document.getElementById("sandboxClearBtn").addEventListener("click", () => {
    sandboxSendA.clear();
    sandboxSendB.clear();
    renderSandboxRosters();
    document.getElementById("sandboxResults").innerHTML = "";
  });

  document.getElementById("sandboxEvalBtn").addEventListener("click", evaluateSandbox);

  document.getElementById("sandboxResults").addEventListener("click", (e) => {
    if (handlePlaybookClick(e)) return;
    const chip = e.target.closest("[data-sb-side]");
    if (chip && sandboxLastEval) {
      sandboxActiveWeekSide = chip.dataset.sbSide;
      renderSandboxEvaluation(sandboxLastEval);
    }
  });

  // Standings modal events
  document.getElementById("standingsTable").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-inspect-rid]");
    if (tr) openTeamModal(Number(tr.dataset.inspectRid));
  });
  document.getElementById("heatmapTable").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-inspect-rid]");
    if (tr) openTeamModal(Number(tr.dataset.inspectRid));
  });
  document.getElementById("teamModalClose").addEventListener("click", () => {
    document.getElementById("teamModal").classList.remove("open");
  });
  document.getElementById("teamModal").addEventListener("click", (e) => {
    if (e.target.id === "teamModal") document.getElementById("teamModal").classList.remove("open");
  });

  // Arbitrage filters
  document.getElementById("arbSearch").addEventListener("input", renderArbitrageTable);
  document.getElementById("arbPosFilter").addEventListener("change", renderArbitrageTable);
  document.getElementById("arbOwnerFilter").addEventListener("change", renderArbitrageTable);

  initialLoad();
})();
