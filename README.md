# Sleeper Redraft Trade Simulator (Public Edition)

Standalone, multi-user-safe Sleeper-only edition of the Trade Simulator & Package Finder.
Any visitor can enter a public Sleeper username (or numeric League ID), pick one of their NFL leagues from the dropdown, and run:

- **5,000-sim Common Random Numbers (CRN)** remaining regular-season + playoff Monte Carlo simulations under league-specific Sleeper scoring settings.
- **Imbalanced Trade Waiver Calibration** powered by FantasyCalc redraft trade values (`1-for-2`, `2-for-1`, `3-for-1`, `3-for-2`, etc.).
- **Playoff Advancement Odds & Trade Finder** (`Mutual Champ %`, `Maximize My Champ %`, `Win Now`).
- **Trade Sandbox & Persuasion Playbook** with week-by-week optimal lineup projections and natural Sleeper DM scripts.

No ESPN integration, cookies, or private credentials are included in this edition.

---

## 1. Run Locally (Windows / macOS / Linux)

From inside the `sleeper_public` directory:

```bash
python -m backend.server --port 8060
```

Then open `http://127.0.0.1:8060` in your browser.

Optional CLI flags:
- `--host 0.0.0.0` (bind address; defaults to `0.0.0.0` / `$HOST`)
- `--port 8060` (listen port; defaults to `8060` / `$PORT`)
- `--preload-user <sleeper_username>` (optional warmup on startup)

---

## 2. Share Immediately From Your PC (Quick Tunnel)

To get a public HTTPS URL immediately while running on your computer on port `8060`:

### Option A: Cloudflare Tunnel (no account required for quick tunnels)
```bash
cloudflared tunnel --url http://127.0.0.1:8060
```
Cloudflare will print a public `https://*.trycloudflare.com` URL you can share with league mates right away.

### Option B: ngrok
```bash
ngrok http 8060
```

---

## 3. Deploy Permanently (Free / Managed Hosting)

This `sleeper_public/` folder is completely self-contained and ready to push to its own GitHub repository (or deploy directly via Dockerfile / Procfile):

### Render (recommended — uses `render.yaml`)
1. Push the contents of `sleeper_public/` to a GitHub repo.
2. In Render, click **New -> Blueprint** (or **New Web Service**), connect the repo, and deploy.
3. Build command: `pip install -r requirements.txt`
4. Start command: `python -m backend.server --host 0.0.0.0 --port $PORT`
*(Note: Free Render instances sleep after 15 min of inactivity; first cold sync fetches the Sleeper NFL players catalog ~10 MB once and caches it locally in `data/cache.db`.)*

### Railway
1. Create a new project from your repo containing `sleeper_public/`.
2. Railway automatically picks up `Procfile` (`web: python -m backend.server --host 0.0.0.0 --port $PORT`) and `requirements.txt`.
3. Generate a public domain in Railway Service Settings -> Networking.

### Fly.io (Docker)
```bash
fly launch
fly deploy
```
Uses `Dockerfile` directly and serves on `0.0.0.0:$PORT`.
