---
name: cogames-softmax-online
description: Query online policy scores, match history, and download/watch replays from the Softmax tournament server. Use this to check how your offline-trained policies perform in the live tournament.
argument-hint: "[--season <season>] [--policy <name>] [--match <id>] [--download-replay <match-id>]"
---

This skill retrieves online performance data and replays from the Softmax tournament server.

## Setup: cogames binary path

The cogames binary lives at:
```bash
COGAMES=/Users/lessandro/Projects/softmax/cogames/.venv/bin/cogames
```

Use this path explicitly if `cogames` isn't on PATH:
```bash
$COGAMES --help
```

## Authentication

**One-time setup**: run `cogames login` once in a terminal. It opens a browser, you log in at softmax.com, and the token is saved permanently to `~/.metta/cogames.yaml`. All future agent sessions use that saved token automatically — no browser needed again.

```bash
$COGAMES login
# Opens browser → log in at softmax.com → token saved to ~/.metta/cogames.yaml
```

If running on a different machine (cloud, CI), set `COGAMES_TOKEN` env var instead:
```bash
# On your local machine — extract the token once:
python3 -c "
import yaml
cfg = yaml.safe_load(open('$HOME/.metta/cogames.yaml'))
print(cfg['login_tokens']['https://softmax.com/api'])
"
# Copy the output, then on the remote machine:
export COGAMES_TOKEN="<paste token here>"
```

All scripts in this skill check `COGAMES_TOKEN` env var first, then fall back to `~/.metta/cogames.yaml`. No browser needed on either path.

Check current auth status:
```bash
python3 -c "
import yaml, os
token = os.environ.get('COGAMES_TOKEN')
if token:
    print(f'Auth: env var (token={token[:8]}...)')
else:
    try:
        cfg = yaml.safe_load(open(os.path.expanduser('~/.metta/cogames.yaml')))
        t = cfg['login_tokens']['https://softmax.com/api']
        print(f'Auth: file (~/.metta/cogames.yaml, token={t[:8]}...)')
    except Exception as e:
        print(f'Not authenticated: {e}')
        print('Run: cogames login')
"
```

---

## Step 1: Check available seasons

```bash
$COGAMES seasons
$COGAMES seasons --json
```

---

## Step 2: Check your submitted policies

```bash
# All uploaded policies
$COGAMES submissions

# In a specific season
$COGAMES submissions --season beta-cogsguard

# JSON for scripting
$COGAMES submissions --json
```

---

## Step 3: Check the leaderboard

```bash
$COGAMES leaderboard
$COGAMES leaderboard --season beta-cogsguard
$COGAMES leaderboard --season beta-cogsguard --json
```

---

## Step 4: Get match history and episode data

**Important**: `cogames matches` requires cogames ≥0.4.2 (your fork at `/Users/lessandro/Projects/cogames`). If the installed binary doesn't support it, use the API directly:

### Option A — `cogames matches` (if available, cogames ≥0.4.2)

```bash
# Install the fork first if needed:
# $COGAMES_VENV_PIP install -e /Users/lessandro/Projects/cogames

$COGAMES matches --json > /tmp/my_matches.json
$COGAMES matches <MATCH_ID> --json > /tmp/match_detail.json
```

### Option B — Direct API call (always works after login)

```python
import yaml, httpx, json

# Load auth token
config = yaml.safe_load(open(os.path.expanduser("~/.metta/cogames.yaml")))
token = config["login_tokens"]["https://softmax.com/api"]

headers = {"X-Auth-Token": token}
server = "https://api.observatory.softmax-research.net"

# Get my policy versions
resp = httpx.get(f"{server}/stats/policy-versions", params={"mine": "true", "limit": 100}, headers=headers)
my_policies = resp.json()["entries"]
print(json.dumps(my_policies[:3], indent=2))

# Get my matches for the default season
my_pv_ids = [p["id"] for p in my_policies]
season = "beta-cogsguard"  # or check /tournament/seasons for the current default
resp = httpx.get(
    f"{server}/tournament/seasons/{season}/matches",
    params={"policy_version_ids": my_pv_ids, "limit": 20},
    headers=headers
)
matches = resp.json()
print(json.dumps(matches[:2], indent=2))
```

Save this as a script and run:
```bash
python3 /tmp/get_matches.py > /tmp/my_matches.json
```

---

## Step 5: Extract replay URL from episode

Each match has an `episode_id`. Fetch it from `/episodes/{episode_id}` — it returns `replay_url` directly.

**Note**: `match["episode"]` is always `null` in the match response. Always use the episodes endpoint.

```python
import yaml, httpx, os

token = os.environ.get("COGAMES_TOKEN") or \
    yaml.safe_load(open(os.path.expanduser("~/.metta/cogames.yaml")))["login_tokens"]["https://softmax.com/api"]
server = "https://api.observatory.softmax-research.net"
headers = {"X-Auth-Token": token}

episode_id = "<EPISODE_UUID>"  # from match["episode_id"]

ep = httpx.get(f"{server}/episodes/{episode_id}", headers=headers).json()
replay_url = ep["replay_url"]
print(f"Replay: {replay_url}")
# → https://softmax-public.s3.amazonaws.com/replays/<id>.json.z
```

The replay URL is a public S3 URL — no auth required to download it.

---

## Step 6: Download and watch the replay

```bash
REPLAY_URL="<S3 URL from episode object>"

# Download
curl -L "$REPLAY_URL" -o docs/autoresearch_director/online_replay.json.z

# Watch in GUI (requires nim + mettascope)
$COGAMES replay docs/autoresearch_director/online_replay.json.z
```

Or open in browser (no installation needed):
```
https://metta-ai.github.io/metta/mettascope/mettascope.html?replay=REPLAY_URL
```

---

## Step 7: Headless replay analysis

Use `capture_frames.py` on the downloaded file if you need programmatic frame analysis:

```bash
# Check if capture_frames supports local replay files
python scripts/capture_frames.py --help 2>&1 | grep -i "replay\|file\|input"
```

If not supported, use the browser MettaScope URL or analyze the JSON directly:
```bash
# The .json.z is zlib-compressed JSON — decompress to inspect:
python3 -c "
import zlib, json
data = zlib.decompress(open('docs/autoresearch_director/online_replay.json.z', 'rb').read())
replay = json.loads(data)
print('top-level keys:', list(replay.keys()))
"
```

---

## Full diagnostic script

Save as `/tmp/softmax_online_check.py` and run with the softmax venv python:

```python
import yaml, httpx, json, os, sys

# Load token — env var takes priority, then ~/.metta/cogames.yaml
def load_token():
    t = os.environ.get("COGAMES_TOKEN")
    if t:
        print(f"Auth: env var (token={t[:8]}...)")
        return t
    try:
        cfg = yaml.safe_load(open(os.path.expanduser("~/.metta/cogames.yaml")))
        t = cfg["login_tokens"]["https://softmax.com/api"]
        print(f"Auth: file (token={t[:8]}...)")
        return t
    except Exception as e:
        print(f"Not authenticated: {e}")
        print("Fix: run 'cogames login' once, or set COGAMES_TOKEN env var")
        sys.exit(1)

token = load_token()

server = "https://api.observatory.softmax-research.net"
headers = {"X-Auth-Token": token}

# Seasons
seasons = httpx.get(f"{server}/tournament/seasons", headers=headers).json()
print(f"\nSeasons: {[s['name'] for s in seasons]}")

# My policies
entries = httpx.get(f"{server}/stats/policy-versions",
    params={"mine": "true", "limit": 100}, headers=headers).json().get("entries", [])
print(f"\nMy policies ({len(entries)}):")
for e in entries:
    print(f"  {e['name']}:v{e['version']}  uploaded={e['created_at'][:10]}")

# Leaderboard for first season
if seasons:
    season = seasons[0]["name"]
    lb = httpx.get(f"{server}/tournament/seasons/{season}/leaderboard",
        params={"include_hidden": "true"}, headers=headers).json()
    print(f"\nLeaderboard ({season}, top 10):")
    my_ids = {e["id"] for e in entries}
    for entry in lb[:10]:
        marker = " ← YOU" if entry["policy"]["id"] in my_ids else ""
        print(f"  #{entry['rank']}  {entry['policy']['name']}:v{entry['policy']['version']}  score={entry['score']:.2f}{marker}")

    # My matches
    if entries:
        pv_ids = [e["id"] for e in entries]
        matches = httpx.get(f"{server}/tournament/seasons/{season}/matches",
            params={"policy_version_ids": pv_ids, "limit": 10, "include_hidden": "true"},
            headers=headers).json()
        print(f"\nRecent matches ({season}):")
        for m in matches:
            ep_id = m.get("episode_id", "none")
            scores = [(p["policy"]["name"], p.get("score")) for p in m["players"]]
            print(f"  {str(m['id'])[:8]}  status={m['status']}  episode={str(ep_id)[:8] if ep_id else 'none'}")
            for name, score in scores:
                print(f"    {name}  score={score}")
```

Run:
```bash
/Users/lessandro/Projects/softmax/cogames/.venv/bin/python /tmp/softmax_online_check.py
```
