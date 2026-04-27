# AmongThem Policy Practice

This walkthrough is the shortest supported path for writing a scripted AmongThem policy, validating the exact bundle
that will run in the tournament worker, uploading it, and checking whether it scored.

## Current status

- Can do: play and train AmongThem inside the Metta repo through the BitWorld integration.
- Can do: run AmongThem through the native PufferLib path with pixel observations or state observations.
- Can do: validate, upload, submit, and inspect policies through the CoGames CLI once an AmongThem season exists.
- Still external to this local walkthrough: creating the weekly AmongThem season, choosing the season name, and waiting
  for asynchronous tournament matches to finish.

## 1. Create an editable policy

```bash
cogames tutorial make-policy --amongthem -o amongthem_policy.py
```

Edit `AmongThemPolicy._choose_actions()` in the generated file. The template receives raw BitWorld observations and must
write integer action indices from the BitWorld trainable action set into `raw_actions`.

## 2. Log in and pick a season

```bash
cogames login
cogames season list
cogames season show <season>
```

Use the AmongThem season name for every command below.

## 3. Validate the policy bundle

Run the same Docker validation that upload uses, without uploading:

```bash
cogames upload \
  -p class=amongthem_policy.AmongThemPolicy \
  -f amongthem_policy.py \
  -n "$USER-amongthem-practice" \
  --season <season> \
  --dry-run
```

If the policy imports extra local modules, add each file or package directory with another `-f`.

## 4. Upload and submit

```bash
cogames ship \
  -p class=amongthem_policy.AmongThemPolicy \
  -f amongthem_policy.py \
  -n "$USER-amongthem-practice" \
  --season <season>
```

`ship` creates the bundle, validates it in Docker, uploads it, and submits it to the season.

## 5. Watch it score

```bash
cogames submissions --season <season> --policy "$USER-amongthem-practice"
cogames leaderboard <season> --policy "$USER-amongthem-practice"
cogames season matches <season> --limit 20
```

Scores appear after the tournament server finishes the asynchronous episode runs. Use match artifacts when a submitted
policy fails after upload:

```bash
cogames matches <match-id>
cogames match-artifacts <match-id>
```
