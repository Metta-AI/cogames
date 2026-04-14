# Experiment Log

## April 13-14 overnight (autonomous loop)

### v1 (April 13, ~8:32 PM)
- Initial upload, starter template
- Qualifying only (retired)

### v1 → v2 (April 13, ~9:21 PM)
- **Changes**: Hub-relative coords, role system, observation parsing, BFS navigation
- **Leaderboard**: 2.16 VOR, rank 333
- **Local VOR**: 1.13

### v2 → v3 (April 14, ~12:05 AM)
- **Changes**: Bugfixes (aligner stuck-loop, heartless cycle, deposit threshold)
- **Leaderboard**: 3.43 VOR, rank 289 (+59%)
- **Local VOR**: 0.74 (then engine update, recalibrated to 0.30)

### v3 → v4 (April 14, ~2:11 AM)
- **Changes**: Hub inventory awareness (+117% local VOR), `last_action_move` stuck detection, 3M/5A/0S role split, miner element preference, frontier-aware junction targeting
- **Leaderboard**: 4.36 VOR, rank 257 (+27%)
- **Local VOR**: 0.71

### v4 → v5 (April 14, ~3:28 AM)
- **Changes**: Visible-only junction deposits, minor tuning
- **Leaderboard**: 6.03 VOR, rank 224 (+38%)
- **Local VOR**: 0.73

### Benchmark: April 14, 11:55 AM (recovery session)
- **Softy VOR (local)**: 0.81 (3 episodes vs random)
- **8v0 score**: 1.56
- **Note**: Local VOR vs random is much lower than tournament VOR

---
