cycle: 62
last_hypothesis: P161 — claim release on heartless (NOISE)
base_version: v29 (=v24+P154 = P12+P18+P15+P65+P73+P76+P89+P102+P103+P104+P154)
local_vor: ~1.23 (c6), ~0.23 (c2)
consecutive_fails: 6 (P156 revert, P157-P161 all noise)
accumulated_delta: 0% (reset after upload)
wins_since_upload: 0
last_upload: v29 (P154 applied, uploaded April 15)
tournament_horse: v29 (qualifying). Previous: v24 at 25.60, rank 2.
mode: noise_floor (all bugs found are real but below measurable impact)
compound_candidates: []
note: |
  April 15, 2026 (cycle 62):
  - Tournament audit: dinky:v27 #1 at 27.31. Softy:v24 #2 at 25.60. v29 qualifying.
  - DISCOVERY: Game engine uses L2² distance (dr²+dc² ≤ r²), NOT Manhattan.
    Softy uses Manhattan everywhere. This IS a real bug, but practical impact is
    below noise floor on machina_1 map (junctions rarely in mismatch zone).
  - P159 (L2 distance fix + radius 20→15): NOISE. c2 -2.6%, c6 -1.3%.
  - P160 (failed_junctions blacklist reset every 2000 ticks): NOISE. c6 +2.1% p=0.33.
    Marginal positive trend but below threshold.
  - P161 (claim release when aligner loses heart): NOISE. c6 -0.2% p=0.92.
  - Role assignment confirmed safe: IDs always 0..N-1, all created before first step.
  - 6 consecutive fails. Remaining bugs have effects below c6 noise floor (~6%).
  - Need fundamentally new capability or different approach to break through.
