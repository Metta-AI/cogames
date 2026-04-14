# Improvement Hypotheses (Prioritized Queue)

## P1: Add scramblers to counter cascade failure [HIGH — Tier 1]
- **Hypothesis**: Adding 1-2 scramblers prevents cascade disconnection, retaining more net-connected junctions
- **Test**: Compare 2M/1Sc/5A and 2M/2Sc/4A vs current 3M/5A
- **Expected impact**: 2-3x VOR if cascade is a major bleed factor
- **Implementation**: Update ROLE_CYCLE, scrambler role already implemented in softy.py
- **Risk**: Fewer aligners = slower initial expansion. Net effect depends on how much cascade costs.

## P2: Network-aware junction targeting (redundancy bonus) [HIGH — Tier 1]
- **Hypothesis**: Prioritizing junctions that create redundant paths prevents cascade disconnection
- **Test**: Add `redundancy = count(cogs junctions within 25 cells)` to scoring
- **Expected impact**: 30-50% VOR from network resilience
- **Implementation**: Modify `nearest_alignable_junction()` scoring formula

## P3: Clips timing exploitation [MEDIUM-HIGH — Tier 2]
- **Hypothesis**: Timing aligner rushes to the 200-tick safe window after clips act maximizes captures
- **Test**: Add step_count awareness to aligner targeting urgency
- **Expected impact**: 20-40% VOR
- **Implementation**: In `_aligner()`, increase urgency near clips action ticks

## P4: Energy-aware navigation [MEDIUM — Tier 2]
- **Hypothesis**: Routing through friendly territory during night prevents energy starvation
- **Test**: Add territory preference to `_go_absolute()` during night ticks
- **Expected impact**: 10-20% VOR
- **Implementation**: Check `step_count % 200` for day/night, bias toward territory paths

## P5: Hub initial heart rush optimization [MEDIUM — Tier 2]
- **Hypothesis**: Ensuring all 5 aligners grab hearts at tick 0-5 maximizes early captures
- **Test**: Verify no wasted time in aligner startup sequence
- **Expected impact**: 5-15% VOR (front-loaded tick value)
- **Implementation**: Audit aligner early-game behavior

## P6: Heartless exploration cycling [LOW-MEDIUM — Tier 3]
- **Hypothesis**: Heartless aligners cycling through sectors discover more junctions
- **Test**: Advance explore_dir_idx every 5 ticks when heartless
- **Expected impact**: 5-10% VOR from better map coverage
- **Implementation**: Add tick-based rotation in heartless aligner branch

## P7: Scrambler timing at tick ~90 [MEDIUM — depends on P1]
- **Hypothesis**: Pre-position scrambler at border before clips' first action at tick 100
- **Test**: Rush scrambler gear and position near closest border junction
- **Expected impact**: Prevents first cascade event
- **Implementation**: Time-aware behavior in scrambler role
- **Depends on**: P1 (adding scramblers first)

## P8: Dynamic role switching mid-game [SPECULATIVE]
- **Hypothesis**: Miners switch to aligners after economy established (~tick 3000)
- **Test**: Add role-switch logic based on hub inventory + tick count
- **Expected impact**: Unknown — could be transformative or harmful
- **Risk**: Gear switch costs hub resources
